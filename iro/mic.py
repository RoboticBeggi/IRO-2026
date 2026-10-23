"""reSpeaker XVF3800 control over USB: where speech comes from, how strong it is, the LED ring.

    python -m iro.mic          print direction / speech / energy 5x a second (talk around it)
    iro mic test               which output hears your voice best (levels, SNR, VAD, Whisper); saves it

The array does the hard work on its own chip (beamforming, echo cancelling, noise suppression,
direction of arrival). Over USB vendor control transfers (Seeed's xvf_host.py protocol) we read:
  DOA_VALUE                    angle 0-359 deg + "speech detected" flag
  AUDIO_MGR_SELECTED_AZIMUTHS  processed speaker direction (rad, NaN when nobody speaks) and the
                               auto-select beam's direction (firmware >= 2.0.1)
  AEC_SPENERGY_VALUES          speech energy per beam (> 0 = speech; louder/closer = higher)
Reads sometimes return the previous command's bytes (seen 23 Oct, firmware 2.0.6), so every value
is range-checked and re-read.
"""

import math
import struct
import sys
import time

import numpy as np

VID, PID = 0x2886, 0x001A
# name: (resid, cmdid, count, struct format)
PARAMS = {
    "VERSION": (48, 0, 3, "B"),
    "DOA_VALUE": (20, 18, 2, "H"),
    "SELECTED_AZIMUTHS": (35, 11, 2, "f"),
    "AEC_AZIMUTH_VALUES": (33, 75, 4, "f"),
    "AEC_SPENERGY_VALUES": (33, 80, 4, "f"),
    "LED_EFFECT": (20, 12, 1, "B"),
    "LED_COLOR": (20, 16, 1, "I"),
    "LED_DOA_COLOR": (20, 17, 2, "I"),
}


class MicArray:
    def __init__(self):
        import usb.core
        import usb.util
        self.usb = usb
        self.dev = usb.core.find(idVendor=VID, idProduct=PID)
        if self.dev is None:
            raise RuntimeError("reSpeaker XVF3800 not found on USB")

    def _read(self, name):
        resid, cmd, n, fmt = PARAMS[name]
        size = struct.calcsize(fmt)
        u = self.usb.util
        for _ in range(50):
            r = self.dev.ctrl_transfer(u.CTRL_IN | u.CTRL_TYPE_VENDOR | u.CTRL_RECIPIENT_DEVICE,
                                       0, 0x80 | cmd, resid, n * size + 1, 1000)
            if r[0] == 0:
                return struct.unpack("<" + fmt * n, bytes(r[1:1 + n * size]))
            time.sleep(0.005)                     # 64 = "busy, retry"
        raise TimeoutError(f"{name}: no answer")

    def write(self, name, values):
        resid, cmd, n, fmt = PARAMS[name]
        u = self.usb.util
        self.dev.ctrl_transfer(u.CTRL_OUT | u.CTRL_TYPE_VENDOR | u.CTRL_RECIPIENT_DEVICE,
                               0, cmd, resid, struct.pack("<" + fmt * n, *values), 1000)

    def read(self, name, valid, tries=4):
        """A value that passes `valid` (rejects bytes left over from another command)."""
        for _ in range(tries):
            v = self._read(name)
            if valid(v):
                return v
            time.sleep(0.003)
        return None

    def sample(self):
        """{t, doa_deg, speech, speaker_deg (None if nobody speaks), energy, dirs} or None.
        dirs: the three direction readings (deg, None when not speaking): speaker (processed),
        doa (DOA_VALUE while speech is detected), free (free-running beam)."""
        doa = self.read("DOA_VALUE", lambda v: v[0] < 360 and v[1] in (0, 1))
        sel = self.read("SELECTED_AZIMUTHS",
                        lambda v: all(math.isnan(x) or 0 <= x <= 2 * math.pi + 1e-3 for x in v))
        en = self.read("AEC_SPENERGY_VALUES", lambda v: all(0 <= x < 1e9 for x in v))
        az = self.read("AEC_AZIMUTH_VALUES", lambda v: all(math.isnan(x) or 0 <= x <= 2 * math.pi + 1e-3 for x in v))
        if doa is None:
            return None
        spk = None if sel is None or math.isnan(sel[0]) else math.degrees(sel[0]) % 360
        # three direction readings; `iro visitors mic-cal` measures which follows a person best
        # (23 Oct: the processed one held 346.7 deg for 5 s while the person walked 80 deg)
        free = None
        if az is not None and en is not None and en[2] > 0 and not math.isnan(az[2]):
            free = math.degrees(az[2]) % 360          # the free-running beam: follows the loudest voice
        return {"t": time.time(), "doa_deg": float(doa[0]), "speech": bool(doa[1]),
                "speaker_deg": spk, "energy": float(max(en[:2])) if en else 0.0,
                "dirs": {"speaker": spk, "doa": float(doa[0]) if doa[1] else None, "free": free}}

    def led_doa(self):
        """LED ring points to the voice (judges can see it hear)."""
        self.write("LED_EFFECT", [4])

    def led_color(self, rgb):
        self.write("LED_EFFECT", [3])
        self.write("LED_COLOR", [rgb])

    def close(self):
        self.usb.util.dispose_resources(self.dev)


def cmd_test(seconds=6):
    """Which array output hears YOUR voice best? Records both channels while you talk, then compares
    speech level, noise, signal-to-noise, Silero VAD speech time and Whisper's transcript; saves the
    winner as mic_channel in config/voice.yaml (used by iro arc talk and iro voice enroll)."""
    from iro.speaker import RATE, Vad, record
    from iro.voice import save_cfg
    input(f"Mic test: stand where visitors stand (~1 m). Enter, then talk normally for {seconds} s "
          "(for example: describe your day). ")
    both = record(seconds, both=True)
    names = {0: "left (conference beam)", 1: "right (ASR beam)"}
    results = {}
    try:
        from iro.talk import start_whisper, transcribe
        start_whisper()
    except Exception:                              # noqa: BLE001 - levels alone still help
        transcribe = None
    for ch in (0, 1):
        a = both[:, ch].copy()
        fr = a[: len(a) // 320 * 320].astype(np.float32).reshape(-1, 320)
        r = np.sqrt((fr ** 2).mean(1))
        noise, speech = np.percentile(r, 10), np.percentile(r, 90)
        snr = 20 * np.log10(max(speech, 1) / max(noise, 1))
        v = Vad()
        segs = v.feed(a) + []
        v.v.flush()
        while not v.v.empty():
            segs.append(np.asarray(v.v.front.samples))
            v.v.pop()
        talk_s = sum(len(x) for x in segs) / RATE
        text = transcribe(a)[0] if transcribe else ""
        results[ch] = (snr, talk_s)
        print(f"  {names[ch]:24s} speech {speech:6.0f}  noise {noise:5.0f}  SNR {snr:4.1f} dB  "
              f"speech found {talk_s:3.1f}s  heard: {text[:70]!r}")
    best = max(results, key=lambda c: (results[c][1] > 1.0, results[c][0]))
    save_cfg(mic_channel=int(best))
    print(f"Using the {names[best]} from now on (config/voice.yaml mic_channel: {best}).")
    if results[best][1] < 1.0:
        print("  Hardly any speech was found: speak louder or closer, and check the red Mute LED is off.")
    return 0


def main():
    if sys.argv[1:2] == ["test"]:
        return cmd_test()
    m = MicArray()
    print("firmware", ".".join(map(str, m._read("VERSION"))))
    m.led_doa()
    try:
        while True:
            s = m.sample()
            if s:
                spk = f"{s['speaker_deg']:5.0f}" if s["speaker_deg"] is not None else "  -  "
                print(f"doa {s['doa_deg']:5.0f}  speech {int(s['speech'])}  speaker {spk}  energy {s['energy']:10.0f}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
