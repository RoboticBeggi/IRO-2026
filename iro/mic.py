"""reSpeaker XVF3800 control over USB: where speech comes from, how strong it is, the LED ring.

    python -m iro.mic          print direction / speech / energy 5x a second (talk around it)

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


def main():
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
