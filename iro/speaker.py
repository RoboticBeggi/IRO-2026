"""Who is speaking: ARC's owner or a visitor? (speaker verification by voice)

    iro voice enroll       teach ARC your voice: read 6 short sentences aloud (~1 min)
    iro voice whoami       say something; ARC tells you whether it thinks you are the owner, and how sure

How it works: a speaker-embedding network (WeSpeaker ResNet34, trained on the VoxCeleb celebrity
interviews: 7000+ speakers) turns a few seconds of speech into 256 numbers that describe the VOICE,
not the words. Your enrollment sentences give a set of these; any new sentence is compared with
them (cosine similarity). Above the threshold set at enrollment = the owner. Runs on the CPU in
a few tens of milliseconds per sentence (sherpa-onnx), so it costs no GPU memory.
Short sentences (< 1 s) and noise are not judged: those count as "visitor".
The voiceprint is personal data: it stays on the robot (config/owner_voice.npz is NOT in git).
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from iro import ROOT

MODEL = Path.home() / "voice-models" / "wespeaker_en_voxceleb_resnet34.onnx"
PRINT = ROOT / "config" / "owner_voice.npz"
RATE = 16000
SENTENCES = [
    "Good morning ARC, it is me again, ready for another day of work.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Please tell the visitors about wayang kulit and the shadow puppets.",
    "I built you to keep old stories alive, so do not let me down.",
    "Seven hundred and forty two robots walked into a museum in Vienna.",
    "Okay, that is enough for now, let us test whether you know my voice.",
]


class Ears:
    """Speaker embeddings (CPU)."""

    def __init__(self):
        import sherpa_onnx
        if not MODEL.exists():
            raise SystemExit(f"missing {MODEL}: see iro/speaker.py (sherpa-onnx speaker-recongition-models)")
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(MODEL), num_threads=4, provider="cpu")
        self.x = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def embed(self, pcm16):
        s = self.x.create_stream()
        s.accept_waveform(RATE, (np.asarray(pcm16, np.float32) / 32768.0))
        s.input_finished()
        v = np.array(self.x.compute(s), np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


class Owner:
    """The enrolled owner's voiceprint; is_owner(pcm) -> (bool, score)."""

    def __init__(self, ears=None):
        self.ears = ears or Ears()
        self.ok = PRINT.exists()
        if self.ok:
            d = np.load(PRINT)
            # 24 Oct: the owner scored 0.81 in a real conversation and was missed (enrolled threshold
            # 0.85); other voices stayed <= 0.70, so 0.75 still separates them
            self.prints, self.threshold = d["prints"], min(float(d["threshold"]), 0.75)

    def score(self, pcm16):
        v = self.ears.embed(pcm16)
        return float(np.max(self.prints @ v))

    def learn(self, pcm16, score, sure=0.82, keep=40):
        """Add a turn the owner surely spoke (score >= sure) to the voiceprint: it keeps up with how he
        sounds in real conversations (distance, mood, tiredness). Newest `keep` are kept."""
        if not self.ok or score < sure or len(pcm16) < RATE * 1.5:
            return
        self.prints = np.vstack([self.prints, self.ears.embed(pcm16)])[-keep:]
        np.savez(PRINT, prints=self.prints, threshold=self.threshold, date=time.strftime("%Y-%m-%d %H:%M"))

    def is_owner(self, pcm16, min_s=1.0):
        if not self.ok or len(pcm16) < RATE * min_s:
            return False, 0.0
        s = self.score(pcm16)
        return s >= self.threshold, s


def array_source():
    """The mic array's PulseAudio input. The desktop's PulseAudio holds the sound card, so arecord on
    plughw is refused ("Device or resource busy", 23 Oct): record through PulseAudio instead."""
    out = subprocess.run(["pactl", "list", "short", "sources"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "XVF3800" in line and ".monitor" not in line:
            return line.split("\t")[1]
    raise SystemExit("the reSpeaker mic array is not an input: is it plugged in? (iro doctor)")


VAD_MODEL = Path.home() / "voice-models" / "silero_vad.onnx"


def mic_channel():
    """Which array output to listen to: 1 = right = the ASR beam (Seeed: made for speech recognition),
    0 = left = the conference beam (heavier noise suppression). `iro mic test` measures and chooses."""
    from iro.voice import cfg
    return int(cfg().get("mic_channel", 1))


class Stream:
    """16 kHz audio from the array through PulseAudio (it holds the card; arecord is refused).
    Both channels arrive; read(n) returns n samples of the chosen one (or both with both=True)."""

    def __init__(self, channel=None):
        self.ch = mic_channel() if channel is None else channel
        self.p = subprocess.Popen(["parec", f"--device={array_source()}", f"--rate={RATE}", "--channels=2",
                                   "--format=s16le", "--raw", "--latency-msec=30"], stdout=subprocess.PIPE)
        self.stdout = self                        # old interface: .stdout.read(nbytes) of mono
        self.read_both(int(RATE * 0.3))           # the first moments after the input wakes are silent

    def read_both(self, n):
        raw = self.p.stdout.read(n * 4)
        return np.frombuffer(raw, np.int16).reshape(-1, 2)

    def read(self, nbytes):
        return self.read_both(nbytes // 2)[:, self.ch].copy().tobytes()

    def terminate(self):
        self.p.terminate()


def stream():
    return Stream()


def record(seconds, both=False):
    p = Stream()
    a = p.read_both(int(RATE * seconds))
    p.terminate()
    return a if both else a[:, p.ch].copy()


class Vad:
    """Silero VAD (sherpa-onnx, CPU): speech segments from a stream of 16 kHz samples. Much better than
    loudness at telling speech from noise, music and clatter (the standard in open voice assistants)."""

    def __init__(self, threshold=0.5, min_silence=0.6, min_speech=0.3, max_speech=15.0):
        import sherpa_onnx
        c = sherpa_onnx.VadModelConfig()
        c.silero_vad.model = str(VAD_MODEL)
        c.silero_vad.threshold = threshold
        c.silero_vad.min_silence_duration = min_silence
        c.silero_vad.min_speech_duration = min_speech
        c.silero_vad.max_speech_duration = max_speech
        c.sample_rate = RATE
        self.win = c.silero_vad.window_size
        self.v = sherpa_onnx.VoiceActivityDetector(c, buffer_size_in_seconds=60)
        self.buf = np.zeros(0, np.float32)

    def feed(self, pcm16):
        """Feed samples; returns finished speech segments (int16 arrays)."""
        self.buf = np.concatenate([self.buf, np.asarray(pcm16, np.float32) / 32768.0])
        while len(self.buf) >= self.win:
            self.v.accept_waveform(self.buf[:self.win])
            self.buf = self.buf[self.win:]
        out = []
        while not self.v.empty():
            out.append((np.asarray(self.v.front.samples) * 32768).astype(np.int16))
            self.v.pop()
        return out

    def speaking(self):
        return self.v.is_speech_detected()

    def reset(self):
        self.v.reset()
        self.buf = np.zeros(0, np.float32)


def loudest_rms(a, win=RATE):
    """RMS of the loudest 1 s: speech level, not room noise."""
    if len(a) < win:
        return 0.0
    x = a.astype(np.float32) ** 2
    c = np.convolve(x, np.ones(win) / win, mode="valid")
    return float(np.sqrt(c.max()))


def cmd_enroll():
    from iro.record import run
    r = run("voice-enroll")
    ears = Ears()
    print("Teaching ARC your voice. Stand where you usually talk to it (about 1 m away), speak normally.\n"
          "Each sentence: press Enter, then read it aloud (you have 5 seconds).\n")
    embs = []
    for i, s in enumerate(SENTENCES, 1):
        input(f"  {i}/{len(SENTENCES)}  Enter, then say: \"{s}\" ")
        a = record(5)
        loud = loudest_rms(a)
        if len(a) < RATE * 2 or loud < 600:
            print(f"    (heard nothing clear, level {loud:.0f}: that one is skipped. Speak up or come closer)")
            continue
        print(f"    ok (level {loud:.0f})")
        embs.append(ears.embed(a))
        import wave
        with wave.open(str(r.dir / f"enroll_{i}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(a.tobytes())
    if len(embs) < 4:
        print("Too few usable recordings (need 4): try again a bit closer to the mic array.")
        return 1
    E = np.array(embs)
    # leave-one-out: how similar each of your sentences is to the others = how "you" scores
    loo = [float(np.max(np.delete(E, i, 0) @ E[i])) for i in range(len(E))]
    # other voices scored up to 0.68 against each other in the 23 Oct test: never go below 0.70
    threshold = float(np.clip(min(loo) - 0.08, 0.70, 0.85))
    np.savez(PRINT, prints=E, threshold=threshold, loo=np.array(loo), date=time.strftime("%Y-%m-%d %H:%M"))
    r.note(f"enrolled {len(E)} sentences; self-similarity {np.round(loo, 2).tolist()}; threshold {threshold:.2f}")
    print(f"\nLearned your voice from {len(E)} sentences (your own similarity {min(loo):.2f}-{max(loo):.2f}); "
          f"owner threshold {threshold:.2f}.\nTest it: iro voice whoami  (and ask someone else to try too)")
    return 0


def cmd_whoami():
    o = Owner()
    if not o.ok:
        print("No owner voice yet: iro voice enroll")
        return 1
    input("Enter, then say any sentence (4 seconds): ")
    a = record(4)
    yes, s = o.is_owner(a)
    print(f"{'OWNER (hello, sir)' if yes else 'visitor'}: similarity {s:.2f} (owner if >= {o.threshold:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit({"enroll": cmd_enroll, "whoami": cmd_whoami}.get((sys.argv[1:] or ["?"])[0], lambda: print(__doc__))())
