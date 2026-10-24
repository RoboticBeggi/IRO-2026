#!/usr/bin/env python3
"""Give ARC's voice moods: the same designed voice (description + seed of the kept take) told to feel
amused / excited / concerned, several takes each; the take whose speaker fingerprint is closest to
the main voice wins (so the MOOD changes, not the person). Kept as voices/<voice>-<mood>/.

    ~/.venvs/... python tools/voice_moods.py [VOICE]      (default: the voice ARC uses)

Live: ARC tags each reply with a mood; iro arc talk speaks it with that mood's reference clip
(cloning copies the style of its reference, so the clip carries the emotion).
"""

import shutil
import subprocess
import sys
import wave

import numpy as np
import yaml

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from iro.speaker import Ears  # noqa: E402
from iro.voice import BIN, MODELS, SCRATCH, VOICES, cfg  # noqa: E402

# Emotion through the CLONED voice (identity kept, ~0.9): VoiceDesign + an emotion clause made a
# different person (0.61-0.71, 23 Oct). The clone takes its delivery from the words and punctuation,
# so each mood has an emotionally written line; several seeds at a livelier temperature, and the
# take that is still the same voice (>= 0.85) with the most pitch/loudness movement wins.
MOODS = {
    "amused": "Ha! Oh, that is rather funny, sir. I shall pretend I did not hear it... but I absolutely did.",
    "excited": "Oh, this is wonderful! A real wayang kulit puppet, right here in front of us! I can hardly wait!",
    "concerned": "Hmm. I am a little worried, sir... The arm is getting hot. Please, let it rest for a moment.",
    "warm": "Ah, it is so good to see you again. Come, sit with me for a moment, and I will tell you a lovely story.",
    "curious": "Oh? Now that is interesting... Where did you find it? Tell me more, I simply must know.",
    "proud": "There. Every single puppet, carved by hand, and the story kept alive for a thousand years. Magnificent.",
}
SEEDS = [1, 2, 3, 4, 5, 6]


def load16(path):
    with wave.open(str(path)) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        sr = w.getframerate()
    return np.interp(np.linspace(0, len(a) - 1, int(len(a) * 16000 / sr)), np.arange(len(a)), a).astype(np.int16)


def liveliness(a, sr=16000):
    """How much the voice moves: spread of pitch (semitones) + spread of loudness (dB) over voiced frames."""
    x = a.astype(np.float32)
    n, hop = 640, 320
    f0, db = [], []
    for i in range(0, len(x) - n, hop):
        fr = x[i:i + n] - x[i:i + n].mean()
        e = float(np.sqrt((fr ** 2).mean()))
        if e < 300:
            continue
        ac = np.correlate(fr, fr, "full")[n - 1:]
        lo, hi = sr // 400, sr // 70                # 70-400 Hz voices
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.3 * ac[0]:
            f0.append(12 * np.log2(sr / k))
            db.append(20 * np.log10(e))
    if len(f0) < 10:
        return 0.0
    return float(np.std(f0) + 0.5 * np.std(db))


def main():
    voice = sys.argv[1] if len(sys.argv) > 1 else cfg()["voice"]
    d0 = VOICES / voice
    ears = Ears()
    ref = ears.embed(load16(d0 / "ref.wav"))
    c = cfg()
    print(f"main voice liveliness {liveliness(load16(d0 / 'ref.wav')):.2f}")
    for mood, line in MOODS.items():
        if (VOICES / f"{voice}-{mood}" / "ref.rvq").exists():
            print(f"{mood}: already made (delete voices/{voice}-{mood} to redo)")
            continue
        best = None
        for seed in SEEDS:
            out = SCRATCH / f"{voice}-{mood}-{seed}.wav"
            subprocess.run([str(BIN / "qwen-tts"), "--model", str(MODELS / c["base"]), "--codec", str(MODELS / c["codec"]),
                            "--ref-spk", str(d0 / "ref.spk"), "--ref-rvq", str(d0 / "ref.rvq"), "--ref-text",
                            str(d0 / "ref.txt"), "--lang", "English", "--seed", str(seed), "--temp", "1.0",
                            "-o", str(out)], input=line, text=True, capture_output=True, check=True)
            a = load16(out)
            sim, live = float(ears.embed(a) @ ref), liveliness(a)
            ok = sim >= 0.85
            print(f"  {mood:9s} seed {seed}: same voice {sim:.2f}  liveliness {live:.2f}{'' if ok else '  (drifted)'}",
                  flush=True)
            if ok and (best is None or live > best[0]):
                best = (live, out, seed, sim)
        if best is None:
            print(f"{mood}: no take stayed the same voice; skipped")
            continue
        live, out, seed, sim = best
        d = VOICES / f"{voice}-{mood}"
        d.mkdir(exist_ok=True)
        shutil.copy(out, d / "ref.wav")
        (d / "ref.txt").write_text(line)
        (d / "about.yaml").write_text(yaml.safe_dump({"description": f"{voice}, {mood} (cloned, seed {seed})",
                                                      "seed": seed, "text": line, "mood_of": voice,
                                                      "same_voice": round(sim, 3), "liveliness": round(live, 2)},
                                                     sort_keys=False))
        subprocess.run([str(BIN / "qwen-codec"), "--model", str(MODELS / c["codec"]), "--talker",
                        str(MODELS / c["base"]), "-i", str(d / "ref.wav")], check=True, capture_output=True)
        print(f"{mood}: kept seed {seed} (same voice {sim:.2f}, liveliness {live:.2f}) -> voices/{voice}-{mood}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
