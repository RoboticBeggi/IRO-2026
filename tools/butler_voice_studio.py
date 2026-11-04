#!/usr/bin/env python3
"""Design ARC's own "AI butler" voice (an ORIGINAL character, not any actor): many takes from Qwen's
VoiceDesign model, each also with a subtle "AI sheen" effect, to pick by ear.

    ~/iro-2026/tools/butler_voice_studio.py            make the takes (~10 min, GPU)
    ~/iro-2026/tools/butler_voice_studio.py play       listen: natural then AI-sheen version of each
Keep the best:  iro voice keep <take> jarvis   (e.g. iro voice keep velvet-wit-2 jarvis)

Prompting follows the Qwen3-TTS voice-design guidance: concrete audible traits over several
dimensions (age, accent, timbre, pitch, pace, emotion, role), 30-60 words, no celebrity names.
The AI sheen follows film AI-voice sound design: keep diction clear; high-pass, a cut in the
low-mids (mud), a presence lift ~3 kHz, a touch of "air", gentle compression, a very short room
and a faint chorus for the slightly synthetic shimmer.
"""

import subprocess
import sys
from pathlib import Path

QT = Path.home() / "src" / "qwentts.cpp"
OUT = Path.home() / "voice-samples" / "qwen"           # where `iro voice keep` looks for takes
LINE = ("Good evening, sir. All systems are running, the coffee is not, and I have taken the liberty "
        "of reading the entire history of Malaysia while you slept. Rather fascinating, actually.")
VOICES = {
    "polished-butler": "Male, late thirties, British received pronunciation. Smooth mid-low baritone with crisp, "
                       "precise diction. Calm, measured pace, unflappable and courteous, with understated dry wit "
                       "and quiet warmth. The voice of a brilliant, loyal AI butler.",
    "velvet-wit": "Male, around forty, refined southern English accent. Warm, velvety mid-range voice, slightly "
                  "breathy. Relaxed and articulate, gently amused, a subtle smile and polite sarcasm in every "
                  "line. Intelligent, charming and reassuring.",
    "crisp-intellect": "Male, mid thirties, precise British accent. Bright, clean baritone with very clear "
                       "articulation, quick but controlled pace. Composed and highly intelligent, faintly "
                       "ironic, with a warm undertone.",
    "deep-calm": "Male, mid forties, British accent. Deep, resonant, soft-spoken baritone, slow and "
                 "deliberate pace. Composed and soothing, unhurried authority, dry humour delivered gently.",
    "playful-butler": "Male, early thirties, English accent. Lively, clear baritone with expressive intonation, "
                      "small rises of delight and mock surprise. Witty, charming, quick-thinking and polite, "
                      "a little cheeky.",
    "elegant-mature": "Male, early fifties, cultivated British accent. Smooth, slightly gravelly mid-low voice. "
                      "Elegant, wise and calm, a gentle dry humour and genuine warmth, perfectly articulated.",
}
SEEDS = [11, 22, 33]
FX = ["gain", "-6", "highpass", "90", "equalizer", "300", "1q", "-3", "equalizer", "3200", "1q", "+3",
      "equalizer", "9000", "2q", "+2", "compand", "0.02,0.15", "6:-70,-60,-20", "-4", "-90", "0.05",
      "chorus", "0.8", "0.9", "30", "0.18", "0.3", "1.5", "-t",
      "reverb", "12", "50", "25", "100", "0", "0", "gain", "-n", "-1"]


def make():
    OUT.mkdir(parents=True, exist_ok=True)
    model = QT / "models" / "qwen-talker-1.7b-voicedesign-Q8_0.gguf"
    codec = QT / "models" / "qwen-tokenizer-12hz-Q8_0.gguf"
    for name, desc in VOICES.items():
        for k, seed in enumerate(SEEDS, 1):
            take = OUT / f"{name}-{k}.wav"
            if not take.exists():
                p = subprocess.run([str(QT / "build" / "qwen-tts"), "--model", str(model), "--codec", str(codec),
                                    "--instruct", desc, "--lang", "English", "--seed", str(seed), "-o", str(take)],
                                   input=LINE, text=True, capture_output=True)
                if p.returncode:
                    print(f"  {take.name}: failed: {p.stderr[-300:]}")
                    continue
                (OUT / f"{name}-{k}.yaml").write_text(f"description: \"{desc}\"\nseed: {seed}\ntext: \"{LINE}\"\n")
            subprocess.run(["sox", str(take), str(OUT / f"{name}-{k}-ai.wav"), *FX], check=False)
            print(f"  {name}-{k}", flush=True)


def play():
    sink = "alsa_output.platform-3510000.hda.hdmi-stereo"
    subprocess.run(["pactl", "set-sink-mute", sink, "0"], check=False)
    for name in VOICES:
        for k in range(1, len(SEEDS) + 1):
            for suffix, label in (("", "natural"), ("-ai", "AI sheen")):
                f = OUT / f"{name}-{k}{suffix}.wav"
                if f.exists():
                    print(f"  {name}-{k}  ({label})", flush=True)
                    subprocess.run(["paplay", f"--device={sink}", str(f)], check=False)
        input(f"=== that was {name}. Enter for the next voice ")


if __name__ == "__main__":
    play() if sys.argv[1:2] == ["play"] else make()
