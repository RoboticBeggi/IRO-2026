"""A gallery of candidate voices for ARC: the same two lines in many voices, to pick by ear.

    ~/.venvs/voice/bin/python tools/voice_gallery.py kokoro        (seconds)
    ~/.venvs/qwentts/bin/python tools/voice_gallery.py qwen         (~10 min: built-in Qwen voices, storyteller style)
    ~/.venvs/qwentts/bin/python tools/voice_gallery.py design [butlers-first]   (voices made from a description)
    ~/.venvs/voice/bin/python tools/voice_gallery.py kokoro-butler  (Kokoro's British men on the butler lines)
Listen: tools/listen_voices.sh ~/voice-samples/gallery
Qwen is too slow to speak live on the Jetson (23-36 s per sentence); a chosen Qwen voice is made live by
cloning it into a fast model, or used for pre-recorded lines (the demo opener, object stories).
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

OUT = Path.home() / "voice-samples" / "gallery"
LINES = [
    "Welcome, traveller! I am ARC. Come closer, and I will tell you a story that is a thousand years old.",
    "Behind the white screen, the Tok Dalang brings every puppet to life, and the shadows begin to dance.",
]
STYLE = "Speak like a warm, captivating museum storyteller: calm, expressive, inviting, with gentle pauses."
# "Gentleman AI butler" voices (user, 23 Oct): refined, witty, expressive, a little quirky.
# Characters only: no real person's voice is imitated.
BUTLER_LINES = [
    "Ah, a visitor. Splendid. I do hope you are fond of stories, because I happen to know a rather good one.",
    "Oh, that is magnificent! The Tok Dalang voices every single character. Frankly, I am a little jealous.",
    "Hmm. I am afraid that one is a mystery, even to me. Though, between us, I find mysteries rather delightful.",
]
BUTLERS = {
    "butler-classic": "A refined, articulate British gentleman in his forties, the voice of a loyal and brilliant AI "
                      "butler: calm, polished received pronunciation, dry wit, warm underneath, subtle amusement.",
    "butler-witty": "An elegant British male voice, quick and witty with playful sarcasm, crisp diction, a smile "
                    "in the voice, expressive rises of delight and mock surprise, slightly eccentric.",
    "butler-warm": "A deep, velvety, warm British gentleman, kind and reassuring, softly spoken, emotionally "
                   "expressive, a twinkle of gentle humour, unhurried and charming.",
    "butler-quirky": "A charming, slightly quirky British gentleman with an enthusiastic, theatrical flair, "
                     "precise and posh but bursting with curiosity, little chuckles, lively changes of pace and pitch.",
    "butler-young": "A youthful, bright British male voice, polite and clever, a modern AI assistant with "
                    "cheeky understated humour and genuine warmth, clear and expressive.",
}
DESIGNS = {
    "wise-elder": "A warm, deep, resonant older male voice, a wise and kind storyteller by a fire, slow and "
                  "expressive, with a gentle smile in his voice.",
    "gentle-guide": "A warm, clear young female museum guide, friendly and gentle, smooth and soothing, "
                    "speaking with wonder and enthusiasm.",
    "friendly-robot": "A bright, friendly, curious young voice of a helpful robot companion, cheerful and "
                      "playful but clear, not mechanical.",
    "documentary": "A calm, rich British documentary narrator, mid-aged male, elegant and captivating, "
                   "measured pace.",
}


def save(name, i, a, sr):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    sf.write(d / f"line{i + 1}.wav", np.asarray(a, np.float32), sr)
    print(f"  {name} line {i + 1}", flush=True)


def kokoro_butler():
    """Kokoro's British men, a touch slower, on the butler lines (the fast, live-capable option)."""
    import onnxruntime as ort
    from kokoro_onnx import Kokoro
    m = Path.home() / "voice-models"
    k = Kokoro.from_session(ort.InferenceSession(str(m / "kokoro-v1.0.onnx"),
                                                 providers=["CUDAExecutionProvider", "CPUExecutionProvider"]),
                            str(m / "voices-v1.0.bin"))
    for v in ["bm_george", "bm_daniel", "bm_lewis", "bm_fable"]:
        for i, line in enumerate(BUTLER_LINES):
            a, sr = k.create(line, voice=v, speed=0.92, lang="en-gb")
            save(f"butler-kokoro-{v}", i, a, sr)


def kokoro():
    import onnxruntime as ort
    from kokoro_onnx import Kokoro
    m = Path.home() / "voice-models"
    k = Kokoro.from_session(ort.InferenceSession(str(m / "kokoro-v1.0.onnx"),
                                                 providers=["CUDAExecutionProvider", "CPUExecutionProvider"]),
                            str(m / "voices-v1.0.bin"))
    for v in ["af_bella", "af_nicole", "af_sky", "af_sarah", "am_fenrir", "am_puck", "am_onyx",
              "bf_isabella", "bm_george", "bm_fable", "bm_lewis"]:
        for i, line in enumerate(LINES):
            a, sr = k.create(line, voice=v, speed=0.95, lang="en-gb" if v[0] == "b" else "en-us")
            save(f"kokoro-{v}", i, a, sr)


def qwen():
    import torch
    from qwen_tts import Qwen3TTSModel
    m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", device_map="cuda",
                                      dtype=torch.bfloat16, attn_implementation="sdpa")
    for spk in ["Ryan", "Aiden", "Serena", "Vivian", "Uncle_Fu", "Sohee"]:
        for i, line in enumerate(LINES):
            with torch.inference_mode():
                w, sr = m.generate_custom_voice(text=line, language="English", speaker=spk, instruct=STYLE)
            save(f"qwen-{spk}-storyteller", i, w[0], sr)


def design():
    import torch
    from qwen_tts import Qwen3TTSModel
    m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device_map="cuda",
                                      dtype=torch.bfloat16, attn_implementation="sdpa")
    jobs = [(f"design-{n}", d, LINES) for n, d in DESIGNS.items()]
    if "butlers-first" in sys.argv:
        jobs = [(n, d, BUTLER_LINES) for n, d in BUTLERS.items()] + jobs
    else:
        jobs += [(n, d, BUTLER_LINES) for n, d in BUTLERS.items()]
    for name, desc, lines in jobs:
        if (OUT / name / f"line{len(lines)}.wav").exists():
            continue                                       # already made (re-runs only add new ones)
        for i, line in enumerate(lines):
            with torch.inference_mode():
                w, sr = m.generate_voice_design(text=line, language="English", instruct=desc)
            save(name, i, w[0], sr)


if __name__ == "__main__":
    {"kokoro": kokoro, "kokoro-butler": kokoro_butler, "qwen": qwen, "design": design}[sys.argv[1]]()
