"""Compare text-to-speech voices on the Jetson: speed, memory, and a sample of each to listen to.

    ~/.venvs/voice/bin/python tools/voice_bench.py [kokoro] [chatterbox-turbo] [chatterbox-ml]

Each voice says the same ARC lines; wav files go to ~/voice-samples/<voice>/ and a summary to
~/voice-samples/summary.txt. Listen and pick the most human one (play: aplay FILE).
Measured per line: time to the first audio (what a visitor waits for, with the reply streamed
sentence by sentence) and real-time factor (seconds of compute per second of speech; < 1 = faster
than speaking).
"""

import resource
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

OUT = Path.home() / "voice-samples"
MODELS = Path.home() / "voice-models"
LINES = [
    "Welcome! I am ARC, a heritage storyteller robot. Would you like to hear about shadow puppets?",
    "In wayang kulit, one master puppeteer, the Tok Dalang, moves every puppet and speaks every voice.",
    "Hmm, I am not sure about that. But I can tell you how the puppets are carved from buffalo hide.",
    "Thank you for visiting! Come back soon, and keep the stories alive.",
]


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def kokoro():
    from kokoro_onnx import Kokoro
    import onnxruntime as ort
    sess = ort.InferenceSession(str(MODELS / "kokoro-v1.0.onnx"),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    k = Kokoro.from_session(sess, str(MODELS / "voices-v1.0.bin"))
    voices = ["af_heart", "am_michael", "bf_emma"]   # US female (top rated), US male, UK female

    def say(text, voice):
        a, sr = k.create(text, voice=voice, speed=1.0, lang="en-us")
        return a, sr
    return say, voices


def chatterbox_turbo():
    import torch
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    m = ChatterboxTurboTTS.from_pretrained(device="cuda")

    def say(text, voice):
        with torch.inference_mode():
            w = m.generate(text)
        return w.squeeze().cpu().numpy(), m.sr
    return say, ["default"]


def chatterbox_ml():
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    m = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

    def say(text, voice):
        with torch.inference_mode():
            w = m.generate(text, language_id="en")
        return w.squeeze().cpu().numpy(), m.sr
    return say, ["default"]


def qwen3():
    # its own venv (needs transformers 4.57): ~/.venvs/qwentts/bin/python tools/voice_bench.py qwen3-tts
    import torch
    from qwen_tts import Qwen3TTSModel
    m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", device_map="cuda",
                                      dtype=torch.bfloat16, attn_implementation="sdpa")

    def say(text, voice):
        with torch.inference_mode():
            wavs, sr = m.generate_custom_voice(text=text, language="English", speaker=voice)
        return np.asarray(wavs[0], np.float32), sr
    return say, ["Ryan", "Aiden"]


ENGINES = {"kokoro": kokoro, "chatterbox-turbo": chatterbox_turbo, "chatterbox-ml": chatterbox_ml,
           "qwen3-tts": qwen3}


def main():
    names = sys.argv[1:] or list(ENGINES)
    OUT.mkdir(exist_ok=True)
    summary = []
    for name in names:
        t0 = time.time()
        say, voices = ENGINES[name]()
        load = time.time() - t0
        say("Warming up.", voices[0])                 # first call compiles / allocates
        for v in voices:
            d = OUT / f"{name}-{v}"
            d.mkdir(exist_ok=True)
            firsts, rtfs = [], []
            for i, line in enumerate(LINES):
                t = time.time()
                a, sr = say(line, v)
                dt = time.time() - t
                first = line.split(".")[0].split("!")[0].split("?")[0] + "."
                t = time.time()
                say(first, v)                          # latency of a first sentence (streamed replies)
                firsts.append(time.time() - t)
                rtfs.append(dt / (len(a) / sr))
                sf.write(d / f"line{i + 1}.wav", np.asarray(a, np.float32), sr)
            row = (f"{name}-{v:12s} load {load:5.1f}s  first-sentence {np.median(firsts):4.2f}s  "
                   f"RTF {np.median(rtfs):4.2f}  peak RAM {rss_mb():5.0f} MB  -> {d}")
            print(row, flush=True)
            summary.append(row)
    with open(OUT / "summary.txt", "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M\n") + "\n".join(summary) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
