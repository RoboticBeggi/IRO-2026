"""ARC's voice: Qwen3-TTS (qwentts.cpp on the GPU). You choose it; ARC always speaks in it.

    iro voice presets ["style"]        hear Qwen's 9 built-in voices (optionally with a style)
    iro voice design NAME "describe the voice" [--takes 3]
                                       Qwen Voice Design makes takes from your description
    iro voice keep TAKE [NAME]         keep a take you like as a voice (e.g. iro voice keep butler-2 jeeves)
    iro voice list                     your voices, * = the one ARC uses
    iro voice play NAME                hear it
    iro voice use NAME                 ARC speaks in this voice from now on
    iro voice say "text" [--voice NAME]   say anything in a voice (shows how long it took)

How a voice stays the same every time: a kept take becomes a reference clip (voices/NAME/ref.wav +
its words). Qwen's Base model clones it for every sentence ("in-context" cloning with the speaker
embedding and the clip's audio codes, pre-extracted by qwen-codec), so ARC never drifts into a
different voice. Built-in voices can also be used directly (`use preset:ryan`).
Takes and previews are scratch files in ~/voice-samples/qwen; kept voices live in the repo (voices/).
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from iro import ROOT

QT = Path(os.path.expanduser("~/src/qwentts.cpp"))
BIN, MODELS = QT / "build", QT / "models"
VOICES = ROOT / "voices"
SCRATCH = Path.home() / "voice-samples" / "qwen"
CFG = ROOT / "config" / "voice.yaml"
PRESETS = ["ryan", "aiden", "serena", "vivian", "uncle_fu", "sohee", "ono_anna", "eric", "dylan"]
SAMPLE = ("Ah, a visitor. Splendid. I do hope you are fond of stories, because I happen to know a rather "
          "good one.")
DEFAULTS = {
    "voice": None,
    "output": "alsa_output.platform-3510000.hda.hdmi-stereo",     # the HDMI screen (23 Oct)
    "lang": "English",
    "codec": "qwen-tokenizer-12hz-Q8_0.gguf",
    "customvoice": "qwen-talker-0.6b-customvoice-Q8_0.gguf",
    "voicedesign": "qwen-talker-1.7b-voicedesign-Q8_0.gguf",
    "base": "qwen-talker-0.6b-base-Q8_0.gguf",
}


def cfg():
    c = dict(DEFAULTS)
    if CFG.exists():
        c.update(yaml.safe_load(CFG.read_text()) or {})
    return c


def save_cfg(**kw):
    c = cfg()
    c.update(kw)
    CFG.write_text(yaml.safe_dump(c, sort_keys=False))


def tts(text, out, model, *args):
    """Run qwen-tts once; returns seconds taken. Raises with its message if it fails."""
    c = cfg()
    cmd = [str(BIN / "qwen-tts"), "--model", str(MODELS / c[model]), "--codec", str(MODELS / c["codec"]),
           "--lang", c["lang"], "-o", str(out), *map(str, args)]
    t0 = time.time()
    p = subprocess.run(cmd, input=text, text=True, capture_output=True)
    if p.returncode != 0 or not Path(out).exists():
        raise RuntimeError(f"qwen-tts failed: {(p.stderr or p.stdout).strip()[-400:]}")
    return time.time() - t0


def seconds(wav):
    import wave                                    # qwen-tts writes 16-bit WAV (wav16) by default
    with wave.open(str(wav)) as w:
        return w.getnframes() / w.getframerate()


def play(wav):
    subprocess.run(["paplay", f"--device={cfg()['output']}", str(wav)], check=False)


def report(wav, dt):
    s = seconds(wav)
    print(f"    {dt:.1f} s to make {s:.1f} s of speech ({dt / s:.2f}x real time{', faster than speaking' if dt < s else ''})")


# ------------------------------------------------------------------ choosing

def cmd_presets(style=None):
    d = SCRATCH / "presets"
    d.mkdir(parents=True, exist_ok=True)
    for spk in PRESETS:
        out = d / f"{spk}.wav"
        args = ["--speaker", spk] + (["--instruct", style] if style else [])
        print(f"  {spk}")
        dt = tts(SAMPLE, out, "customvoice", *args)
        report(out, dt)
        play(out)
    print("Use one directly: iro voice use preset:NAME  (or design your own: iro voice design ...)")
    return 0


def cmd_design(name, description, takes=3):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for k in range(1, takes + 1):
        out = SCRATCH / f"{name}-{k}.wav"
        print(f"  take {name}-{k}")
        dt = tts(SAMPLE, out, "voicedesign", "--instruct", description, "--seed", 1000 + k)
        (SCRATCH / f"{name}-{k}.yaml").write_text(yaml.safe_dump(
            {"description": description, "seed": 1000 + k, "text": SAMPLE}, sort_keys=False))
        report(out, dt)
        play(out)
    print(f"Keep the one you like: iro voice keep {name}-2  (or design again with other words)")
    return 0


def cmd_keep(take, name=None):
    """A take (or a preset preview) becomes a voice: ref.wav + its words + clone latents."""
    name = name or take.rsplit("-", 1)[0]
    src = SCRATCH / f"{take}.wav"
    meta_f = SCRATCH / f"{take}.yaml"
    if not src.exists():
        src, meta_f = SCRATCH / "presets" / f"{take}.wav", None
    if not src.exists():
        raise SystemExit(f"no take called {take} in {SCRATCH}")
    meta = yaml.safe_load(meta_f.read_text()) if meta_f and meta_f.exists() else {"text": SAMPLE, "preset": take}
    d = VOICES / name
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, d / "ref.wav")
    (d / "ref.txt").write_text(meta["text"])
    (d / "about.yaml").write_text(yaml.safe_dump({**meta, "kept_from": take, "date": time.strftime("%Y-%m-%d %H:%M")},
                                                 sort_keys=False))
    c = cfg()
    subprocess.run([str(BIN / "qwen-codec"), "--model", str(MODELS / c["codec"]), "--talker", str(MODELS / c["base"]),
                    "-i", str(d / "ref.wav")], check=True, capture_output=True)
    print(f"Kept as voice '{name}' (voices/{name}). Make ARC use it: iro voice use {name}")
    return 0


def cmd_list():
    cur = cfg()["voice"]
    names = sorted(p.name for p in VOICES.iterdir() if p.is_dir()) if VOICES.exists() else []
    for n in names:
        about = yaml.safe_load((VOICES / n / "about.yaml").read_text()) or {}
        desc = about.get("description") or f"built-in voice {about.get('preset', '')}"
        print(f"{'*' if n == cur else ' '} {n:16s} {desc[:90]}")
    if cur and cur.startswith("preset:"):
        print(f"* {cur}")
    if not names and not cur:
        print("No voices yet: iro voice presets / iro voice design NAME \"...\"")
    return 0


def say(text, voice=None, out=None):
    """Make `text` in a voice; returns (wav path, seconds taken)."""
    voice = voice or cfg()["voice"]
    if not voice:
        raise SystemExit("no voice chosen yet: iro voice use NAME")
    out = Path(out or SCRATCH / "say.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    if voice.startswith("preset:"):
        dt = tts(text, out, "customvoice", "--speaker", voice.split(":", 1)[1])
    else:
        d = VOICES / voice
        spk, rvq = d / "ref.spk", d / "ref.rvq"
        dt = tts(text, out, "base", "--ref-spk", spk, "--ref-rvq", rvq, "--ref-text", d / "ref.txt")
    return out, dt


def main():
    a = sys.argv[1:]
    sub = a[0] if a else "list"
    if sub == "presets":
        return cmd_presets(a[1] if len(a) > 1 else None)
    if sub == "design" and len(a) >= 3:
        takes = int(a[a.index("--takes") + 1]) if "--takes" in a else 3
        return cmd_design(a[1], a[2], takes)
    if sub == "keep" and len(a) >= 2:
        return cmd_keep(a[1], a[2] if len(a) > 2 else None)
    if sub == "list":
        return cmd_list()
    if sub == "play" and len(a) == 2:
        play(VOICES / a[1] / "ref.wav")
        return 0
    if sub == "use" and len(a) == 2:
        if not a[1].startswith("preset:") and not (VOICES / a[1]).exists():
            raise SystemExit(f"no voice '{a[1]}': iro voice list")
        save_cfg(voice=a[1])
        print(f"ARC now speaks as '{a[1]}'.")
        return 0
    if sub == "say" and len(a) >= 2:
        voice = a[a.index("--voice") + 1] if "--voice" in a else None
        text = " ".join(x for i, x in enumerate(a[1:], 1) if x != "--voice" and (i == 1 or a[i - 1] != "--voice"))
        wav, dt = say(text, voice)
        report(wav, dt)
        play(wav)
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
