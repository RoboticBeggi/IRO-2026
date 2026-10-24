#!/usr/bin/env python3
"""Timed test of the whole voice loop without a person: recorded questions go through Whisper, ARC and
the Jarvis voice exactly as in `iro arc talk`. Measures, per question: transcription time, time from
"the visitor stopped" to ARC's first sound, and the silences between parts of the spoken reply.
Plays through the speakers (so you can listen too) and saves what was said.

    ~/iro-2026/tools/talk_bench.py [WAV ...]     default: the questions in tools/talk_bench/
"""

import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iro import arc, talk  # noqa: E402
from iro.skills import Body  # noqa: E402

QUESTIONS = Path(__file__).resolve().parent / "talk_bench"


def load16(p):
    with wave.open(str(p)) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        sr = w.getframerate()
    return np.interp(np.linspace(0, len(a) - 1, int(len(a) * talk.RATE / sr)), np.arange(len(a)), a).astype(np.int16)


def gaps(pcm, thr=250, min_gap=0.15):
    """Silences longer than min_gap inside spoken audio (seconds)."""
    f = np.abs(pcm[: len(pcm) // 240 * 240].reshape(-1, 240)).max(1) > thr      # 10 ms frames
    out, run = [], 0
    for v in f:
        if not v:
            run += 1
        else:
            if run * 0.01 >= min_gap:
                out.append(run * 0.01)
            run = 0
    return out


def main():
    wavs = [Path(x) for x in sys.argv[1:]] or sorted(QUESTIONS.glob("*.wav"))
    t = time.time()
    arc.start(quiet=True)
    talk.start_whisper()
    voice = talk.Voice()
    voice.precache([(o, m) for o in sorted({x for v in arc.OPENERS.values() for x in v})
                    for m in [None] + sorted(voice.moods)])
    talk.transcribe(np.zeros(talk.RATE, np.int16))
    arc.ask([{"role": "user", "content": "Hello"}], max_tokens=4, wiki=False)
    print(f"ready in {time.time() - t:.0f} s\n")
    body = Body()
    rows = []
    for w in wavs:
        u = load16(w)
        voice.wavs.clear()
        t0 = time.time()
        text, lang = talk.transcribe(u)
        t_heard = time.time() - t0
        voice.first_audio = None
        mood = {"mood": None}
        split = talk.Splitter(voice, "English", mood)
        replies = arc.respond(body, [], text, "en", on_text=split.feed, on_mood=lambda m: mood.update(mood=m))
        split.flush()
        while voice.first_audio is None and voice.speaking():
            time.sleep(0.01)
        first = (voice.first_audio or time.time()) - t0
        while voice.speaking():
            time.sleep(0.05)
        said = np.concatenate(voice.wavs) if voice.wavs else np.zeros(1, np.int16)
        total = time.time() - t0
        g = gaps(said)
        rows.append((t_heard, first, total, len(said) / talk.OUT_RATE, max(g, default=0)))
        print(f"{w.name}: heard {t_heard:.2f}s \"{text}\"\n   first sound {first:.2f}s | spoke {len(said) / 24000:.1f}s "
              f"| longest pause inside the reply {max(g, default=0):.2f}s | mood {mood['mood']}\n   ARC: "
              f"{' '.join(x['text'] for x in replies)[:200]}\n")
    a = np.array(rows)
    print(f"MEDIAN: heard {np.median(a[:, 0]):.2f}s, first sound {np.median(a[:, 1]):.2f}s, "
          f"longest inner pause {np.median(a[:, 4]):.2f}s (worst {a[:, 4].max():.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
