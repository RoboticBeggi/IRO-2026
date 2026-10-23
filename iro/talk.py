"""iro arc talk: speak with ARC. The mic array hears you, Whisper writes it down, ARC answers, and
ARC's chosen Qwen voice speaks it, sentence by sentence while the rest is still being written.

    iro arc talk [--robot] [--en]     --robot: skills move the arm; --en: always answer in English

Pipeline (all on the Jetson, offline):
  hear     reSpeaker XVF3800 (beamformed, noise-suppressed) 16 kHz; a sentence starts when the
           array's own speech detector or the loudness says so, and ends after 0.7 s of quiet
  write    whisper.cpp server, large-v3-turbo on the GPU, language detected (99 languages)
  think    ARC (iro.arc.respond): facts, offline Wikipedia, skills, robot status
  speak    qwentts.cpp tts-server streams 24 kHz PCM as it generates (first audio ~0.3 s);
           each finished sentence is spoken while ARC writes the next
While ARC speaks, the mic is ignored (the screen's speaker is not echo-cancelled by the array).
Every turn is recorded (iro review): what was heard (wav), transcript, reply, timings.
"""

import base64
import json
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

WHISPER_BIN = Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-server"
WHISPER_MODEL = Path.home() / "voice-models" / "ggml-large-v3-turbo-q8_0.bin"
WHISPER_PORT, TTS_PORT = 8092, 8093
NAMES = "ARC, wayang kulit, Tok Dalang, Kelantan, songket, silat, tepak sirih, Mak Yong, Hanuman, Seri Rama."
FIXES = {r"\bark\b": "ARC", r"\barc\b": "ARC", r"\bwayang kilit\b": "wayang kulit", r"\btok daling\b": "Tok Dalang"}
RATE = 16000
PHANTOM = re.compile(r"^\W*(thank you|thanks for watching|thank you for watching|you|bye|\.+|"
                     r"subtitles by.*|terima kasih\.?)\W*$", re.I)
LANGS = {"ms": "ms", "en": "en", "de": "de"}


def get(url, timeout=1):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:                              # noqa: BLE001
        return False


def post_json(url, body, timeout=60):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def wait_up(url, what, log, secs=120):
    t0 = time.time()
    while not get(url):
        if time.time() - t0 > secs:
            raise SystemExit(f"{what} did not start in {secs} s: see {log}")
        time.sleep(0.3)


# ------------------------------------------------------------------ write (Whisper)

def start_whisper():
    url = f"http://127.0.0.1:{WHISPER_PORT}"
    if not get(url + "/health"):
        subprocess.Popen([str(WHISPER_BIN), "-m", str(WHISPER_MODEL), "--port", str(WHISPER_PORT), "-l", "auto",
                          "-nt", "-bs", "1", "-bo", "1", "-sns", "--prompt", NAMES],
                         stdout=open("/tmp/arc-whisper.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
        wait_up(url + "/health", "Whisper", "/tmp/arc-whisper.log")


def transcribe(pcm16):
    """(text, language) of 16 kHz mono int16 audio."""
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm16.tobytes())
    boundary = "----arc" + str(time.time_ns())
    parts = []
    for k, v in (("response_format", "verbose_json"), ("temperature", "0.0"), ("prompt", NAMES)):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
                 f"Content-Type: audio/wav\r\n\r\n".encode() + buf.getvalue() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(f"http://127.0.0.1:{WHISPER_PORT}/inference", b"".join(parts),
                                 {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    text = (d.get("text") or "").strip()
    segs = d.get("segments") or []
    # Whisper invents text on noise ("Thank you.", "Thanks for watching!"): drop low-confidence results
    nsp = float(np.mean([x.get("no_speech_prob", 0.0) for x in segs])) if segs else 0.0
    lp = float(np.mean([x.get("avg_logprob", 0.0) for x in segs])) if segs else 0.0
    if nsp > 0.6 or lp < -1.2:
        return "", d.get("detected_language")
    # "thank you" is real when a visitor says it: dropped only when Whisper doubts there was speech
    if PHANTOM.match(text) and (nsp > 0.25 or lp < -0.8 or text.strip(" .") in ("you", "")):
        return "", d.get("detected_language")
    for pat, rep in FIXES.items():
        text = re.sub(pat, rep, text, flags=re.I)
    return text, d.get("detected_language") or d.get("language")


# ------------------------------------------------------------------ speak (Qwen)

class Voice:
    """qwentts.cpp tts-server with ARC's chosen voice; say() queues a sentence, a worker streams it."""

    def __init__(self):
        from iro.voice import MODELS, QT, VOICES, cfg
        c = cfg()
        self.voice = c["voice"] or "preset:ryan"
        self.sink = c["output"]
        model = c["customvoice"] if self.voice.startswith("preset:") else c["base"]
        self.url = f"http://127.0.0.1:{TTS_PORT}"
        if not get(self.url + "/health"):
            subprocess.Popen([str(QT / "build" / "tts-server"), "--model", str(MODELS / model), "--codec",
                              str(MODELS / c["codec"]), "--port", str(TTS_PORT), "--lang", c["lang"]],
                             stdout=open("/tmp/arc-tts.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
            wait_up(self.url + "/health", "the Qwen voice", "/tmp/arc-tts.log")
        if self.voice.startswith("preset:"):
            self.name = self.voice.split(":", 1)[1]
        else:                                      # a kept voice: register its clip once
            d = VOICES / self.voice
            post_json(self.url + "/v1/audio/voices", {
                "name": self.voice, "ref_text": (d / "ref.txt").read_text().strip(),
                "spk_b64": base64.b64encode((d / "ref.spk").read_bytes()).decode(),
                "rvq_b64": base64.b64encode((d / "ref.rvq").read_bytes()).decode()}).read()
            self.name = self.voice
        self.q = queue.Queue()
        self.busy = threading.Event()
        self.last_end = 0.0
        self.first_audio = None
        self.wavs = []
        threading.Thread(target=self._worker, daemon=True).start()

    def say(self, sentence, lang="English"):
        self.busy.set()
        self.q.put((sentence, lang))

    def _worker(self):
        while True:
            text, lang = self.q.get()
            play = subprocess.Popen(["pacat", "--raw", "--format=s16le", "--rate=24000", "--channels=1",
                                     f"--device={self.sink}"], stdin=subprocess.PIPE)
            got = bytearray()
            try:
                with post_json(self.url + "/v1/audio/speech", {"input": text, "voice": self.name, "language": lang,
                                                              "response_format": "pcm"}, timeout=120) as r:
                    while True:
                        chunk = r.read(4800)
                        if not chunk:
                            break
                        if self.first_audio is None:
                            self.first_audio = time.time()
                        got += chunk
                        play.stdin.write(chunk)
                        play.stdin.flush()
            except Exception as e:                 # noqa: BLE001 - skip the sentence, keep talking
                print(f"\n(voice problem: {e})")
            play.stdin.close()
            play.wait()
            self.wavs.append(bytes(got))
            self.last_end = time.time()
            if self.q.empty():
                self.busy.clear()

    def speaking(self, tail=0.4):
        return self.busy.is_set() or time.time() - self.last_end < tail


def sentences(buf):
    """(complete sentences, rest) of streamed text."""
    parts = re.split(r"(?<=[.!?])\s+", buf)
    return [p.strip() for p in parts[:-1] if p.strip()], parts[-1]


# ------------------------------------------------------------------ hear (mic array)

class Ears:
    """The array's ASR channel through Silero VAD: yields one spoken sentence (int16) at a time.
    (23 Oct: replaced a loudness threshold, which missed quiet voices and fired on clatter.)"""

    def __init__(self, mic=None):
        from iro.speaker import Vad, stream
        self.mic = mic
        self.proc = stream()                       # through PulseAudio (it holds the sound card)
        self.vad = Vad()

    def utterances(self, voice):
        frame = 512                                # 32 ms, Silero's window
        while True:
            raw = self.proc.stdout.read(frame * 2)
            if not raw:
                return
            a = np.frombuffer(raw, np.int16)
            if voice.speaking():                   # ARC is talking: do not listen to ourselves
                self.vad.reset()
                continue
            for seg in self.vad.feed(a):
                if len(seg) > RATE * 0.4:
                    yield seg

    def close(self):
        self.proc.terminate()


# ------------------------------------------------------------------ the loop

def main():
    from iro.arc import cfg as arc_cfg
    from iro.arc import respond, start, topic
    from iro.record import run
    from iro.skills import Body
    a = sys.argv[1:]
    english = "--en" in a
    r = run("arc-talk")
    print("Starting ARC's mind, hearing and voice (first time ~1 min)...")
    start(quiet=True)
    start_whisper()
    voice = Voice()
    try:
        from iro.visitors import MicThread
        mic = MicThread()
    except Exception:                              # noqa: BLE001 - loudness alone still works
        mic = None
    body = Body(robot="--robot" in a)
    ears = Ears(mic)
    from concurrent.futures import ThreadPoolExecutor
    from iro.speaker import Owner
    owner_id = Owner()
    pool = ThreadPoolExecutor(1)
    if not owner_id.ok:
        print("(no owner voice yet: everyone is a visitor. Teach it yours: iro voice enroll)")
    hist = []
    greet = topic()["greeting"]["en"]
    print(f"ARC ({arc_cfg()['model']}, voice {voice.voice}). Speak to it. Ctrl+C to stop.\nARC: {greet}")
    voice.say(greet)
    lang_names = {"ms": "Malay", "en": "English", "de": "German"}
    try:
        for k, u in enumerate(ears.utterances(voice)):
            t0 = time.time()
            who = pool.submit(owner_id.is_owner, u)        # voiceprint on the CPU while Whisper runs
            text, lang = transcribe(u)
            is_owner, score = who.result()
            t_heard = time.time() - t0
            if not text or len(text) < 2 or text.startswith("["):
                continue
            lang = "en" if english else LANGS.get(lang, "en")
            print(f"\n{'sir' if is_owner else 'visitor'} ({lang}, voice {score:.2f}, {t_heard:.1f}s): {text}")
            with wave.open(str(r.dir / f"heard_{k:03d}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(u.tobytes())
            buf = {"text": ""}
            voice.first_audio = None
            t_think = time.time()

            def on_text(s):
                s = re.sub(r"[*_#`]+|[\U0001F000-\U0001FAFF\u2600-\u27BF]", "", s)   # spoken: no markdown/emoji
                print(s, end="", flush=True)
                buf["text"] += s
                done, buf["text"] = sentences(buf["text"])
                for sent in done:
                    voice.say(sent, lang_names.get(lang, "English"))

            print("ARC: ", end="", flush=True)
            replies = respond(body, hist, text, lang, on_text=on_text, owner=is_owner)
            if buf["text"].strip():
                voice.say(buf["text"].strip(), lang_names.get(lang, "English"))
            while voice.first_audio is None and voice.speaking():
                time.sleep(0.02)
            first = (voice.first_audio or time.time()) - t0
            print(f"\n     [heard in {t_heard:.1f}s, first word spoken {first:.1f}s after you stopped]")
            r.log("talk", k=k, heard=text, lang=lang, owner=int(is_owner), voice_score=round(score, 3),
                  t_transcribe=round(t_heard, 2),
                  t_first_audio=round(first, 2), t_think=round(time.time() - t_think, 2),
                  reply=" ".join(x["text"] for x in replies), act=replies[0].get("act") or "")
    except KeyboardInterrupt:
        pass
    finally:
        ears.close()
        if mic:
            mic.close()
        for i, pcm in enumerate(voice.wavs):
            with wave.open(str(r.dir / f"said_{i:03d}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(24000)
                w.writeframes(pcm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
