"""iro arc talk: speak with ARC. The mic array hears you, Whisper writes it down, ARC answers, and
ARC's chosen Qwen voice speaks it while the rest is still being written.

    iro arc talk [--robot] [--barge-in]    --robot: skills move the arm; --barge-in: you can interrupt
                                           (only with the speaker on the array's 3.5 mm jack, so its
                                           echo cancelling removes ARC's own voice)

Pipeline (all on the Jetson, offline), with the 24 Oct upgrades (PLAN.md "Voice upgrades"):
  hear     reSpeaker XVF3800 through PulseAudio, Silero VAD finds each sentence; a sentence that ends
           on "and", "the", a comma... waits for the rest (#20)
  write    whisper.cpp large-v3-turbo on the GPU, language fixed (default English: auto-detect was
           2x slower and turned English into Malay, #1)
  think    ARC (iro.arc.respond); a pre-recorded "let me think" plays at once when it must look
           something up (#5); small talk never goes to Wikipedia (#7)
  speak    qwentts.cpp tts-server: the first clause is spoken as soon as it is written (#4), the rest
           as one paragraph for natural intonation (#12); synthesis runs ahead of playback (#3) into
           ONE continuous audio stream (#2) with trimmed silences and a short breath between parts
           (#15); text made speakable (#13); per-mood expressiveness and loudness (#16, #18);
           common lines pre-recorded at start-up (#6)
  show     the LED ring: green = listening, purple = thinking, blue = speaking (#23)
Everything is warmed up at start (#8, #9). Every turn is recorded (iro review).
"""

import base64
import json
import os
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
WHISPER_MODEL = Path.home() / "voice-models" / "ggml-large-v3-turbo-q5_0.bin"
WHISPER_PORT, TTS_PORT = 8092, 8093
NAMES = "ARC, wayang kulit, Tok Dalang, Kelantan, songket, silat, tepak sirih, Mak Yong, Hanuman, Seri Rama."
FIXES = {r"\bark\b": "ARC", r"\barc\b": "ARC", r"\bwayang kilit\b": "wayang kulit", r"\btok daling\b": "Tok Dalang"}
RATE, OUT_RATE = 16000, 24000
PHANTOM = re.compile(r"^\W*(thank you|thanks for watching|thank you for watching|you|bye|\.+|"
                     r"subtitles by.*|terima kasih\.?)\W*$", re.I)
LANGS = {"ms": "ms", "en": "en", "de": "de"}
LANG_NAMES = {"ms": "Malay", "en": "English", "de": "German"}
# end of turn (#20): a sentence ending like this is probably not finished
UNFINISHED = re.compile(r"(,|\b(and|but|or|so|because|the|a|an|of|to|with|for|my|your|um|uh|er|like|that|which|"
                        r"dan|tetapi|atau|yang|und|aber|oder|der|die|das))\W*$", re.I)
# pre-recorded lines (#5, #6)
ACKS = {"visitor": ["Hmm, let me think.", "Ah, good question. One moment.", "Let me see what I know."],
        "owner": ["One moment, sir.", "Hmm, let me look that up, sir.", "Certainly, sir. Thinking."]}
TEMPS = {None: 0.7, "amused": 0.95, "excited": 0.95, "curious": 0.85, "concerned": 0.75, "warm": 0.8, "proud": 0.85}
BREATH_S = 0.25                                     # natural pause between parts (#15)


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
        try:
            tail = Path(log).read_text(errors="ignore")[-2000:]
        except OSError:
            tail = ""
        if "FATAL" in tail or "out of memory" in tail:     # 23 Oct: the voice died and talk waited 2 min
            raise SystemExit(f"{what} failed to start (memory?): see {log}\n  " + tail.strip().splitlines()[-1])
        time.sleep(0.3)


def stt_lang():
    from iro.voice import cfg
    return cfg().get("stt_lang", "en")


# ------------------------------------------------------------------ write (Whisper)

MOONSHINE = Path.home() / "voice-models" / "sherpa-onnx-moonshine-base-en-int8"
_moon = None


def stt_engine():
    """moonshine (default, ARC Lite): CPU, ~0.25 s, English; whisper: GPU, ~1 GB, 99 languages."""
    from iro.voice import cfg
    return cfg().get("stt", "moonshine")


def start_whisper():
    """Load the speech recogniser (name kept for older callers)."""
    global _moon
    if stt_engine() == "moonshine":
        if _moon is None:
            import sherpa_onnx
            m = MOONSHINE
            _moon = sherpa_onnx.OfflineRecognizer.from_moonshine(
                preprocessor=str(m / "preprocess.onnx"), encoder=str(m / "encode.int8.onnx"),
                uncached_decoder=str(m / "uncached_decode.int8.onnx"), cached_decoder=str(m / "cached_decode.int8.onnx"),
                tokens=str(m / "tokens.txt"), num_threads=4)
        return
    url = f"http://127.0.0.1:{WHISPER_PORT}"
    if not get(url + "/health"):
        from iro.hw.system import expendable
        expendable(subprocess.Popen([str(WHISPER_BIN), "-m", str(WHISPER_MODEL), "--port", str(WHISPER_PORT), "-l",
                                     stt_lang(), "-nt", "-bs", "1", "-bo", "1", "-sns", "--prompt", NAMES],
                                    stdout=open("/tmp/arc-whisper.log", "w"), stderr=subprocess.STDOUT,
                                    start_new_session=True))
        wait_up(url + "/health", "Whisper", "/tmp/arc-whisper.log")


def transcribe(pcm16, lang=None):
    """(text, language) of 16 kHz mono int16 audio. Moonshine (24 Oct, ARC Lite: 0.25 s on the CPU,
    as accurate as Whisper on the owner's recorded voice) + names fixed by sound (iro/names.py)."""
    from iro.names import fix
    if stt_engine() == "moonshine":
        start_whisper()
        st = _moon.create_stream()
        st.accept_waveform(RATE, np.asarray(pcm16, np.float32) / 32768.0)
        _moon.decode_stream(st)
        text = st.result.text.strip()
        if not text or PHANTOM.match(text) and len(pcm16) < RATE * 0.8:
            return "", "en"
        text = fix(text)
        for pat, rep_ in FIXES.items():
            text = re.sub(pat, rep_, text, flags=re.I)
        return text, "en"
    return _transcribe_whisper(pcm16, lang)


def _transcribe_whisper(pcm16, lang=None):
    """(text, language) via the whisper.cpp server. lang: fixed language ("en") or "auto"."""
    import io
    lang = lang or stt_lang()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm16.tobytes())
    boundary = "----arc" + str(time.time_ns())
    parts = []
    for k, v in (("response_format", "verbose_json"), ("temperature", "0.0"), ("prompt", NAMES), ("language", lang)):
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
    det = d.get("detected_language") or (lang if lang != "auto" else None)
    if nsp > 0.6 or lp < -1.2:
        return "", det
    # "thank you" is real when a visitor says it: dropped only when Whisper doubts there was speech
    if PHANTOM.match(text) and (nsp > 0.25 or lp < -0.8 or text.strip(" .") in ("you", "")):
        return "", det
    for pat, rep in FIXES.items():
        text = re.sub(pat, rep, text, flags=re.I)
    return text, det


# ------------------------------------------------------------------ speak (Qwen)

def trim(a, thr=250):
    """Remove leading/trailing near-silence from an int16 chunk (#15)."""
    loud = np.flatnonzero(np.abs(a) > thr)
    if not len(loud):
        return a[:0]
    pad = int(OUT_RATE * 0.03)
    return a[max(0, loud[0] - pad): loud[-1] + pad]


class Voice:
    """Qwen voice server + one continuous player. say() queues text; a synth thread makes audio ahead
    of time (#3); a player thread writes it to ONE pacat stream (#2) with a breath between parts."""

    def __init__(self, lines=()):
        from iro.voice import MODELS, QT, VOICES, cfg, unmute
        c = cfg()
        self.voice = c["voice"] or "preset:ryan"
        self.sink = c["output"]
        unmute()
        model = c["customvoice"] if self.voice.startswith("preset:") else c["base"]
        self.url = f"http://127.0.0.1:{TTS_PORT}"
        if not get(self.url + "/health"):
            from iro.hw.system import expendable
            expendable(subprocess.Popen([str(QT / "build" / "tts-server"), "--model", str(MODELS / model), "--codec",
                                         str(MODELS / c["codec"]), "--port", str(TTS_PORT), "--lang", c["lang"]],
                                        stdout=open("/tmp/arc-tts.log", "w"), stderr=subprocess.STDOUT,
                                        start_new_session=True,
                                        env={**os.environ, "QWEN_KV_LEN": "1024"}))   # 900 -> 225 MB (23 Oct)
            wait_up(self.url + "/health", "the Qwen voice", "/tmp/arc-tts.log")
        self.moods, self.gain = {}, {}
        if self.voice.startswith("preset:"):
            self.name = self.voice.split(":", 1)[1]
        else:                                      # a kept voice and its moods: register each clip once
            dirs = [VOICES / self.voice] + sorted(VOICES.glob(f"{self.voice}-*"))
            rms = {}
            for d in dirs:
                if not (d / "ref.rvq").exists():
                    continue
                post_json(self.url + "/v1/audio/voices", {
                    "name": d.name, "ref_text": (d / "ref.txt").read_text().strip(),
                    "spk_b64": base64.b64encode((d / "ref.spk").read_bytes()).decode(),
                    "rvq_b64": base64.b64encode((d / "ref.rvq").read_bytes()).decode()}).read()
                if d.name != self.voice:
                    self.moods[d.name[len(self.voice) + 1:]] = d.name
                with wave.open(str(d / "ref.wav")) as w:
                    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
                rms[d.name] = float(np.sqrt(np.mean(x[np.abs(x) > 300] ** 2)))
            self.name = self.voice
            base = rms.get(self.name, 3000.0)
            self.gain = {n: float(np.clip(base / v, 0.6, 1.6)) for n, v in rms.items()}   # even loudness (#18)
        self.cache = {}
        self.cache_dir = Path.home() / "voice-samples" / "cache" / self.voice
        self.text_q, self.audio_q = queue.Queue(), queue.Queue()
        self.pending = 0
        self.lock = threading.Lock()
        self.play_until = 0.0
        self.first_audio = None
        self.stop_flag = threading.Event()
        self.wavs = []
        self.player = None
        threading.Thread(target=self._synth, daemon=True).start()
        threading.Thread(target=self._play, daemon=True).start()
        self.precache([(line, None) for line in lines])   # common lines (#6); also warms the voice (#9)

    def precache(self, items):
        """Pre-recorded lines, kept on disk (made once, ~1-2 s each), loaded into memory."""
        import hashlib
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        todo = 0
        for text, mood in items:
            f = self.cache_dir / (hashlib.sha1(f"{mood}|{text}".encode()).hexdigest()[:16] + ".npy")
            if f.exists():
                self.cache[(text, mood)] = np.load(f)
                continue
            if not todo:
                print("  recording ARC's openers in every mood (once; ~2 min)...", flush=True)
            todo += 1
            a = self.render(text, mood)
            np.save(f, a)
            self.cache[(text, mood)] = a

    # -- making audio
    def _stream(self, text, mood, lang="English"):
        """Yield int16 chunks of `text` from the server as they are generated."""
        from iro.speakable import speakable
        name = self.moods.get(mood, self.name)
        g = self.gain.get(name, 1.0)
        body = {"input": speakable(text), "voice": name, "language": lang, "response_format": "pcm",
                "max_new_tokens": 600, "temperature": TEMPS.get(mood, 0.7)}
        rest = b""
        with post_json(self.url + "/v1/audio/speech", body, timeout=120) as r:
            while True:
                chunk = r.read(4800)
                if not chunk:
                    break
                chunk = rest + chunk
                n = len(chunk) // 2 * 2
                rest = chunk[n:]
                a = np.frombuffer(chunk[:n], np.int16)
                if g != 1.0:
                    a = np.clip(a.astype(np.float32) * g, -32767, 32767).astype(np.int16)
                yield a

    def render(self, text, mood=None):
        """Whole text as one trimmed int16 array (for the cache)."""
        return trim(np.concatenate(list(self._stream(text, mood)) or [np.zeros(0, np.int16)]))

    def say(self, text, lang="English", mood=None):
        """Queue a part of a reply (a clause, a paragraph or a cached line)."""
        text = text.strip()
        if not text:
            return
        with self.lock:
            self.pending += 1
        self.text_q.put((text, lang, mood))

    def _synth(self):
        while True:
            text, lang, mood = self.text_q.get()
            try:
                if (text, mood) in self.cache:
                    self.audio_q.put(self.cache[(text, mood)])
                else:
                    head = True                    # trim the silence the voice adds at the start
                    tail = np.zeros(0, np.int16)
                    for a in self._stream(text, mood, lang):
                        if self.stop_flag.is_set():
                            break
                        if head:
                            loud = np.flatnonzero(np.abs(a) > 250)
                            if not len(loud):
                                continue
                            a, head = a[max(0, loud[0] - 720):], False
                        a = np.concatenate([tail, a])
                        keep = int(OUT_RATE * 0.4)     # hold back 0.4 s to trim the trailing silence
                        if len(a) > keep:
                            self.audio_q.put(a[:-keep])
                            tail = a[-keep:]
                        else:
                            tail = a
                    loud = np.flatnonzero(np.abs(tail) > 250)
                    self.audio_q.put(tail[: loud[-1] + 720] if len(loud) else tail[:0])
            except Exception as e:                 # noqa: BLE001 - skip this part, keep talking
                print(f"\n(voice problem: {e})")
            self.audio_q.put(None)                 # end of part: a breath follows
            with self.lock:
                self.pending -= 1

    def _sink(self):
        """The configured output if it exists, else the default, else any: an HDMI screen drops its audio
        device when it sleeps or switches input ("No such entity", 24 Oct: ARC talked to nothing)."""
        out = subprocess.run(["pactl", "list", "short", "sinks"], capture_output=True, text=True).stdout
        sinks = [ln.split("\t")[1] for ln in out.splitlines() if "\t" in ln]
        if self.sink in sinks:
            return self.sink
        info = subprocess.run(["pactl", "info"], capture_output=True, text=True).stdout
        default = next((ln.split(":", 1)[1].strip() for ln in info.splitlines() if ln.startswith("Default Sink")), "")
        for cand in [default] + [x for x in sinks if "hdmi" in x] + [x for x in sinks if "XVF3800" in x] + sinks:
            if cand in sinks:
                print(f"\n(sound output {self.sink} is missing; using {cand})")
                return cand
        return self.sink

    def _ensure_player(self):
        if self.player is None or self.player.poll() is not None:
            dev = self._sink()
            self.player = subprocess.Popen(["pacat", "--raw", "--format=s16le", f"--rate={OUT_RATE}", "--channels=1",
                                            f"--device={dev}", "--latency-msec=60"], stdin=subprocess.PIPE)

    def _write(self, a):
        self._ensure_player()
        now = time.time()
        self.play_until = max(self.play_until, now) + len(a) / OUT_RATE
        try:
            self.player.stdin.write(a.tobytes())
            self.player.stdin.flush()
        except BrokenPipeError:
            self.player = None

    def _play(self):
        got = []
        while True:
            a = self.audio_q.get()
            if self.stop_flag.is_set():
                got = []
                continue
            if a is None:                          # end of a part: a short natural breath
                if got:
                    self.wavs.append(np.concatenate(got))
                    got = []
                    self._write(np.zeros(int(OUT_RATE * BREATH_S), np.int16))
                continue
            if len(a):
                if self.first_audio is None:
                    self.first_audio = time.time()
                got.append(a)
                self._write(a)

    def speaking(self, tail=0.3):
        with self.lock:
            busy = self.pending > 0
        return busy or not self.audio_q.empty() or time.time() < self.play_until + tail

    def stop(self):
        """Barge-in: drop everything queued and stop the sound now (#21)."""
        self.stop_flag.set()
        for q in (self.text_q, self.audio_q):
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
        with self.lock:
            self.pending = 0
        if self.player:
            self.player.kill()
            self.player = None
        self.play_until = 0.0
        time.sleep(0.05)
        self.stop_flag.clear()


class Splitter:
    """Streamed reply -> the first clause at once (#4), the rest as one paragraph (#12). If the reply
    grows long before it ends, it is flushed at sentence ends so the voice never runs dry."""

    MAX_WORDS = 45                                 # the 2B ignores "about 35 words": spoke 16.8 s (24 Oct)

    def __init__(self, voice, lang, mood_ref):
        from iro.arc import OPENERS
        self.voice, self.lang, self.mood = voice, lang, mood_ref
        self.buf, self.first_done, self.opener_done = "", False, False
        self.openers = sorted({o for v in OPENERS.values() for o in v}, key=len, reverse=True)
        self.said_words, self.done = 0, False

    def _say(self, text):
        if self.done:
            return
        words = len(text.split())
        if self.said_words + words > self.MAX_WORDS and self.said_words:
            cut = re.split(r"(?<=[.!?])\s+", text)          # finish the sentence that crosses the limit
            keep = []
            for x in cut:
                keep.append(x)
                if self.said_words + len(" ".join(keep).split()) >= self.MAX_WORDS:
                    break
            text, self.done = " ".join(keep), True
        self.said_words += len(text.split())
        self.voice.say(text, self.lang, self.mood["mood"])

    def feed(self, s):
        self.buf += s
        if not self.opener_done:                   # the pre-recorded opener goes out the moment it is written
            lead = self.buf.lstrip()
            hit = next((o for o in self.openers if lead.startswith(o + " ")), None)
            if hit:
                self._say(hit)
                self.buf, self.opener_done = lead[len(hit) + 1:], True
            elif len(lead) > 24:
                self.opener_done = True            # no opener (e.g. a fact answer): carry on as usual
            else:
                return
        if not self.first_done:
            # "Hello!" alone may go first (it waited for the whole next sentence: 3+ s, 24 Oct)
            m = re.search(r"^(.{5,}?[.!?])\s|^((?:\S+\s+){3,}?\S*[,;:—-])\s", self.buf)
            if m:
                cut = m.end()
                self._say(self.buf[:cut])
                self.buf, self.first_done = self.buf[cut:], True
        elif len(self.buf) > 220:                  # long story: hand over whole sentences as they come
            parts = re.split(r"(?<=[.!?])\s+", self.buf)
            if len(parts) > 1:
                self._say(" ".join(parts[:-1]))
                self.buf = parts[-1]

    def flush(self):
        self._say(self.buf)
        self.buf = ""


# ------------------------------------------------------------------ hear (mic array)

class Ears:
    """The array's chosen channel through Silero VAD, in a background thread: finished sentences go
    to a queue, so the loop can wait a little for the rest of an unfinished one (#20)."""

    def __init__(self, voice, barge_in=False):
        from iro.speaker import Vad, stream
        self.voice, self.barge_in = voice, barge_in
        self.proc = stream()                       # through PulseAudio (it holds the sound card)
        self.vad = Vad()
        self.q = queue.Queue()
        self.run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        frame = 512                                # 32 ms, Silero's window
        heard_over = 0
        while self.run:
            raw = self.proc.stdout.read(frame * 2)
            if not raw:
                return
            a = np.frombuffer(raw, np.int16)
            if self.voice.speaking():
                if not self.barge_in:              # no echo cancelling: do not listen to ourselves
                    self.vad.reset()
                    continue
                self.vad.feed(a)                   # with echo cancelling: the visitor may interrupt
                heard_over = heard_over + 1 if self.vad.speaking() else 0
                if heard_over >= 10:               # ~0.3 s of speech over ARC: stop talking (#21)
                    self.voice.stop()
                    heard_over = 0
                continue
            for seg in self.vad.feed(a):
                if len(seg) > RATE * 0.4:
                    self.q.put(seg)

    def next(self, timeout=None):
        return self.q.get(timeout=timeout)

    def close(self):
        self.run = False
        self.proc.terminate()


# ------------------------------------------------------------------ the LED ring (#23)

class Ring:
    def __init__(self, mic):
        self.m = getattr(mic, "m", None)

    def show(self, state):
        if not self.m:
            return
        try:
            if state == "listening":
                self.m.led_doa()                   # points at the voice
            else:
                self.m.led_color({"thinking": 0x8000FF, "speaking": 0x0060FF}[state])
        except Exception:                          # noqa: BLE001 - the ring is a nicety
            pass


# ------------------------------------------------------------------ the loop

def main():
    from concurrent.futures import ThreadPoolExecutor

    from iro.arc import cfg as arc_cfg
    from iro.arc import respond, start, topic
    from iro.record import run
    from iro.skills import Body
    from iro.speaker import Owner
    a = sys.argv[1:]
    r = run("arc-talk")
    from iro.hw.system import mem_available_mb
    need = 2200 if stt_engine() == "moonshine" else 3000   # ARC ~1.3 GB + voice ~0.6 (+ Whisper ~0.8) + headroom
    have = mem_available_mb()
    if have < need and not any(x in a for x in ("--force",)):
        print(f"Only {have} MB of memory is free; ARC's voice loop needs about {need} MB. Close other programs "
              "(VS Code on the Jetson itself uses ~1.5 GB: use Remote-SSH from a laptop instead), stop old servers "
              "(iro arc stop), or run anyway with --force.")
        return 1
    print("Starting ARC's mind, hearing and voice...")
    t_start = time.time()
    start(quiet=True)
    start_whisper()
    greet = topic()["greeting"]["en"]
    voice = Voice(lines=[greet])
    from iro.arc import OPENERS
    voice.precache([(o, m) for o in sorted({x for v in OPENERS.values() for x in v})
                    for m in [None] + sorted(voice.moods)])
    # warm everything once (#8, #9): Wikipedia's index from disk, Whisper, ARC's prompt cache
    try:
        from iro.wiki import archives, search
        for lg in archives():
            search("history", lg, k=1)
    except Exception:                              # noqa: BLE001 - no encyclopedia is fine
        pass
    transcribe(np.zeros(RATE, np.int16))
    from iro.arc import ask
    ask([{"role": "user", "content": "Hello"}], max_tokens=4, wiki=False)
    try:
        from iro.visitors import MicThread
        mic = MicThread()
    except Exception:                              # noqa: BLE001 - VAD alone still works
        mic = None
    ring = Ring(mic)
    body = Body(robot="--robot" in a)
    ears = Ears(voice, barge_in="--barge-in" in a)
    owner_id = Owner()
    pool = ThreadPoolExecutor(1)
    if not owner_id.ok:
        print("(no owner voice yet: everyone is a visitor. Teach it yours: iro voice enroll)")
    hist = []
    print(f"Ready in {time.time() - t_start:.0f} s. ARC ({arc_cfg()['model']}, voice {voice.voice}, hearing "
          f"{stt_lang()}). Speak to it. Ctrl+C to stop.\nARC: {greet}")
    ring.show("speaking")
    voice.say(greet)
    k = 0
    try:
        while True:
            while voice.speaking():
                time.sleep(0.05)
            ring.show("listening")
            u = ears.next()
            t0 = time.time()
            ring.show("thinking")
            who = pool.submit(owner_id.is_owner, u)        # voiceprint on the CPU while Whisper runs
            text, lang = transcribe(u)
            # end of turn (#20): an unfinished sentence waits up to 1.2 s for the rest
            while text and UNFINISHED.search(text):
                try:
                    more = ears.next(timeout=1.2)
                except queue.Empty:
                    break
                u = np.concatenate([u, more])
                text, lang = transcribe(u)
                who = pool.submit(owner_id.is_owner, u)
            is_owner, score = who.result()
            t_heard = time.time() - t0
            if not text or len(text) < 2 or text.startswith("["):
                continue
            lang = LANGS.get(lang, "en")
            if is_owner:
                owner_id.learn(u, score)
            print(f"\n{'sir' if is_owner else 'visitor'} ({lang}, voice {score:.2f}, {t_heard:.1f}s): {text}")
            with wave.open(str(r.dir / f"heard_{k:03d}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(u.tobytes())
            voice.first_audio = None
            # ARC Lite: facts come from the article summary in < 1 s, so no "let me think" any more
            # (24 Oct: it only made the answer longer). Kept for slow paths: none at the moment.
            ack = False
            mood = {"mood": None}
            split = Splitter(voice, LANG_NAMES.get(lang, "English"), mood)
            t_think = time.time()

            def on_text(s):
                s = re.sub(r"[*_#`]+|[\U0001F000-\U0001FAFF☀-➿]", "", s)   # spoken: no markdown/emoji
                print(s, end="", flush=True)
                split.feed(s)

            print("ARC: ", end="", flush=True)
            ring.show("speaking")
            replies = respond(body, hist, text, lang, on_text=on_text, owner=is_owner,
                              on_mood=lambda m: mood.update(mood=m))
            split.flush()
            if mood["mood"]:
                print(f"     (mood: {mood['mood']})", end="")
            while voice.first_audio is None and voice.speaking():
                time.sleep(0.02)
            first = (voice.first_audio or time.time()) - t0
            print(f"\n     [heard in {t_heard:.1f}s, first word {first:.1f}s after you stopped"
                  f"{' (acknowledged)' if ack else ''}]")
            r.log("talk", k=k, heard=text, lang=lang, owner=int(is_owner), voice_score=round(score, 3),
                  t_transcribe=round(t_heard, 2), t_first_audio=round(first, 2), ack=int(ack),
                  t_think=round(time.time() - t_think, 2), mood=mood["mood"] or "",
                  reply=" ".join(x["text"] for x in replies), act=replies[0].get("act") or "")
            k += 1
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
                w.setframerate(OUT_RATE)
                w.writeframes(pcm.tobytes())
    return 0


if __name__ == "__main__":
    sys.exit(main())
