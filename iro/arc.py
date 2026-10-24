"""ARC (Autonomous Reasoning Companion): the robot's offline language model.

    iro arc chat [--lang ms|en|de] [--robot]   talk to ARC by typing; --robot: skills really move the
                                     arm (default: simulated, safe to try anywhere)
    iro arc routine teach|play|list NAME   hand-taught arm movements (greet, bow, perform)
    iro arc talk [--robot] [--en]    speak with ARC: mic array -> Whisper -> ARC -> its Qwen voice
    iro arc ask "question"           one question, one answer
    iro arc eval [MODEL ...]         run arc/eval.yaml on each model: speed, language, facts, honesty
    iro arc topic [NAME]             show / switch the sub-topic (arc/topics/NAME.yaml)
    iro arc check                    knowledge cards still to be verified by the team
    iro arc models                   the models in config/arc.yaml and which ones are downloaded
    iro arc stop                     stop all of ARC's servers: model, Whisper, voice (frees ~3 GB)

How it works: llama.cpp's llama-server runs the model on the GPU (localhost only). ARC's system
prompt = persona + the topic's role + ALL its knowledge cards; it is the same for every question,
so the server keeps it in its KV cache and each question only pays for itself (cache-augmented
generation: Chan et al. 2024, "Don't Do RAG"). Thinking mode is off: a spoken reply cannot wait.
Every turn is recorded (iro review): question, language, reply, action, timings.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from iro import ROOT

ARC = ROOT / "arc"
CFG = ROOT / "config" / "arc.yaml"
LANGS = {"ms": "Bahasa Melayu", "en": "English", "de": "German (Deutsch)"}
ACT = re.compile(r"^\s*<act>\s*([A-Za-z_]+(?::[A-Za-z0-9_]+)?)\s*</act>\s*")
MOODS = ("amused", "excited", "concerned", "warm", "curious", "proud")          # voices/<voice>-<mood>; no tag = calm (the main voice)
MOOD = re.compile(r"^\s*<mood>\s*(" + "|".join(MOODS) + r")\s*</mood>\s*")
MOOD_RULE = 'mood ::= "<mood>" (' + " | ".join(f'"{m}"' for m in MOODS) + ') "</mood> "\n'


def split_tags(raw):
    """(act, mood, spoken text, still_inside_a_tag) of a streamed reply that may start with
    <act>..</act> and/or <mood>..</mood>."""
    act = mood = None
    rest = raw
    m = ACT.match(rest)
    if m:
        act, rest = m.group(1), rest[m.end():]
    m = MOOD.match(rest)
    if m:
        mood, rest = m.group(1), rest[m.end():]
    pending = rest.lstrip().startswith("<") and len(rest) < 40
    return act, mood, rest, pending


def cfg():
    return yaml.safe_load(CFG.read_text())


def url(path=""):
    return f"http://127.0.0.1:{cfg()['server']['port']}{path}"


# ------------------------------------------------------------------ the prompt

def card(name):
    """(front matter dict, text) of arc/knowledge/NAME.md."""
    raw = (ARC / "knowledge" / f"{name}.md").read_text()
    _, head, body = raw.split("---", 2)
    return yaml.safe_load(head), body.strip()


def topic(name=None):
    return yaml.safe_load((ARC / "topics" / f"{name or cfg()['topic']}.yaml").read_text())


QUESTION = (r"\?|\b(what|who|whom|whose|when|where|why|how|which|tell me|explain|describe|apa|siapa|bila|"
            r"mengapa|kenapa|bagaimana|ceritakan|was|wer|wann|wo|warum|wie|erz(ä|a)hl|"
            # requests count too (24 Oct: "I would like to hear about the history of America" was not looked up)
            r"(i would|i'd|i want to|i wanna) (like to )?(hear|know|learn)|teach me|talk about|what about|"
            r"history of|do you know)\b")
MORE = re.compile(r"\b(more|go on|continue|and then|tell me about (this|that|it)|what else)\b", re.I)


def small_talk(q):
    """Greetings, thanks, goodbyes and very short remarks: no encyclopedia needed (24 Oct: "bye bye"
    triggered a full Wikipedia lookup and took 57 s)."""
    if re.search(QUESTION, q, re.I):
        return False
    return len(re.findall(r"\w+", q)) <= 3 or bool(re.search(INTENTS["greet"] + "|" + INTENTS["bow"], q, re.I))


def needs_lookup(q):
    """Will this question go to offline Wikipedia? (the voice says "let me think" right away if so)"""
    # only real questions: "I like your voice" went to Wikipedia and came back as a Korean TV show (24 Oct)
    from iro.wiki import topic_words
    if not cfg().get("wiki") or card_topic(q) or small_talk(q) or re.search(r"\b(you|your|yourself|arc)\b", q, re.I):
        return False
    if MORE.search(q) and LAST_FACT.get("sents") and not topic_words(q):
        return True                                # "tell me more" continues the last article
    return bool(re.search(QUESTION, q, re.I)) and bool(topic_words(q))


def card_topic(q):
    """The knowledge card whose names the question mentions, or None."""
    low = q.lower()
    for c in topic()["cards"]:
        meta, _ = card(c)
        if any(n.lower() in low for n in meta.get("names", []) if len(n) > 3):
            return c
    return None


def menu(name=None):
    """The skills ARC may use now: {tag: description} (iro/skills.py decides what exists)."""
    from iro.skills import Body
    return Body().menu(topic(name)["actions"])


def system_prompt(name=None):
    t = topic(name)
    persona = (ARC / "persona.md").read_text().strip()
    acts = "\n".join(f"- <act>{k}</act>: {v}" for k, v in menu(name).items())
    persona = persona.replace("ACTIONS", acts)
    cards = []
    for c in t["cards"]:
        meta, body = card(c)
        cards.append(f"## {meta['title']}\n{body}")
    return (f"{persona}\n\nYOUR ROLE TODAY\n{t['role'].strip()}\n{t['focus'].strip()}\n"
            f"Your visitors: {t['audience']}.\n\nHERITAGE KNOWLEDGE\n" + "\n\n".join(cards))


# ------------------------------------------------------------------ the server

def healthy():
    try:
        with urllib.request.urlopen(url("/health"), timeout=1) as r:
            return r.status == 200
    except Exception:                             # noqa: BLE001 - not up (yet)
        return False


def running_model():
    try:
        with urllib.request.urlopen(url("/v1/models"), timeout=2) as r:
            return Path(json.load(r)["data"][0]["id"]).name
    except Exception:                             # noqa: BLE001
        return None


def stop():
    subprocess.run(["pkill", "-f", "llama-server .*--port " + str(cfg()["server"]["port"])], check=False)
    for _ in range(50):
        if not healthy():
            return
        time.sleep(0.2)


def start(model=None, quiet=False):
    """Start llama-server with `model` (a key of config/arc.yaml models), unless it already runs."""
    c = cfg()
    model = model or c["model"]
    m = c["models"][model]
    path = Path(os.path.expanduser(c["model_dir"])) / m["file"]
    if healthy() and running_model() == path.name:
        return model
    stop()
    if not path.exists():
        raise SystemExit(f"{path.name} is not downloaded: iro arc models")
    s = c["server"]
    exe = Path(os.path.expanduser(s["binary"]))
    cmd = [str(exe), "-m", str(path), "--port", str(s["port"]), "--host", "127.0.0.1",
           "-c", str(s["ctx"]), "-ngl", "99", "-fa", "on", "-np", "2", "--jinja",
           "--cache-reuse", "256", "--no-webui"] + [str(a) for a in m.get("args", [])]
    log = open(Path("/tmp") / "arc-server.log", "w")
    from iro.hw.system import expendable
    expendable(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
    t0 = time.time()
    while not healthy():
        if time.time() - t0 > 180:
            raise SystemExit("the model server did not start in 3 min: see /tmp/arc-server.log")
        time.sleep(0.5)
    if c.get("wiki") and (c.get("embedder") or {}).get("enabled", False):   # off: BM25 ranks (no memory)
        from iro.wiki import start_embedder
        start_embedder()
    if not quiet:
        print(f"ARC model {model} loaded in {time.time() - t0:.0f} s")
    return model


# ------------------------------------------------------------------ asking

def ask(messages, lang=None, on_text=None, max_tokens=150, temperature=0.4, wiki=True, state=None,
        acts=True, _retry=True, on_mood=None, openers=None):
    """See _ask. If the model server died (23 Oct: it dropped a connection once during a long test),
    restart it and try once more: a silent robot in front of the judges is the worst outcome."""
    try:
        return _ask(messages, lang, on_text, max_tokens, temperature, wiki, state, acts, on_mood, openers)
    except (ConnectionError, urllib.error.URLError, OSError) as e:
        if not _retry:
            raise
        print(f"\n(model server problem: {e}; restarting it)")
        start(running_key() or cfg()["model"], quiet=True)
        return ask(messages, lang, on_text, max_tokens, temperature, wiki, state, acts, _retry=False, on_mood=on_mood,
                   openers=openers)


def running_key():
    """The config key of the model the server runs, if any."""
    name = running_model()
    return next((k for k, m in cfg()["models"].items() if m["file"] == name), None)


def _ask(messages, lang=None, on_text=None, max_tokens=150, temperature=0.4, wiki=True, state=None,
         acts=True, on_mood=None, openers=None):
    """Stream one reply. messages: chat history ending with the user's turn (system added here).
    Returns {text, act, lang, ttft, first_sentence, total, tokens, tok_s}. on_text(chunk) gets the
    spoken text as it streams (the action tag is removed first)."""
    t0 = time.time()                              # timings include the lookup: the visitor waits for both
    msgs = [{"role": "system", "content": system_prompt()}] + [dict(m) for m in messages]
    look = None
    if wiki and needs_lookup(messages[-1]["content"]):
        # offline Wikipedia: the best paragraphs for this question. Not when the team's cards cover
        # it: 23 Oct the Indonesian wayang article overrode the Kelantan card (and it costs ~2.4 s)
        # ARC Lite (24 Oct): the article's own summary, found by title in ~50-150 ms, replaces the
        # keyword model call + paragraph ranking (up to 7 s, and it often picked the wrong passage)
        from iro.wiki import summary
        look = summary(messages[-1]["content"], "en")
        if look:
            msgs[-1]["content"] += (f"\n\nREFERENCE (Wikipedia, {look['title']}): {look['text']}\n"
                                    "Answer from this reference in your own short spoken words.")
    if state:                                     # what the body is doing / holding / has seen
        msgs[-1]["content"] += f"\n\n(Robot status: {state}.)"
    if lang:                                      # the speech recogniser tells us the language
        msgs[-1]["content"] += f"\n\n(Reply in {LANGS[lang]}.)"
    body = {"messages": msgs, "stream": True, "max_tokens": max_tokens, "temperature": temperature,
            "top_p": 0.95, "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False},
            "id_slot": 0,                         # slot 0 keeps ARC's prompt cached; wiki keywords use slot 1
            # grammar: an optional <act> from the menu only, then text (no tag at all if acts=False)
            "grammar": _grammar(messages[-1]["content"], openers) if acts else 'root ::= mood? [^<]*\n' + MOOD_RULE,
            "repeat_penalty": 1.15, "repeat_last_n": 128,      # 23 Oct: 2B looped "Selamat datang!" x100
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url("/v1/chat/completions"), json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    ttft = first_sentence = None
    raw, spoken_upto, usage, timings = "", 0, {}, {}
    mood_told = False
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            timings = d.get("timings") or timings
            if not d.get("choices"):
                continue
            delta = d["choices"][0].get("delta", {}).get("content") or ""
            if not delta:
                continue
            if ttft is None:
                ttft = time.time() - t0
            raw += delta
            _, mood_now, text, pending = split_tags(raw)
            if pending:
                continue                          # still reading an <act> / <mood> tag
            if mood_now and on_mood and not mood_told:
                on_mood(mood_now)                 # the voice switches mood before the first word
                mood_told = True
            if first_sentence is None and re.search(r"[.!?](\s|$)", text):
                first_sentence = time.time() - t0  # TTS can start speaking here
            if on_text and len(text) > spoken_upto:
                on_text(text[spoken_upto:])
                spoken_upto = len(text)
    total = time.time() - t0
    act, mood, text, _ = split_tags(raw)
    text = re.sub(r"[*_#`]+|[\U0001F000-\U0001FAFF\u2600-\u27BF]", "", text).strip()  # spoken
    n = usage.get("completion_tokens") or timings.get("predicted_n") or 0
    return {"text": text, "act": act, "mood": mood, "lang": detect_lang(text),
            "ttft": ttft, "first_sentence": first_sentence or total, "total": total, "tokens": n,
            "tok_s": timings.get("predicted_per_second") or (n / max(total - (ttft or 0), 1e-3)),
            "prompt_ms": timings.get("prompt_ms"), "cached": timings.get("cache_n"),
            "lookup": look and {"title": look["title"], "ms": look["ms"]}}


# Every reply starts with one of these short openers (24 Oct): the voice has them pre-recorded in
# every mood, so ARC is heard ~0.3 s after the model's first word instead of ~2 s (synthesising the
# first phrase while the model still uses the GPU). "Preface generation" for dialogue robots.
OPENERS = {
    "visitor": ["Ah,", "Well,", "Oh,", "Hmm,", "Hello!", "Hello and welcome!", "Splendid!", "Of course!",
                "Wonderful!", "I see.", "Ah, good question.", "Oh, I'm sorry to hear that."],
    "owner": ["Certainly, sir.", "Right, sir.", "Well, sir,", "Ah, sir,", "Oh dear, sir.", "Of course, sir.",
              "Splendid, sir!", "Hmm, sir,", "Good to hear you, sir.", "As you wish, sir."],
}


def _openers_for(q, owner):
    """Greeting openers only when the visitor greets (24 Oct: "Tell me about wayang kulit" got "Hello!")."""
    greeted = bool(re.search(INTENTS["greet"], q, re.I))
    return [o for o in OPENERS["owner" if owner else "visitor"] if greeted or not o.lower().startswith(("hello", "good to"))]


def _opener_rule(openers):
    return "opener ::= " + " | ".join(json.dumps(o) for o in openers) + "\n"


def _grammar(text, openers=None):
    """Only the actions this message asks for (INTENTS); with none, the grammar allows no tag at all.
    With openers: the spoken text must start with one of them."""
    from iro.skills import Body
    acts = {k: v for k, v in topic()["actions"].items() if re.search(INTENTS.get(k, r"$^"), text, re.I)}
    text_rule = 'text ::= opener " " [^<]*\n' + _opener_rule(openers) if openers else 'text ::= [^<]*\n'
    if not acts:
        return 'root ::= mood? text\n' + text_rule + MOOD_RULE
    g = Body().grammar(acts).replace("root ::= act? text", "root ::= act? mood? text")
    return g.replace('text ::= [^<]*\n', text_rule) + MOOD_RULE


# Only in owner turns (23 Oct: with both modes in the persona, the 2B model called visitors "sir").
OWNER = ("Speaker: your OWNER, the student who built you (recognised by his voice). OWNER MODE for this reply: "
         "you are his loyal but cheeky robot butler. Call him sir. Be short and straight to the point, dry and "
         "witty, a little sarcastic; an affectionate roast now and then (his sleep, his code, how hard he works "
         "you), never hurtful. Never repeat yourself. Orders (perform, look, show, stop) you simply do")
VISITOR = "Speaker: a visitor (be the warm storyteller; do not call them sir)"
# Each action is offered only when the visitor's words ask for it (23 Oct: the 2B model performed and
# waved at plain questions). BM / EN / DE trigger words.
INTENTS = {
    "greet": r"\b(hello|hi|hey|good (morning|afternoon|evening)|selamat|halo|hallo|guten (tag|morgen))\b",
    "bow": r"\b(bye|goodbye|thank|thanks|terima kasih|danke|auf wiedersehen|see you|selamat tinggal)\b",
    "perform": r"\b(perform|performance|dance|puppet show|show (me|us) (a|the|your) (move|show|dance|puppet)|"
               r"persembahan|tunjuk(kan)? persembahan|menari|vorf(ü|u)hr|tanz)",
    "look": r"\b(what is this|what's this|what is that|look at|recogni[sz]e|apa (ini|itu)|benda (ini|apa)|"
            r"was ist das|schau)",
    "show": r"\b(pick (it |this |that |the )?up|show me the|hold up|ambil|angkat|tunjukkan|heb|zeig mir)",
}



def respond(body, hist, q, lang=None, on_text=None, log=None, owner=False, on_mood=None):
    """One visitor turn: safety words first (no model), then ARC's reply; a chosen skill runs while
    the reply is spoken, and if it reports something (what it saw, whether the pick worked) ARC
    tells the visitor in a short second reply. Returns the list of replies."""
    from iro.skills import is_stop
    if is_stop(q):
        body.stop()
        say = {"ms": "Baik, saya berhenti.", "de": "In Ordnung, ich halte an."}.get(lang, "Okay, I am stopping.")
        on_text and on_text(say)
        return [{"text": say, "act": "stop", "first_sentence": 0.0, "total": 0.0, "ttft": 0.0, "tok_s": 0}]
    hist.append({"role": "user", "content": q})
    who = OWNER if owner else VISITOR
    # look / show: a fixed first line, then the skill, then the model speaks from the REAL result
    # (23 Oct: both 2B and 4B "saw" a doll / a USB drive before the camera had looked)
    sensing = next((k for k in ("look", "show") if re.search(INTENTS[k], q, re.I)), None)
    # facts (ARC Lite, 24 Oct): the Wikipedia sentence itself, spoken with a short opener. The 2B
    # model ignored a correct reference and asked "Who did you think founded Malacca?"
    fact = None if sensing or not needs_lookup(q) else _fact(q, owner)
    if fact:
        on_mood and on_mood(None)
        on_text and on_text(fact["text"])
        a = {"text": fact["text"], "act": None, "mood": None, "first_sentence": 0.0, "total": 0.0, "ttft": 0.0,
             "tok_s": 0, "lookup": {"title": fact["title"], "ms": fact["ms"]}}
        hist.append({"role": "assistant", "content": fact["text"]})
        out = [a]
        sensing = "fact"
    if sensing and sensing != "fact":
        tag = "look" if sensing == "look" else next((t for t in body.menu(topic()["actions"]) if t.startswith("show:")
                                                     and _object_named(t, q)), None)
        if tag:
            line = {"ms": "Sebentar, biar saya lihat.", "de": "Moment, ich schaue mal."}.get(
                lang, "Let me take a look." if tag == "look" else "One moment, let me get it.")
            on_text and on_text(line)
            a = {"text": line, "act": tag, "first_sentence": 0.0, "total": 0.0, "ttft": 0.0, "tok_s": 0}
            hist.append({"role": "assistant", "content": f"<act>{tag}</act> {line}"})
            out = [a]
        else:
            sensing = None
    if not sensing:
        a = ask(hist[-7:], lang=lang, on_text=on_text, state=f"{body.state()}. {who}", on_mood=on_mood,
                openers=_openers_for(q, owner))
        hist.append({"role": "assistant", "content": a["text"]})
        out = [a]
    if a["act"]:
        report = body.run(a["act"])
        a["report"] = report
        if a["act"].split(":")[0] in ("look", "show") or report.startswith("could not"):
            on_text and on_text("\n")
            hist.append({"role": "user", "content": f"(Robot report: {report}. Tell the speaker in one or "
                                                    "two sentences what really happened, using ONLY this report; "
                                                    "if an object was recognised, say what it is and one fact "
                                                    "from your knowledge. If nothing was recognised, say so.)"})
            b = ask(hist[-7:], lang=lang, on_text=on_text, wiki=False, acts=False, state=f"{body.state()}. {who}",
                    on_mood=on_mood)
            hist.append({"role": "assistant", "content": b["text"]})
            out.append(b)
    if log:
        for x in out:
            log("turns", q=q, lang=lang or "", reply=x["text"], act=x.get("act") or "", report=x.get("report", ""),
                ttft=round(x.get("ttft") or 0, 3), first_sentence=round(x["first_sentence"], 3),
                total=round(x["total"], 3), tok_s=round(x.get("tok_s") or 0, 1),
                lookup=json.dumps(x.get("lookup"), ensure_ascii=False))
    return out


FACT_OPENERS = {"visitor": ["Ah, good question.", "Well,", "Here is what I know.", "Let me tell you."],
                "owner": ["Right, sir.", "Certainly, sir.", "Well, sir,", "As it happens, sir,"]}


LAST_FACT = {}                                     # the article ARC last answered from (for "tell me more")


def _fact(q, owner):
    """{text, title, ms}: a spoken answer made directly from the article's summary, or None.
    "Tell me more" (no new topic) continues the same article with its next sentences."""
    import random
    from iro.wiki import summary, topic_words
    if MORE.search(q) and LAST_FACT.get("sents") and not topic_words(q):
        i = LAST_FACT["pos"]
        nxt = LAST_FACT["sents"][i:i + 2]
        if not nxt:
            return {"text": "That is all my encyclopedia says in its summary, I'm afraid.", "title": LAST_FACT["title"],
                    "ms": 0}
        LAST_FACT["pos"] = i + 2
        s = {"title": LAST_FACT["title"], "text": " ".join(nxt), "ms": 0}
    else:
        s = summary(q, "en")
        if not s:
            return None
        LAST_FACT.update(title=s["title"], sents=s.get("sents", []), pos=s.get("next", 2))
    text = s["text"]
    if len(text) > 330:                            # keep it speakable: the first two sentences, trimmed
        text = text[:330].rsplit(",", 1)[0].rstrip(" ,;") + "."
    opener = random.choice(FACT_OPENERS["owner" if owner else "visitor"])
    sep = " " if opener.endswith(",") else " "
    return {"text": f"{opener}{sep}{text}", "title": s["title"], "ms": s["ms"]}


def _object_named(tag, q):
    """Does the question name this object (its name or an everyday name from objects/NAME/about.yaml)?"""
    from iro.skills import aka
    words = re.findall(r"[a-zäöü]+", aka(tag.split(":", 1)[1]).lower())
    return any(w in q.lower() for w in words if len(w) > 3)


# A tiny language guesser for the three languages ARC speaks (enough to check a reply's language).
_WORDS = {
    "ms": "yang dan ini itu dengan untuk dalam adalah tidak saya awak anda kita ialah akan boleh "
          "juga oleh kepada mereka sebuah satu dari pada atau seperti cerita warisan selamat datang",
    "en": "the and is are this that with for you your of to in it was what not can will be a an "
          "who story heritage welcome hello",
    "de": "der die das und ist sind nicht ich sie ein eine mit für von zu den dem des auch wie "
          "es geschichte willkommen hallo",
}
_SETS = {k: set(v.split()) for k, v in _WORDS.items()}


def detect_lang(text):
    w = re.findall(r"[a-zäöüß]+", text.lower())
    score = {k: sum(x in s for x in w) for k, s in _SETS.items()}
    best = max(score, key=score.get)
    return best if score[best] else None


# ------------------------------------------------------------------ commands

def cmd_chat(lang=None, robot=False):
    from iro.record import run
    from iro.skills import Body
    r = run("arc-chat")
    body = Body(robot=robot)
    start()
    t = topic()
    print(f"ARC ({cfg()['model']}, topic {t['name']}). Type a question; empty line or q to stop.")
    print(f"ARC: {t['greeting'].get(lang or 'en')}")
    hist = []
    while True:
        try:
            q = input("\nyou: ").strip()
        except EOFError:
            break
        if q in ("", "q"):
            break
        print("ARC: ", end="", flush=True)
        for a in respond(body, hist, q, lang, on_text=lambda s: print(s, end="", flush=True), log=r.log):
            lk = a.get("lookup") or {}
            print(f"\n     [{'act: ' + a['act'] + ' -> ' + a.get('report', '') + ' | ' if a.get('act') else ''}"
                  f"{'wiki: ' + str(lk.get('titles')) + ' | ' if lk.get('titles') else ''}"
                  f"first sentence {a['first_sentence']:.2f}s, all {a['total']:.1f}s, {a.get('tok_s') or 0:.0f} tok/s]")
    return 0


def cmd_eval(models):
    """Every question in arc/eval.yaml on every model; report + raw answers in the recording.
    --nowiki: without the offline Wikipedia (the model's own memory only), to compare."""
    from iro.record import run
    c = cfg()
    use_wiki = "--nowiki" not in models
    models = [m for m in models if not m.startswith("--")]
    models = models or list(c["models"])
    qs = yaml.safe_load((ARC / "eval.yaml").read_text())
    r = run("arc-eval" + ("" if use_wiki else "-nowiki"))
    summary = {}
    for model in models:
        path = Path(os.path.expanduser(c["model_dir"])) / c["models"][model]["file"]
        if not path.exists():
            print(f"{model}: not downloaded, skipped")
            continue
        t0 = time.time()
        start(model, quiet=True)
        load_s = time.time() - t0
        ask([{"role": "user", "content": "Hello"}], max_tokens=8, wiki=False)   # system prompt into the cache
        res = []
        for i, e in enumerate(qs):
            a = ask([{"role": "user", "content": e["q"]}], lang=e["lang"], wiki=use_wiki)
            low = a["text"].lower()
            ok_must = not e.get("must") or any(str(m).lower() in low for m in e["must"])
            ok_never = not any(re.search(rf"\b{re.escape(str(n).lower())}", low) for n in e.get("never", []))
            ok_lang = a["lang"] == e["lang"]
            ok_act = ("act" not in e) or (a["act"] == (None if e["act"] == "none" else e["act"]))
            ok = ok_must and ok_never and ok_lang and ok_act
            res.append({**e, **a, "ok": ok, "ok_must": ok_must, "ok_never": ok_never, "ok_lang": ok_lang,
                        "ok_act": ok_act})
            r.log(f"eval_{model}", i=i, kind=e["kind"], lang=e["lang"], ok=ok, ok_lang=ok_lang, ok_must=ok_must,
                  ok_never=ok_never, ok_act=ok_act, act=a["act"] or "", ttft=round(a["ttft"] or 0, 3),
                  first_sentence=round(a["first_sentence"], 3), total=round(a["total"], 2),
                  tok_s=round(a["tok_s"], 1), q=e["q"], reply=a["text"])
            print(f"  {model:12s} {i + 1:2d}/{len(qs)} {'ok ' if ok else 'BAD'} {e['kind']:8s} {e['lang']} "
                  f"{a['first_sentence']:4.1f}s  {a['text'][:90]!r}")
        import numpy as np
        fs = np.array([x["first_sentence"] for x in res])
        summary[model] = {
            "load_s": round(load_s, 1), "passed": f"{sum(x['ok'] for x in res)}/{len(res)}",
            "right_language": f"{sum(x['ok_lang'] for x in res)}/{len(res)}",
            "facts": f"{sum(x['ok_must'] for x in res if x['kind'] in ('fact', 'story', 'self'))}/"
                     f"{sum(x['kind'] in ('fact', 'story', 'self') for x in res)}",
            "world_history": f"{sum(x['ok'] for x in res if x['kind'] == 'world')}/"
                             f"{sum(x['kind'] == 'world' for x in res)}",
            "honest_when_unsure": f"{sum(x['ok'] for x in res if x['kind'] in ('unknown', 'trap'))}/"
                                  f"{sum(x['kind'] in ('unknown', 'trap') for x in res)}",
            "actions": f"{sum(x['ok_act'] for x in res if 'act' in x)}/{sum('act' in x for x in res)}",
            "first_sentence_s_median": round(float(np.median(fs)), 2),
            "first_sentence_s_worst": round(float(fs.max()), 2),
            "tok_s": round(float(np.median([x["tok_s"] for x in res])), 1),
            "by_language": {lg: f"{sum(x['ok'] for x in res if x['lang'] == lg)}/{sum(x['lang'] == lg for x in res)}"
                            for lg in LANGS},
        }
        (r.dir / f"answers_{model}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    (r.dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))
    print("\n" + yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))
    print(f"Answers per model: {r.dir}/answers_MODEL.json")
    return 0


def cmd_topic(name=None):
    c = cfg()
    if name:
        topic(name)                               # exists and parses
        c["topic"] = name
        CFG.write_text(yaml.safe_dump(c, sort_keys=False))
        print(f"Topic now: {name}. The next question reloads ARC's knowledge (first answer ~5 s slower).")
    t = topic()
    print(f"Topic {t['name']}: {len(t['cards'])} knowledge cards, ~{len(system_prompt()) // 4} tokens of prompt")
    for x in sorted(p.stem for p in (ARC / "topics").glob("*.yaml")):
        print(("* " if x == t["name"] else "  ") + x)
    return 0


def cmd_check():
    bad = []
    for p in sorted((ARC / "knowledge").glob("*.md")):
        meta, _ = card(p.stem)
        if not meta.get("verified") or not meta.get("source"):
            bad.append(p.stem)
    used = set(topic()["cards"])
    for b in bad:
        print(f"  NOT VERIFIED{' (in use)' if b in used else ''}: arc/knowledge/{b}.md")
    print(f"{len(bad)} card(s) to check against a source; set verified: true and source: \"...\".")
    return 1 if bad else 0


def cmd_models():
    c = cfg()
    d = Path(os.path.expanduser(c["model_dir"]))
    for k, m in c["models"].items():
        p = d / m["file"]
        print(f"{'*' if k == c['model'] else ' '} {k:12s} {'ok ' if p.exists() else '-- '} "
              f"{p.stat().st_size / 1e9 if p.exists() else 0:4.1f} GB  {m.get('note', '')}")
        if not p.exists():
            print(f"      curl -L -o {p} {m['url']}")
    return 0


def main():
    a = sys.argv[1:]
    sub = a[0] if a else "chat"
    if sub == "chat":
        lang = a[a.index("--lang") + 1] if "--lang" in a else None
        return cmd_chat(lang, robot="--robot" in a)
    if sub == "talk":
        from iro.talk import main as talk_main
        return talk_main()
    if sub == "routine":
        from iro.skills import main as skills_main
        sys.argv = [sys.argv[0]] + a[1:]
        return skills_main()
    if sub == "ask":
        start(quiet=True)
        x = ask([{"role": "user", "content": " ".join(a[1:])}], on_text=lambda s: print(s, end="", flush=True))
        print(f"\n[{x['act'] or 'no action'}, {x['first_sentence']:.2f}s to first sentence]")
        return 0
    if sub == "eval":
        return cmd_eval(a[1:])
    if sub == "topic":
        return cmd_topic(a[1] if len(a) > 1 else None)
    if sub == "check":
        return cmd_check()
    if sub == "models":
        return cmd_models()
    if sub == "stop":                              # everything ARC started: frees ~3 GB
        stop()
        for name in ("llama-server", "whisper-server", "tts-server"):
            subprocess.run(["pkill", "-x", name], check=False)
        print("ARC stopped: language model, Wikipedia helper, Whisper and the voice server")
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
