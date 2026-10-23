"""ARC (Autonomous Reasoning Companion): the robot's offline language model.

    iro arc chat [--lang ms|en|de] [--robot]   talk to ARC by typing; --robot: skills really move the
                                     arm (default: simulated, safe to try anywhere)
    iro arc routine teach|play|list NAME   hand-taught arm movements (greet, bow, perform)
    iro arc ask "question"           one question, one answer
    iro arc eval [MODEL ...]         run arc/eval.yaml on each model: speed, language, facts, honesty
    iro arc topic [NAME]             show / switch the sub-topic (arc/topics/NAME.yaml)
    iro arc check                    knowledge cards still to be verified by the team
    iro arc models                   the models in config/arc.yaml and which ones are downloaded
    iro arc stop                     stop the model server (frees ~3 GB)

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
    subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while not healthy():
        if time.time() - t0 > 180:
            raise SystemExit("the model server did not start in 3 min: see /tmp/arc-server.log")
        time.sleep(0.5)
    if c.get("wiki"):                             # the embedder too: starting it mid-question cost 5.8 s
        from iro.wiki import start_embedder
        start_embedder()
    if not quiet:
        print(f"ARC model {model} loaded in {time.time() - t0:.0f} s")
    return model


# ------------------------------------------------------------------ asking

def ask(messages, lang=None, on_text=None, max_tokens=220, temperature=0.4, wiki=True, state=None,
        acts=True, _retry=True):
    """See _ask. If the model server died (23 Oct: it dropped a connection once during a long test),
    restart it and try once more: a silent robot in front of the judges is the worst outcome."""
    try:
        return _ask(messages, lang, on_text, max_tokens, temperature, wiki, state, acts)
    except (ConnectionError, urllib.error.URLError, OSError) as e:
        if not _retry:
            raise
        print(f"\n(model server problem: {e}; restarting it)")
        start(running_key() or cfg()["model"], quiet=True)
        return ask(messages, lang, on_text, max_tokens, temperature, wiki, state, acts, _retry=False)


def running_key():
    """The config key of the model the server runs, if any."""
    name = running_model()
    return next((k for k, m in cfg()["models"].items() if m["file"] == name), None)


def _ask(messages, lang=None, on_text=None, max_tokens=220, temperature=0.4, wiki=True, state=None,
         acts=True):
    """Stream one reply. messages: chat history ending with the user's turn (system added here).
    Returns {text, act, lang, ttft, first_sentence, total, tokens, tok_s}. on_text(chunk) gets the
    spoken text as it streams (the action tag is removed first)."""
    t0 = time.time()                              # timings include the lookup: the visitor waits for both
    msgs = [{"role": "system", "content": system_prompt()}] + [dict(m) for m in messages]
    look = None
    if wiki and cfg().get("wiki"):                # offline Wikipedia: the best paragraphs for this question
        from iro.wiki import lookup
        look = lookup(messages[-1]["content"], lang or "en")
        if look["passages"]:
            ref = "\n\n".join(f"[{ti}] {tx}" for ti, tx, _ in look["passages"])
            msgs[-1]["content"] += f"\n\nREFERENCE (offline encyclopedia; use only what answers the question):\n{ref}"
    if state:                                     # what the body is doing / holding / has seen
        msgs[-1]["content"] += f"\n\n(Robot status: {state}.)"
    if lang:                                      # the speech recogniser tells us the language
        msgs[-1]["content"] += f"\n\n(Reply in {LANGS[lang]}.)"
    body = {"messages": msgs, "stream": True, "max_tokens": max_tokens, "temperature": temperature,
            "top_p": 0.95, "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False},
            "id_slot": 0,                         # slot 0 keeps ARC's prompt cached; wiki keywords use slot 1
            # grammar: an optional <act> from the menu only, then text (no tag at all if acts=False)
            "grammar": _grammar() if acts else 'root ::= [^<]*\n',
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url("/v1/chat/completions"), json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    ttft = first_sentence = None
    raw, spoken_upto, usage, timings = "", 0, {}, {}
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
            if raw.lstrip().startswith("<") and "</act>" not in raw and len(raw) < 40:
                continue                          # still reading the action tag
            text = ACT.sub("", raw, count=1)
            if first_sentence is None and re.search(r"[.!?](\s|$)", text):
                first_sentence = time.time() - t0  # TTS can start speaking here
            if on_text and len(text) > spoken_upto:
                on_text(text[spoken_upto:])
                spoken_upto = len(text)
    total = time.time() - t0
    m = ACT.match(raw)
    text = ACT.sub("", raw, count=1).strip()
    n = usage.get("completion_tokens") or timings.get("predicted_n") or 0
    return {"text": text, "act": m.group(1) if m else None, "lang": detect_lang(text),
            "ttft": ttft, "first_sentence": first_sentence or total, "total": total, "tokens": n,
            "tok_s": timings.get("predicted_per_second") or (n / max(total - (ttft or 0), 1e-3)),
            "prompt_ms": timings.get("prompt_ms"), "cached": timings.get("cache_n"),
            "lookup": look and {"keywords": look["keywords"], "ms": look["ms"],
                                "titles": [ti for ti, _, _ in look["passages"]]}}


def _grammar():
    from iro.skills import Body
    return Body().grammar(topic()["actions"])


def respond(body, hist, q, lang=None, on_text=None, log=None):
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
    a = ask(hist[-7:], lang=lang, on_text=on_text, state=body.state())
    hist.append({"role": "assistant", "content": a["text"]})
    out = [a]
    if a["act"]:
        report = body.run(a["act"])
        a["report"] = report
        if a["act"].split(":")[0] in ("look", "show") or report.startswith("could not"):
            on_text and on_text("\n")
            hist.append({"role": "user", "content": f"(Robot report: {report}. Tell the visitor in one or "
                                                    "two sentences; if you recognised an object, say what it is "
                                                    "and one interesting fact about it.)"})
            b = ask(hist[-7:], lang=lang, on_text=on_text, wiki=False, acts=False, state=body.state())
            hist.append({"role": "assistant", "content": b["text"]})
            out.append(b)
    if log:
        for x in out:
            log("turns", q=q, lang=lang or "", reply=x["text"], act=x.get("act") or "", report=x.get("report", ""),
                ttft=round(x.get("ttft") or 0, 3), first_sentence=round(x["first_sentence"], 3),
                total=round(x["total"], 3), tok_s=round(x.get("tok_s") or 0, 1),
                lookup=json.dumps(x.get("lookup"), ensure_ascii=False))
    return out


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
    if sub == "stop":
        stop()
        print("ARC model server stopped")
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
