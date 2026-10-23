"""Offline Wikipedia for ARC: search, read and pick the best paragraphs (no internet).

    python -m iro.wiki "who built the Hofburg?"     what ARC would read for this question

The encyclopedias are Kiwix ZIM files in ~/wiki (download.kiwix.org/zim/wikipedia/): each has its
own full-text index (Xapian), so finding articles takes milliseconds even in all of English
Wikipedia (49 GB). Steps for one question:
  1. keywords: ARC's own model turns the question (any language) into English search words, or
     NONE for small talk (greetings, thanks) so those skip the lookup
  2. search each encyclopedia (English, plus Malay for a Malay question); the article whose title
     IS the keywords comes first
  3. cut the top articles into paragraphs and rank them by meaning with EmbeddingGemma (a
     multilingual embedding model, so a Malay question finds an English paragraph)
  4. the best few paragraphs (with their article titles) go into ARC's prompt
"""

import html
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

WIKI = Path.home() / "wiki"
EMB_PORT = 8091
KEYWORDS_PROMPT = (
    "You turn a visitor's question to a museum robot into English search keywords for Wikipedia. "
    "Reply with 2 to 6 keywords only, most important first, proper names spelled as in English "
    "Wikipedia. If the message is only a greeting, thanks, goodbye, a request to perform or move, "
    "or a question about the robot itself, reply NONE.")


@lru_cache(maxsize=None)
def archives():
    """{lang: [Archive, ...]} of the ZIM files present, biggest first (a full wiki before a subset)."""
    from libzim.reader import Archive
    out = {}
    for p in sorted(WIKI.glob("wikipedia_*.zim"), key=lambda p: -p.stat().st_size):
        try:
            a = Archive(str(p))
        except RuntimeError:                       # still downloading
            continue
        if not a.has_fulltext_index:
            continue
        out.setdefault(p.name.split("_")[1], []).append(a)
    return out


# ------------------------------------------------------------------ reading articles

_TAG = re.compile(r"<[^>]+>")


def paragraphs(a, path, max_par=40):
    """Plain-text paragraphs of an article (no tables, references or infobox), with its title."""
    e = a.get_entry_by_path(path)
    if e.is_redirect:
        e = e.get_redirect_entry()
    raw = bytes(e.get_item().content).decode("utf-8", "ignore")
    raw = re.sub(r"<(table|style|script|sup|figure)\b.*?</\1>", " ", raw, flags=re.S | re.I)
    out = []
    for p in re.findall(r"<p\b[^>]*>(.*?)</p>", raw, flags=re.S | re.I):
        t = html.unescape(_TAG.sub("", p))
        t = re.sub(r"\[\d+\]|\s+", " ", t).strip()
        if len(t) > 60:
            out.append(t)
        if len(out) >= max_par:
            break
    return e.title, out


def search(query, lang="en", k=4):
    """[(archive, path)] of the best articles for `query` in the `lang` encyclopedias."""
    from libzim.search import Query, Searcher
    hits = []
    for a in archives().get(lang, []):
        # the article titled exactly like the keywords (e.g. "Wayang kulit") is almost always right
        for cand in (query, query.split(",")[0]):
            t = cand.strip().replace(" ", "_")
            for v in (t, t[:1].upper() + t[1:]):
                if a.has_entry_by_path(v):
                    hits.append((a, v))
                    break
        r = Searcher(a).search(Query().set_query(query.replace(",", " ")))
        hits += [(a, p) for p in r.getResults(0, k)]
        if hits:
            break                                  # the biggest encyclopedia that has it is enough
    seen, out = set(), []
    for a, p in hits:
        if p not in seen:
            seen.add(p)
            out.append((a, p))
    return out[:k]


# ------------------------------------------------------------------ meaning (embeddings)

def emb_healthy():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{EMB_PORT}/health", timeout=1) as r:
            return r.status == 200
    except Exception:                              # noqa: BLE001
        return False


def start_embedder():
    if emb_healthy():
        return
    from iro.arc import cfg
    c = cfg()
    exe = Path(os.path.expanduser(c["server"]["binary"]))
    model = Path(os.path.expanduser(c["model_dir"])) / c["embedder"]["file"]
    subprocess.Popen([str(exe), "-m", str(model), "--port", str(EMB_PORT), "--host", "127.0.0.1",
                      "--embedding", "-ngl", "99", "-c", "2048", "-b", "512", "-ub", "512", "--no-webui"],
                     stdout=open("/tmp/arc-embed.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while not emb_healthy():
        if time.time() - t0 > 60:
            raise RuntimeError("the embedding server did not start: /tmp/arc-embed.log")
        time.sleep(0.3)


def embed(texts):
    """Unit vectors (n, d) for texts, from EmbeddingGemma on the GPU."""
    req = urllib.request.Request(f"http://127.0.0.1:{EMB_PORT}/v1/embeddings",
                                 json.dumps({"input": texts}).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        v = np.array([d["embedding"] for d in json.load(r)["data"]], np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ------------------------------------------------------------------ the whole lookup

def keywords(question):
    """English search keywords from ARC's model, or None for small talk."""
    from iro.arc import url
    body = {"messages": [{"role": "system", "content": KEYWORDS_PROMPT}, {"role": "user", "content": question}],
            "max_tokens": 24, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False},
            "id_slot": 1, "cache_prompt": True}   # its own slot: ARC's big prompt stays cached in slot 0
    req = urllib.request.Request(url("/v1/chat/completions"), json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        kw = json.load(r)["choices"][0]["message"]["content"].strip().splitlines()[0]
    kw = re.sub(r"[\"'*`]", "", kw).strip(" .")
    return None if not kw or kw.upper().startswith("NONE") else kw


def lookup(question, lang="en", top=3, chunk_words=110):
    """{keywords, passages: [(title, text, score)], ms: {...}} for a question; passages empty if none."""
    t = {"t0": time.time()}
    kw = keywords(question)
    t["keywords"] = time.time()
    out = {"keywords": kw, "passages": [], "ms": {}}
    if kw and archives():
        langs = ["ms", "en"] if lang == "ms" else ([lang, "en"] if lang != "en" else ["en"])
        arts = []
        for lg in dict.fromkeys(langs):
            arts += search(kw, lg, k=3)
        t["search"] = time.time()
        chunks = []
        for a, p in arts:
            title, pars = paragraphs(a, p)
            buf = []
            for par in pars:                       # ~110-word chunks that keep whole sentences
                buf.append(par)
                if sum(len(b.split()) for b in buf) >= chunk_words:
                    chunks.append((title, " ".join(buf)))
                    buf = []
            if buf:
                chunks.append((title, " ".join(buf)))
        t["read"] = time.time()
        if chunks:
            start_embedder()
            q = embed([f"task: question answering | query: {question}"])[0]
            d = embed([f"title: {ti} | text: {tx}" for ti, tx in chunks[:48]])
            s = d @ q
            best = np.argsort(-s)[:top]
            out["passages"] = [(chunks[i][0], chunks[i][1], float(s[i])) for i in best]
        t["rank"] = time.time()
    keys = list(t)
    out["ms"] = {k: round((t[k] - t[keys[i - 1]]) * 1000) for i, k in enumerate(keys) if i}
    return out


def main():
    from iro.arc import start
    q = " ".join(sys.argv[1:]) or "Who built the Hofburg in Vienna?"
    start(quiet=True)
    print("encyclopedias:", {k: len(v) for k, v in archives().items()})
    r = lookup(q)
    print(f"keywords: {r['keywords']}   time ms: {r['ms']}")
    for ti, tx, s in r["passages"]:
        print(f"\n[{s:.2f}] {ti}: {tx[:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
