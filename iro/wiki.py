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
    from iro.hw.system import expendable
    expendable(subprocess.Popen([str(exe), "-m", str(model), "--port", str(EMB_PORT), "--host", "127.0.0.1",
                                 "--embedding", "-ngl", "99", "-c", "2048", "-b", "512", "-ub", "512", "--no-webui"],
                                stdout=open("/tmp/arc-embed.log", "w"), stderr=subprocess.STDOUT,
                                start_new_session=True))
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

_STOP = set("the a an and or of to in on at for is are was were be by with from as that this it what who when "
             "where which how why do does did can you me my your about tell please".split())


def bm25(docs, query, k1=1.5, b=0.75):
    """BM25 scores of docs for query (Robertson & Zaragoza): the classic search-engine ranking. Needs no
    model and no memory; the article search (Xapian) already found the right articles."""
    tok = lambda t: [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in _STOP]   # noqa: E731
    D = [tok(d) for d in docs]
    q = set(tok(query))
    N, avg = len(D), (sum(len(d) for d in D) / max(len(D), 1)) or 1.0
    df = {w: sum(w in set(d) for d in D) for w in q}
    out = np.zeros(N)
    for i, d in enumerate(D):
        for w in q:
            f = d.count(w)
            if f:
                idf = np.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
                out[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avg))
    return out


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
            from iro.arc import cfg
            if (cfg().get("embedder") or {}).get("enabled", False):
                start_embedder()
                q = embed([f"task: question answering | query: {question}"])[0]
                d = embed([f"title: {ti} | text: {tx}" for ti, tx in chunks[:48]])
                s = d @ q
            else:                                  # 23 Oct: the embedder's 0.78 GB did not fit with ARC's voice
                s = bm25([f"{ti} {tx}" for ti, tx in chunks[:48]], f"{kw} {question}")
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

# ------------------------------------------------------------------ ARC Lite: answer from the article's summary
# (24 Oct) Every Wikipedia article starts with a summary written by its editors: usually the exact
# answer. Found by TITLE from the question's own words (no model call: the keyword step took up to
# 7 s), disambiguation pages followed by context, 1-2 clean sentences returned in ~20-80 ms.

QWORDS = set("who whom whose what which when where why how is are was were do does did can could would will "
             "tell me about please explain describe the a an of in on at to for and or did you know i want "
             "know more give story history founded found built made invented discovered created wrote "
             "composed started began happened mean meaning called name named this that it its them those these "
             "sure okay ok yes yeah please something anything more like hear learn would want some little bit "
             "interesting interested".split())
CONTEXT = set("malaysia malaysian malay malacca melaka kelantan terengganu sultan sultanate king kingdom austria "
              "austrian vienna habsburg heritage tradition traditional culture history historical".split())
ABBREV = re.compile(r"\b(c|ca|St|Dr|Mr|Mrs|Ms|Jr|Sr|vs|e\.g|i\.e|U\.S|No|approx|est|b|d|r)\.$")


def _strip_parens(t):
    out, depth = [], 0
    for ch in t:
        if ch in "([":
            depth += 1
        elif ch in ")]" and depth:
            depth -= 1
        elif not depth:
            out.append(ch)
    return re.sub(r"\s+([,.;:])", r"\1", re.sub(r"\s{2,}", " ", "".join(out))).strip()


def sentences_of(t):
    parts, cur = [], ""
    for piece in re.split(r"(?<=[.!?])\s+", t):
        cur = f"{cur} {piece}".strip() if cur else piece
        if not ABBREV.search(cur):                 # "c. 1400" is not the end of a sentence
            parts.append(cur)
            cur = ""
    if cur:
        parts.append(cur)
    return parts


def topic_words(q):
    """The question's content words (no fillers, pronouns, question words): empty = no real topic."""
    return [w for w in re.findall(r"[A-Za-zÀ-ÿ'-]+", q) if w.lower() not in QWORDS]


def _candidates(q):
    """Title guesses from the question: "History of X" first when asked about history (24 Oct: "the history
    of America" became a lookup of "America"), then content-word n-grams, longest first."""
    content = topic_words(q)
    out = []
    m = re.search(r"\bhistory of (?:the )?([A-Za-zÀ-ÿ' -]+)", q, re.I)
    if m:
        x = " ".join(w for w in m.group(1).split() if w.lower() not in QWORDS)
        if x:
            out += [f"History of {x}", f"History of the {x}"]
    for n in range(min(4, len(content)), 0, -1):
        for i in range(len(content) - n + 1):
            g = content[i:i + n]
            out.append(" ".join(g))
    return out


def _stem(w):
    return w.lower()[:5]


# for SCORING, the question's verbs matter ("founded" finds the founding sentence, "composed" the
# composer): only true filler words are dropped (24 Oct: the 1511 sentence beat the founding one)
FILLER = set("who whom whose what which when where why how is are was were do does did can could would will "
             "tell me about please explain describe the a an of in on at to for and or you know i want".split())


def _qstems(q):
    return {_stem(w) for w in re.findall(r"[A-Za-zÀ-ÿ]+", q) if w.lower() not in FILLER and len(w) > 2}


def _resolve(a, path, q):
    """Follow redirects and disambiguation pages to one article: (title, html)."""
    for _ in range(3):
        e = a.get_entry_by_path(path)
        if e.is_redirect:
            e = e.get_redirect_entry()
        raw = bytes(e.get_item().content).decode("utf-8", "ignore")
        first = re.search(r"<p\b[^>]*>(.*?)</p>", raw, flags=re.S | re.I)
        if first and "may refer to" in first.group(1):
            qs = _qstems(q)
            ctx = {_stem(w) for w in CONTEXT}
            best, best_s = None, -1
            for href, li in re.findall(r'<li><a href="([^"#]+)"[^>]*>.*?</a>(.*?)</li>', raw, flags=re.S):
                st = {_stem(w) for w in re.findall(r"[a-z]+", (html.unescape(_TAG.sub(" ", li)) + " " +
                                                             href.replace("_", " ")).lower())}
                sc = 3 * len(qs & st) + len(ctx & st)  # the question's words count most; context breaks ties
                if sc > best_s:
                    best, best_s = href, sc
            if best and a.has_entry_by_path(best):
                path = best
                continue
        return e.title, raw
    return e.title, raw


def _lead_sentences(raw, paragraphs=3):
    raw = re.sub(r"<(table|style|script|sup|figure)\b.*?</\1>", " ", raw, flags=re.S | re.I)
    out = []
    for p in re.findall(r"<p\b[^>]*>(.*?)</p>", raw, flags=re.S | re.I):
        text = re.sub(r"\s+", " ", html.unescape(_TAG.sub("", p))).strip()
        if len(text) > 60:
            out += sentences_of(_strip_parens(re.sub(r"\[\d+\]", "", text)))
            paragraphs -= 1
            if not paragraphs:
                break
    return out


def summary(q, lang="en", n=2):
    """{title, text, ms} for a question: several candidate articles (title matches, "The ..." forms,
    full-text hits) are scored by how many of the question's words their opening contains; the
    answer is the first sentence (what it is) + the sentence that best matches the question
    (where the date / founder / composer usually is). None if nothing fits."""
    t0 = time.time()
    arcs = archives().get(lang) or archives().get("en") or []
    if not arcs:
        return None
    a = arcs[0]
    paths = []
    for cand in _candidates(q):
        forms = [cand.replace(" ", "_"), cand.title().replace(" ", "_"), cand.capitalize().replace(" ", "_")]
        forms += ["The_" + f for f in forms]
        for v in dict.fromkeys(forms):
            if len(v) > 2 and a.has_entry_by_path(v) and v not in paths:
                paths.append(v)
        if len(paths) >= 3:
            break
    words = " ".join(w for w in re.findall(r"\w+", q) if w.lower() not in QWORDS)
    if words:
        from libzim.search import Query, Searcher
        paths += [h for h in Searcher(a).search(Query().set_query(words)).getResults(0, 3) if h not in paths]
    qs = _qstems(q)
    best = None
    for rank, path in enumerate(paths[:6]):
        try:
            title, raw = _resolve(a, path, q)
        except Exception:                          # noqa: BLE001 - a broken entry: skip it
            continue
        sents = _lead_sentences(raw)
        if not sents:
            continue
        lead = " ".join(sents[:4]).lower()
        score = len(qs & {_stem(w) for w in re.findall(r"[a-z]+", lead)}) + \
            len(qs & {_stem(w) for w in re.findall(r"[a-z]+", title.lower())}) - 0.1 * rank
        if re.search(r"\((film|song|album|TV|band|novel|video game|soundtrack)", title):
            score -= 2                             # the thing itself, not a film named after it
        if best is None or score > best[0]:
            best = (score, title, sents)
    if best is None or best[0] <= 0:
        return None
    _, title, sents = best
    pick = [sents[0]]
    if n > 1 and len(sents) > 1:
        sc = [len(qs & {_stem(w) for w in re.findall(r"[A-Za-z]+", x)}) for x in sents[1:8]]
        if max(sc) > 0:
            pick.append(sents[1 + int(np.argmax(sc))])
        else:
            pick.append(sents[1])
    rest = [x for x in sents if x not in pick]
    return {"title": title, "text": " ".join(pick), "ms": round((time.time() - t0) * 1000),
            "sents": rest, "next": 0}

    a = arcs[0]
    path = None
    for cand in _candidates(q):
        for v in {cand.replace(" ", "_"), cand.title().replace(" ", "_"), cand.capitalize().replace(" ", "_")}:
            if len(v) > 2 and a.has_entry_by_path(v):
                path = v
                break
        if path:
            break
    if path is None:                               # no title match: the full-text index
        from libzim.search import Query, Searcher
        words = " ".join(w for w in re.findall(r"\w+", q) if w.lower() not in QWORDS)
        if not words:
            return None
        hits = list(Searcher(a).search(Query().set_query(words)).getResults(0, 1))
        if not hits:
            return None
        path = hits[0]
    title, raw = _resolve(a, path, q)
    raw = re.sub(r"<(table|style|script|sup|figure)\b.*?</\1>", " ", raw, flags=re.S | re.I)
    for p in re.findall(r"<p\b[^>]*>(.*?)</p>", raw, flags=re.S | re.I):
        text = re.sub(r"\s+", " ", html.unescape(_TAG.sub("", p))).strip()
        if len(text) > 60:
            text = _strip_parens(re.sub(r"\[\d+\]", "", text))
            return {"title": title, "text": " ".join(sentences_of(text)[:n]),
                    "ms": round((time.time() - t0) * 1000)}
    return None

