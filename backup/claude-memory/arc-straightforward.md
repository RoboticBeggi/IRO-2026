---
name: arc-straightforward
description: "ARC must be absolutely straightforward in every mood: short direct answers, no yapping, jokes or questions back"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 1cabac2b-8711-4426-81e7-373cd5327053
  modified: 2026-10-24T08:24:48.926Z
---

ARC's replies must be absolutely straight to the point, in any mood, for owner and visitors alike: the first sentence answers, 1-2 short sentences (persona at most 25 words; Splitter MAX_WORDS 28). No small talk, jokes, roasting, filler, guessing "did you mean", or questions back. Owner mode is "brief, plain, direct" (the earlier cheeky-butler/roast style was dropped 24 Oct).

**Why:** on 24 Oct the owner got rambling, made-up replies ("obedient butler", "Malaysian shadow puppet cuisine", "let me look with my wrist camera") and called them unacceptable: "i dont want to hear yapping a lot".

**How to apply:** prefer deterministic answers (heritage card lines `_card_fact`, Wikipedia `_fact`, `_math`, `_courtesy`) over free LLM text. Keep the persona short and strict. Check voice.yaml `reply_words` hasn't been raised by a misheard "more detail". Related: [[arc-llm]].
