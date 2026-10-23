---
name: arc-llm
description: "ARC (Autonomous Reasoning Companion) = the robot's offline LLM; design decisions, files, open items (Oct 2026)"
metadata:
  node_type: memory
  type: project
  originSessionId: 1cabac2b-8711-4426-81e7-373cd5327053
  modified: 2026-10-23T00:37:16.317Z
---

User named the robot's LLM **ARC – Autonomous Reasoning Companion** (23 Oct 2026). Must run fully offline on the Jetson (venue has no internet, rules forbid notes/internet), speak BM/EN/DE, and be "knowledgeable of everything in history" (user wants it as close to Claude/Gemini as possible).

Design (built 23 Oct): llama.cpp `llama-server` (~/llama.cpp, CUDA sm_87), models in ~/llm-models (outside git), config/arc.yaml. Candidates benchmarked with `iro arc eval`: Gemma-SEA-LION-v4.5-E2B (Malay-tuned Gemma 4 E2B, default), gemma-4-E2B, Qwen3.5-4B, Qwen3.5-2B. Team-verified fact cards in arc/knowledge (cache-augmented: whole topic pack in the cached system prompt) + offline Wikipedia via Kiwix ZIM + libzim (~/wiki, iro/wiki.py: LLM keywords -> Xapian search -> EmbeddingGemma-300M rerank). Skills (iro/skills.py): grammar-constrained `<act>` tags from a menu (greet/bow/perform keyframe routines, look = instant recognition, show:NAME = iro pick --drop), status line each turn, stop words bypass the LLM, skill reports fed back.

**Why:** small models hallucinate; judges' Q&A punishes wrong facts (e.g. sources disagree whether Wayang Kulit Kelantan is UNESCO-listed). Memory (7.6 GB shared) is the limit, not disk.

**How to apply:** never put unsourced facts in arc/knowledge (cards stay verified:false until the team checks); keep the system prompt stable (KV cache); 2 server slots (answers / wiki keywords). Open: speech in/out (STT/TTS) not built yet; rule 2.3 / "no notes" question to the IRO Secretariat should also cover an on-robot offline encyclopedia. Related: [[record-and-review]], [[jetson-hardware-state]].
