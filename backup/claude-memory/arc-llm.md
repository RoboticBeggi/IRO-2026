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

**Voice (23 Oct, user: English first, "must go global"):** benchmarked on the Jetson (tools/voice_bench.py, stt_bench.py; test venvs ~/.venvs/voice and ~/.venvs/qwentts, models ~/voice-models). TTS: Kokoro-82M GPU (onnxruntime-gpu from the Jetson index) first sentence 0.55 s, RTF 0.16-0.21 — only real-time option; on CPU RTF 1.1 (too slow). Chatterbox-Turbo 4.5 s / RTF 2.1 / 4.1 GB; Qwen3-TTS 0.6B 23-36 s / RTF 8-10 — not live-capable. STT: Whisper large-v3-turbo q8 via whisper.cpp CUDA (~/whisper.cpp) ~1.1 s per sentence loaded, 99 languages, `--prompt` with names fixes "Tok Dalang"; ARC heard as "Ark" -> word fix. Parakeet-TDT 0.6B v2 int8 CPU 2 s, English only. User still to pick the voice by ear (tools/listen_voices.sh, via reSpeaker 3.5 mm).

**Voice decision (23 Oct): user wants Qwen3-TTS voices only** ("they sound natural"; wants to choose: gentleman/JARVIS-like butler, expressive, a bit quirky — original characters, no real-person imitation). Runtime: qwentts.cpp (~/src/qwentts.cpp, CUDA sm_87 build, GGUFs from Serveurperso/Qwen3-TTS-GGUF in models/): 0.6B customvoice Q8_0 = RTF 0.935, first audio frame 262 ms, 18.6 ms/frame talker + 40 ms/frame code predictor (the bottleneck), +~6 s model load (use tts-server to keep loaded). PyTorch qwen-tts was RTF 8-10 (dead end; also OOMed on 1.7B with a build running). `iro voice` (iro/voice.py): presets / design / keep / use / say; kept voices in repo voices/NAME (ref.wav+txt, .spk/.rvq via qwen-codec), cloned with 0.6B base for consistency. Output: HDMI sink (config/voice.yaml), default restored in ~/.config/pulse/default.pa.
