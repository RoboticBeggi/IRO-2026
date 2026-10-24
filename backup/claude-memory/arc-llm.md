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

**23 Oct evening:** memory budget (tools/mem_budget.sh, desktop+VS Code use ~3.3 GB): SEA-LION E2B 3.2 GB (per_layer_token_embd 1.9 GB; -ot to CPU saves nothing on Jetson) -> switched ARC to **Qwen3.5-2B** (1.25 GB, ctx 8192) so Whisper turbo (1.19 GB) + embedder (0.76 GB, batch 512) + Qwen TTS fit with ~1 GB spare. 4B not clearly better (also hallucinated) and crashed on memory. Fixes for the 2B: actions offered only when INTENTS words match (arc.py), repeat_penalty 1.15, look/show = fixed first line then answer only from the skill report, markdown/emoji stripped. Owner mode: iro/speaker.py (WeSpeaker ResNet34 via sherpa-onnx, CPU, enroll/whoami; voiceprint config/owner_voice.npz gitignored; threshold >= 0.70, other voices scored up to 0.68); OWNER text only in owner turns (calls him sir, cheeky affectionate roasts — user asked "cruel sometimes", kept playful, never hurtful). Voice loop: iro/talk.py (`iro arc talk`). USB lesson: devices missing all day = hub/cable/arm-board issues, fixed by replugging via the Pi hubs; tools/usb_ports.sh maps sockets.

**24 Oct — memory/OOM rule:** the full ARC stack (Qwen3.5-2B ctx 6144 + Whisper turbo q5 + Qwen TTS KV 1024) plus VS Code on the Jetson ran it out of memory; the OOM killer took VS Code (the user's session). Now: every ARC server is spawned via iro.hw.system.expendable() (oom_score_adj 1000, dies first), `iro arc talk` refuses to start under 3000 MB free (--force), `iro arc stop` kills llama/whisper/tts servers. Do NOT start the full stack + tests while the user works in VS Code on the Jetson; stop servers after tests; suggest VS Code Remote-SSH from a laptop. Voice-loop upgrades 1-20/23 built in iro/talk.py (24 Oct bench: first sound 10-24 s -> median 1.77 s; inner pauses <= 0.5 s); open: Wikipedia answers weak with the 2B (BM25 picks wrong paragraph), Whisper mishears "wayang kulit" in accented speech.

**24 Oct, audio breaks fixed:** the qwentts server sends ~0.5 s bursts and runs at RTF 1.1-1.3 while the LLM shares the GPU, so the speaker ran dry mid-sentence. `talk.Voice` now has a jitter buffer (`_hold_for`: 0.6 s + est·(RTF-1)/RTF, RTF measured per part), re-buffers 0.5 s after an underrun, and counts `underruns` (talk_bench prints "breaks"). The bench went from 2 breaks to 0, first sound median 1.38 s. Wiki: disambiguation `<li>` may start with text ("The <a…>"); "history of X" always looks it up, even when a card matches.

**24 Oct, long pauses (run 20261024-134526):** the said_ clips hide the waits between parts. Now `Voice.heard_as()` / reply_NNN.wav keep the real silences, and talk_bench replays recorded questions (`--owner`). Root cause: GPU sharing. LLM and voice together each run at half speed (voice RTF 0.84 -> 1.55, LLM 25 -> 12 tok/s). Fixes:
- `gpu_free` gate: new speech waits until the LLM is done; cached openers play meanwhile.
- One sentence per part; all openers (incl. FACT_OPENERS) precached; "Well," waits in case it becomes "Well, sir,".
- MAX_WORDS 35, and the LLM is stopped at the cap.
- Config YAML cached (0.27 -> 0.014 s per turn).
- A second reply after an action waits until the first reply's audio is made.
- Requests ("could you wait") are no Wikipedia lookup.

Remaining: one 1.3-2.9 s pause after the opener in LLM replies. This is GPU-bound: the total work exceeds the speech length. The Q4 talker crashes CUDA (illegal memory access).
