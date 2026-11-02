#!/usr/bin/env bash
# How much memory each part of ARC really takes on the Jetson (CPU and GPU share 7.6 GB):
# starts them one by one and measures MemAvailable before/after. Stops them at the end.
set -u
avail() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
up() { for _ in $(seq 120); do curl -s -o /dev/null -w '%{http_code}' "$1" 2>/dev/null | grep -q 200 && return 0; sleep 0.5; done; return 1; }
stop() { pkill -f "[l]lama.cpp/build/bin/llama-server"; pkill -f "[w]hisper-server"; pkill -f "[t]ts-server"; sleep 3; }
LLM=${LLM:-$HOME/llm-models/Gemma-SEA-LION-v4.5-E2B-IT-Q4_K_M.gguf}
CTX=${CTX:-12288}
TTS=${TTS:-qwen-talker-0.6b-customvoice-Q8_0.gguf}
WH=${WH:-$HOME/voice-models/ggml-large-v3-turbo-q8_0.bin}
stop
b=$(avail); echo "baseline (desktop etc. running): $b MB available"
~/llama.cpp/build/bin/llama-server -m "$LLM" --port 8090 -c "$CTX" -ngl 99 -fa on -np 2 --jinja --no-webui >/tmp/mb-llm.log 2>&1 &
up localhost:8090/health; sleep 2; m=$(avail); echo "ARC language model ($(basename "$LLM"), ctx $CTX): $((b - m)) MB"; b=$m
~/llama.cpp/build/bin/llama-server -m ~/llm-models/embeddinggemma-300M-Q8_0.gguf --port 8091 --embedding -ngl 99 -c 2048 -b ${EB:-2048} -ub ${EB:-2048} --no-webui >/tmp/mb-emb.log 2>&1 &
up localhost:8091/health; sleep 2; m=$(avail); echo "Wikipedia embedder (batch ${EB:-2048}): $((b - m)) MB"; b=$m
~/whisper.cpp/build/bin/whisper-server -m "$WH" --port 8092 -l auto -nt >/tmp/mb-wh.log 2>&1 &
up localhost:8092/health; sleep 2; m=$(avail); echo "Whisper ($(basename "$WH")): $((b - m)) MB"; b=$m
(cd ~/src/qwentts.cpp && ./build/tts-server --model "models/$TTS" --codec models/qwen-tokenizer-12hz-Q8_0.gguf --port 8093 >/tmp/mb-tts.log 2>&1 &)
up localhost:8093/health; sleep 2; m=$(avail); echo "Qwen voice ($TTS): $((b - m)) MB"; b=$m
echo "left for vision, arm, ROS: $b MB available"
[ "${KEEP:-0}" = 1 ] || stop
