#!/usr/bin/env bash
# Throttle test: run ARC's model flat out for ~1 min and count the Jetson's power-limit events.
# Compare power modes:  sudo nvpmodel -m 1 (25W)  /  sudo nvpmodel -m 2 (MAXN_SUPER), then run this.
# oc3 = VDD_IN instantaneous power (NVIDIA forum): each event throttles the clocks for a moment.
set -euo pipefail
S=/sys/class/hwmon/hwmon3
MODEL=${1:-$HOME/llm-models/Gemma-SEA-LION-v4.5-E2B-IT-Q4_K_M.gguf}
mode=$(nvpmodel -q 2>/dev/null | head -1 | sed 's/NV Power Mode: //')
a1=$(cat $S/oc1_event_cnt); a2=$(cat $S/oc2_event_cnt); a3=$(cat $S/oc3_event_cnt); t0=$(date +%s)
out=$("$HOME/llama.cpp/build/bin/llama-bench" -m "$MODEL" -ngl 99 -fa 1 -p 512 -n 128 -r 5 2>/dev/null | grep -E "pp512|tg128")
t1=$(date +%s)
b1=$(cat $S/oc1_event_cnt); b2=$(cat $S/oc2_event_cnt); b3=$(cat $S/oc3_event_cnt)
echo "mode: $mode   ($((t1 - t0)) s of full load)"
echo "$out" | awk -F'|' '{printf "  %-8s %s tokens/s\n", $8, $9}'
echo "  throttle events: under-voltage $((b1 - a1)), average power $((b2 - a2)), power spikes $((b3 - a3))"
