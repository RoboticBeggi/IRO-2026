#!/usr/bin/env bash
# Play every voice sample made by tools/voice_bench.py, one voice at a time, so you can pick the most
# human one. Plays through the reSpeaker's 3.5 mm output (where the robot's speaker goes, so the
# array's echo cancelling hears it); HDMI instead: DEV=hdmi tools/listen_voices.sh
DEV=${DEV:-plughw:CARD=Array,DEV=0}
[ "$DEV" = hdmi ] && DEV=default
for d in ~/voice-samples/*/; do
    echo "=== $(basename "$d")"
    for f in "$d"line*.wav; do aplay -q -D "$DEV" "$f"; sleep 0.4; done
    read -r -p "    (Enter for the next voice) " _
done
