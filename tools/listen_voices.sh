#!/usr/bin/env bash
# Play every voice sample made by tools/voice_bench.py, one voice at a time, so you can pick the most
# human one. Plays on the default sound output (the HDMI screen's speakers, set 23 Oct with
# `pactl set-default-sink alsa_output.platform-3510000.hda.hdmi-stereo`).
# Through the reSpeaker's 3.5 mm jack instead: DEV=respeaker tools/listen_voices.sh
# A folder of voices as argument (default ~/voice-samples): tools/listen_voices.sh ~/voice-samples/gallery
DIR=${1:-$HOME/voice-samples}
play() {
    if [ "${DEV:-}" = respeaker ]; then aplay -q -D plughw:CARD=Array,DEV=0 "$1"; else paplay "$1"; fi
}
for d in "$DIR"/*/; do
    [ "$(basename "$d")" = gallery ] && continue
    echo "=== $(basename "$d")"
    for f in "$d"line*.wav; do play "$f"; sleep 0.4; done
    read -r -p "    (Enter for the next voice) " _
done
