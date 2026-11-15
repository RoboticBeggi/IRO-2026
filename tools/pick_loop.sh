#!/usr/bin/env bash
# Pick-and-place loop: the arm moves an object from spot to spot by itself and re-picks it each time.
# For testing accuracy, speed and reach without a person moving the object:
#     tools/pick_loop.sh Tape 24,-6 27,8 22,4        (spots in cm from the arm base)
# Prints one summary line per pick (from the run's events.txt). Stops at the first failed pick.
set -uo pipefail
NAME=${1:?usage: tools/pick_loop.sh NAME X,Y [X,Y ...]}; shift
for dest in "$@"; do
    "$(dirname "$0")/iro" pick "$NAME" --to="$dest" 2>&1 | grep -v "median filter\|Live feed\|Recorded:" | sed 's/^/    /'
    run=$(ls -td ~/iro-runs/*pick-"$NAME" | head -1)
    ev="$run/events.txt"
    ok=$(grep -c "picked on try" "$ev")
    echo "-> to $dest: $([ "$ok" = 1 ] && echo OK || echo FAILED) | $(grep -o 'ready in [0-9.]* s' "$ev") | $(grep -o 'aim: [0-9]* px' "$ev" | head -1) | centring steps $(($(wc -l < "$run/centre.csv" 2>/dev/null || echo 1) - 1)) | $(grep -o 'down in [0-9.]* s' "$ev" | tail -1) | $(grep -o 'whole pick: [0-9.]* s' "$ev") | $(grep -o 'placed at.*' "$ev") | $(grep -o 'they now fit to [0-9]* mm' "$ev")"
    [ "$ok" = 1 ] || { echo "stopped: see $run"; exit 1; }
done
