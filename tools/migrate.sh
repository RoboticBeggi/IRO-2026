#!/usr/bin/env bash
# Move the whole robot setup from the old Jetson to a new one over the network (no USB drive).
# Run ON THE NEW JETSON, with both on the same network:
#
#     tools/migrate.sh orin_nano@10.207.38.123          (the OLD Jetson's user@address)
#
# Copies the repo, models, voices, offline Wikipedia, recordings, built programs (llama.cpp,
# qwentts.cpp, whisper.cpp: same GPU type, sm_87, so they run as they are) and the Python
# environment. rsync over SSH: safe to stop and run again, it continues where it stopped.
# The system part (apt, ROS 2, device names, crontab) is tools/bootstrap.sh afterwards.
set -euo pipefail

SRC="${1:?usage: tools/migrate.sh user@old-jetson-address}"
RS=(rsync -aH --partial --info=progress2 --human-readable --exclude='__pycache__' -e "ssh -o ServerAliveInterval=15")

echo "== Checking the old Jetson ($SRC)"
ssh -o ConnectTimeout=8 "$SRC" true || {
    echo "Cannot reach $SRC. Same network? On school Wi-Fi, devices are often kept apart: use a phone"
    echo "hotspot for both Jetsons, or one network cable between them."; exit 1; }

old_home=$(ssh "$SRC" 'echo $HOME')
old_l4t=$(ssh "$SRC" 'head -1 /etc/nv_tegra_release' | sed 's/, GCID.*//')
new_l4t=$(head -1 /etc/nv_tegra_release | sed 's/, GCID.*//')
echo "   old: $old_l4t   home $old_home"
echo "   new: $new_l4t   home $HOME"
if [ "$old_l4t" != "$new_l4t" ]; then
    echo "!! Different JetPack/L4T versions. The built programs and the Python environment may not run;"
    echo "   they will be copied anyway, then rebuilt if 'iro doctor' complains. Ctrl+C now to stop."; sleep 8
fi
if [ "$old_home" != "$HOME" ]; then
    echo "!! Different home folder ($old_home vs $HOME). The Python environment and crontab contain the old"
    echo "   path: bootstrap.sh will rebuild the environment. Easiest: same user name (orin_nano) on both."; sleep 8
fi
need=$(ssh "$SRC" "cd ~ && du -sck iro-2026 iro-runs voice-models voice-samples weights src/qwentts.cpp whisper.cpp llama.cpp llm-models .venvs/iro wiki 2>/dev/null | tail -1 | cut -f1")
free=$(df -k --output=avail "$HOME" | tail -1)
echo "   to copy: $((need / 1024 / 1024)) GB, free here: $((free / 1024 / 1024)) GB"
[ "$free" -gt "$((need + 10 * 1024 * 1024))" ] || { echo "Not enough disk space here."; exit 1; }

# small and important first, the big Wikipedia files last
ITEMS=(iro-2026 .claude/projects voice-samples voice-models weights iro-runs
       src/qwentts.cpp whisper.cpp llama.cpp llm-models .venvs/iro
       .arduino15 .local/opt .config/pulse/default.pa wiki)
for it in "${ITEMS[@]}"; do
    if ! ssh "$SRC" "test -e ~/$it"; then echo "== (skip $it: not on the old Jetson)"; continue; fi
    echo "== $it"
    mkdir -p "$HOME/$(dirname "$it")"
    "${RS[@]}" "$SRC:$it" "$HOME/$(dirname "$it")/"
done

echo "== Checking the copies"
cd "$HOME/iro-2026" && git status --short | head -3 && git log --oneline -1
for f in wiki/*.zim; do
    a=$(stat -c %s "$HOME/$f"); b=$(ssh "$SRC" "stat -c %s ~/$f")
    [ "$a" = "$b" ] && echo "   ok  $f" || echo "   !!  $f size differs: run this script again"
done

cat <<EOF

Copied. Next, on THIS Jetson:
  1. tools/bootstrap.sh            (system: apt + ROS 2 Humble, device names, serial access, ROS workspace,
                                    auto-backup; asks for your sudo password; needs internet)
  2. sudo apt install ros-humble-slam-toolbox ros-humble-navigation2 ros-humble-nav2-bringup \\
         ros-humble-nav2-collision-monitor ros-humble-robot-localization
  3. log out and in (serial-port access), then:  iro doctor
  4. GitHub auto-backup from this Jetson: tools/setup-push-key.sh (adds a key for this machine)
  5. Claude Code: install and log in again; its notes about this project were copied (~/.claude/projects)
EOF
