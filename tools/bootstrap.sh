#!/usr/bin/env bash
# Install or repair the whole IRO 2026 software setup (`iro setup`).
# Works on a freshly flashed Jetson (JetPack 6) and is safe to run again:
# every step skips what is already done. Asks for the sudo password once.
#
# Fresh Jetson, with internet:
#   git clone https://github.com/roboticbeggi/iro-2026.git ~/iro-2026
#   ~/iro-2026/tools/bootstrap.sh
# Fresh Jetson, no internet (competition): see RECOVERY.md, "offline".
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/.venvs/iro"
JETSON_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126/+simple"
ROS_PKGS=(
    ros-humble-ros-base ros-dev-tools
    ros-humble-rviz2 ros-humble-robot-state-publisher ros-humble-joint-state-publisher-gui
    ros-humble-xacro ros-humble-tf2-ros
    ros-humble-rplidar-ros ros-humble-v4l2-camera ros-humble-depthai-ros-driver
    ros-humble-diagnostic-aggregator ros-humble-rqt-robot-monitor
    ros-humble-teleop-twist-keyboard ros-humble-image-transport-plugins
)

echo "== 1/6 System packages, permissions, device names (sudo)"
sudo apt-get update
sudo apt-get install -y git python3-pip python3.10-venv ffmpeg rsync curl software-properties-common
sudo usermod -aG dialout "$USER"          # serial ports: arm, lidar, PRIZM
sudo cp "$ROOT/tools/99-iro-usb.rules" /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
# live camera page (EFAD) reachable from a laptop on the same network
if command -v ufw >/dev/null && sudo ufw status | grep -q "Status: active"; then sudo ufw allow 8080/tcp; fi

echo "== 2/6 ROS 2 Humble"
if [ ! -f /etc/apt/sources.list.d/ros2.list ]; then
    sudo add-apt-repository -y universe
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
        | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
    sudo apt-get update
fi
sudo apt-get install -y "${ROS_PKGS[@]}"
[ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || sudo rosdep init || true
rosdep update >/dev/null || true

echo "== 3/6 Python environment ($VENV)"
pip3 install --user -q uv
export PATH="$HOME/.local/bin:$PATH"
[ -d "$VENV" ] || uv venv --python /usr/bin/python3 "$VENV"
export VIRTUAL_ENV="$VENV"
# CUDA torch for Jetson first, from NVIDIA's Jetson AI Lab index, so LeRobot
# does not pull a CPU-only torch from PyPI.
uv pip install --index-url "$JETSON_INDEX" -r "$ROOT/env/requirements-torch.txt"
uv pip install -r "$ROOT/env/requirements.txt" --override "$ROOT/env/overrides.txt"
uv pip install --no-deps -r "$ROOT/env/requirements-nodeps.txt"
unset VIRTUAL_ENV

echo "== 4/6 Arduino tools for the PRIZM (arduino-cli + avrdude, no sudo)"
if ! command -v arduino-cli >/dev/null; then
    mkdir -p "$HOME/.local/opt/arduino-cli"
    curl -fsSL https://downloads.arduino.cc/arduino-cli/arduino-cli_latest_Linux_ARM64.tar.gz \
        | tar xz -C "$HOME/.local/opt/arduino-cli" arduino-cli
    ln -sf "$HOME/.local/opt/arduino-cli/arduino-cli" "$HOME/.local/bin/arduino-cli"
fi
arduino-cli core list | grep -q "arduino:avr" || { arduino-cli core update-index && arduino-cli core install arduino:avr; }

echo "== 5/6 The iro command, and ROS packages"
ln -sf "$ROOT/tools/iro" "$HOME/.local/bin/iro"
# Every new terminal gets the project environment + tab completion for iro
if ! grep -q "iro-2026/env.sh" "$HOME/.bashrc"; then
    printf '\n# IRO 2026 robot (added by tools/bootstrap.sh)\n[ -f "%s/env.sh" ] && . "%s/env.sh" >/dev/null\n[ -f /opt/ros/humble/setup.bash ] && . /opt/ros/humble/setup.bash\n[ -f "%s/ros2_ws/install/setup.bash" ] && . "%s/ros2_ws/install/setup.bash"\n[ -f "%s/tools/iro-completion.bash" ] && . "%s/tools/iro-completion.bash"\n' \
        "$ROOT" "$ROOT" "$ROOT" "$ROOT" "$ROOT" "$ROOT" >> "$HOME/.bashrc"
fi
set +u; . /opt/ros/humble/setup.bash; . "$ROOT/env.sh"; set -u
(cd "$ROOT/ros2_ws" && colcon build --symlink-install)

echo "== 6/6 Memory notes for Claude Code, backups"
MEM="$HOME/.claude/projects/$(echo "$ROOT" | sed 's#[/_.]#-#g')/memory"
if [ -d "$ROOT/backup/claude-memory" ]; then
    mkdir -p "$MEM" && cp -n "$ROOT/backup/claude-memory/"* "$MEM/" || true
fi
crontab -l 2>/dev/null | grep -q tools/backup.sh || "$ROOT/tools/install-autobackup.sh"
"$ROOT/tools/iro" feed install           # live camera page always on (port 8080)

cat <<EOF

Setup done. Log out and back in once (new permissions), open a terminal, then:
  iro doctor
If GitHub backup is not set up on this Jetson yet: $ROOT/tools/setup-push-key.sh
EOF
