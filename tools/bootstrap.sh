#!/usr/bin/env bash
# Rebuild the IRO 2026 software setup on a freshly flashed Jetson (JetPack 6).
# Safe to run again on a working machine; every step skips what is already done.
#
#   git clone https://github.com/roboticbeggi/iro-2026.git ~/iro-2026
#   ~/iro-2026/tools/bootstrap.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/.venvs/iro"
JETSON_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126/+simple"

echo "== 1/4 System packages and permissions (asks for your sudo password)"
sudo apt-get update
sudo apt-get install -y git python3-pip python3.10-venv ffmpeg rsync
sudo usermod -aG dialout "$USER"          # serial ports: arm, lidar
sudo cp "$ROOT/tools/99-iro-usb.rules" /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

echo "== 2/4 Python environment ($VENV)"
pip3 install --user -q uv
export PATH="$HOME/.local/bin:$PATH"
[ -d "$VENV" ] || uv venv --python /usr/bin/python3 "$VENV"
export VIRTUAL_ENV="$VENV"
# CUDA torch for Jetson first, from NVIDIA's Jetson AI Lab index, so LeRobot
# does not pull a CPU-only torch from PyPI.
uv pip install --index-url "$JETSON_INDEX" -r "$ROOT/env/requirements-torch.txt"
uv pip install -r "$ROOT/env/requirements.txt"

echo "== 3/4 Restore memory notes for Claude Code"
MEM="$HOME/.claude/projects/$(echo "$ROOT" | sed 's#[/_.]#-#g')/memory"
if [ -d "$ROOT/backup/claude-memory" ]; then
    mkdir -p "$MEM" && cp -n "$ROOT/backup/claude-memory/"* "$MEM/" || true
fi

echo "== 4/4 Check"
"$VENV/bin/python" -c "import torch, lerobot; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"

cat <<EOF

Done. Log out and back in once (for the dialout group), then:
  source $ROOT/env.sh
  python $ROOT/arm/scan_motors.py
Calibration is already in arm/calibration/, no need to recalibrate unless the
arm was rebuilt. To turn the automatic backup back on: tools/install-autobackup.sh
EOF
