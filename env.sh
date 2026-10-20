# Source this in every terminal:  source ~/iro-2026/env.sh
# Activates the Python environment and keeps LeRobot calibration inside the
# repo (arm/calibration/), so it is backed up with everything else.

IRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export IRO_ROOT
export PATH="$HOME/.local/bin:$PATH"
export HF_LEROBOT_CALIBRATION="$IRO_ROOT/arm/calibration"

# Stable port names from tools/99-iro-usb.rules; fall back to the raw device.
if [ -e /dev/so101 ]; then export SO101_PORT=/dev/so101; else export SO101_PORT=/dev/ttyACM0; fi
export SO101_ID=iro_follower

if [ -f "$HOME/.venvs/iro/bin/activate" ]; then
    . "$HOME/.venvs/iro/bin/activate"
else
    echo "env.sh: ~/.venvs/iro missing, run tools/bootstrap.sh" >&2
fi
