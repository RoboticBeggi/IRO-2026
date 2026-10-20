"""iro arm calibrate: a fresh LeRobot calibration of the SO-101, checked afterwards.

The old file is moved aside first, so LeRobot cannot offer to reuse it (pressing
Enter at that prompt silently keeps a bad calibration).
"""

import json
import os
import subprocess
import sys
import time

from iro import ROOT, load_config

CAL_DIR = ROOT / "arm" / "calibration" / "robots" / "so_follower"
OLD_DIR = ROOT / "arm" / "calibration" / "old"

MIDDLE_POSE = """
STEP 1 - MIDDLE POSE. When LeRobot asks, put EVERY joint in the middle of its travel:
  shoulder_pan   arm facing straight forward
  shoulder_lift  upper arm straight up (vertical)
  elbow_flex     forearm horizontal, pointing forward: a right angle (L shape)
                 NOT folded down onto the base, NOT straight out
  wrist_flex     wrist straight, in line with the forearm (gripper points straight forward)
  wrist_roll     gripper jaws level (not turned)
  gripper        half open
  The elbow matters most: last time it was set near one end, so its reading wrapped.

STEP 2 - RANGES. Move each joint SLOWLY from one end to the other (wrist_roll is skipped).
  Watch LeRobot's MIN / MAX table while you do it:
  a joint showing MIN 0 and MAX 4095 has wrapped -> press Ctrl+C and start again
  with a better middle pose for that joint.
"""


def main():
    cfg = load_config()
    robot_id = os.environ.get("SO101_ID", "iro_follower")
    cal = CAL_DIR / f"{robot_id}.json"
    print(MIDDLE_POSE)
    input("Arm powered, ready to hold it? Press Enter to start. ")
    if cal.exists():
        OLD_DIR.mkdir(parents=True, exist_ok=True)
        old = OLD_DIR / f"{robot_id}_{time.strftime('%Y%m%d-%H%M%S')}.json"
        cal.rename(old)
        print(f"(previous calibration kept in {old.relative_to(ROOT)})")

    r = subprocess.run(["lerobot-calibrate", "--robot.type=so101_follower",
                        f"--robot.port={cfg['devices']['arm']['port']}", f"--robot.id={robot_id}"])
    if r.returncode != 0 or not cal.exists():
        print("\nCalibration did not finish. Run `iro arm calibrate` again.")
        return 1

    data = json.loads(cal.read_text())
    continuous = set(cfg["devices"]["arm"].get("continuous_joints", []))
    print("\nResult:")
    bad = []
    for j, c in data.items():
        span = c["range_max"] - c["range_min"]
        wrapped = j not in continuous and c["range_min"] <= 0 and c["range_max"] >= 4095
        tiny = j not in continuous and span < 400
        note = "WRAPPED (middle pose was off for this joint)" if wrapped else \
            "very small range: was it moved end to end?" if tiny else "ok"
        if wrapped or tiny:
            bad.append(j)
        print(f"  {j:14} {c['range_min']:5} - {c['range_max']:5}  ({span * 360 / 4096:5.0f} deg)  {note}")
    if bad:
        print(f"\nNOT GOOD for {bad}. Run `iro arm calibrate` again (the arm will refuse to move until fixed).")
        return 1
    print("\nCalibration good. Backing it up...")
    subprocess.run([str(ROOT / "tools" / "backup.sh")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
