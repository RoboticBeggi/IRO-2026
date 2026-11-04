"""iro arm check: find which way each joint counts, by moving it by hand.

The position maths (and the RViz mirror) assume each servo counts the same way
as the URDF. If one counts backwards, everything computed from it is mirrored.
This wizard asks you to move one joint at a time in a stated direction and
compares that with the model. Results go into config/robot.yaml (joint_signs)
and config/pick_taught.yaml (which way the gripper opens).

Start from the calibration middle pose (the L shape), where every joint is near 0.
"""

import re
import sys

import numpy as np
import yaml

from iro import ROOT, load_config
from iro.hw.servos import open_bus
from iro.kinematics import JOINTS, Kinematics

TICKS = 4096
MIN_MOVE = 150   # raw ticks (~13 deg): smaller moves are too ambiguous

# joint, what to do, and a measure of the tip/gripper that the move should INCREASE
STEPS = [
    ("shoulder_pan", "Turn the whole arm to YOUR LEFT (stand behind it, facing the way it points), about 30 deg.",
     lambda k, q: k.fk(q)[1, 3]),                        # tip y (left) increases
    ("shoulder_lift", "Tilt the UPPER ARM forward (away from the base), about 30 deg. Keep the elbow angle fixed.",
     lambda k, q: k.fk(q)[0, 3] - k.fk(q)[2, 3]),        # tip goes forward and down
    ("elbow_flex", "Bend the ELBOW so the forearm points more DOWN, about 30 deg.",
     lambda k, q: -k.fk(q)[2, 3]),                       # tip goes down
    ("wrist_flex", "Tilt the GRIPPER DOWN at the wrist, about 30 deg.",
     lambda k, q: -k.fk(q)[2, 3]),
    ("wrist_roll", "Turn the GRIPPER CLOCKWISE as seen from behind the arm (looking along it), about 30 deg.",
     None),                                              # handled specially below
]


def roll_measure(k, q):
    """How far the gripper's 'up' side has turned toward the right (-y), seen from behind."""
    R = k.fk(q, "gripper_link")[:3, :3]
    R0 = k.fk(np.zeros(5), "gripper_link")[:3, :3]
    up_local = R0.T @ np.array([0, 0, 1.0])  # the gripper axis that points up in the middle pose
    return -(R @ up_local)[1]


def model_direction(k, measure, idx):
    q = np.zeros(5)
    dq = np.zeros(5)
    dq[idx] = 1e-3
    return np.sign(measure(k, q + dq) - measure(k, q))


def main():
    cfg = load_config()
    d = cfg["devices"]["arm"]
    k = Kinematics()
    signs, grip = {}, {}
    print(__doc__)
    with open_bus(d["port"], d["servo_ids"]) as bus:
        input("Hold the arm: it goes limp when you press Enter. ")
        bus.disable_torque()
        input("Put it in the MIDDLE POSE (L shape: upper arm up, forearm and gripper straight forward). Enter. ")

        def read():
            return bus.sync_read("Present_Position", normalize=False)

        for joint, text, measure in STEPS:
            idx = JOINTS.index(joint)
            measure = measure or roll_measure
            while True:
                r0 = read()[joint]
                input(f"\n{joint}: {text}\n  Enter when done. ")
                delta = read()[joint] - r0
                if abs(delta) >= MIN_MOVE:
                    break
                print(f"  only moved {delta * 360 / TICKS:.0f} deg; move it further (about 30 deg).")
            want = model_direction(k, measure, idx)
            signs[joint] = int(np.sign(delta) * want)
            print(f"  moved {delta * 360 / TICKS:+.0f} deg raw -> {'same as the model' if signs[joint] > 0 else 'REVERSED'}")
            input("  Put it back to the middle pose. Enter. ")

        cal = __import__("json").loads((ROOT / "arm" / "calibration" / "robots" / "so_follower" /
                                        "iro_follower.json").read_text())["gripper"]
        span = cal["range_max"] - cal["range_min"]
        input("\ngripper: CLOSE the jaws fully. Enter. ")
        closed = 100 * (read()["gripper"] - cal["range_min"]) / span
        input("gripper: OPEN the jaws fully. Enter. ")
        opened = 100 * (read()["gripper"] - cal["range_min"]) / span
        grip = {"gripper_closed_pct": round(float(closed)),
                "gripper_open_pct": round(float(closed + 0.8 * (opened - closed)))}
        print(f"  closed at {closed:.0f}%, open at {opened:.0f}% -> open target {grip['gripper_open_pct']}%")
        input("\nHold the arm: it stiffens where it is when you press Enter. ")
        now = read()
        bus.sync_write("Goal_Position", now, normalize=False)
        bus.enable_torque()
    signs["gripper"] = 1

    # Write joint_signs on its single line in robot.yaml (keeps the comments)
    path = ROOT / "config" / "robot.yaml"
    text = path.read_text()
    line = "joint_signs: {" + ", ".join(f"{j}: {s}" for j, s in signs.items()) + "}"
    text, n = re.subn(r"joint_signs: \{[^}]*\}", line, text)
    assert n == 1, "joint_signs line not found in robot.yaml"
    path.write_text(text)
    taught = ROOT / "config" / "pick_taught.yaml"
    data = yaml.safe_load(taught.read_text()) if taught.exists() else {}
    data.update(grip)
    taught.write_text("# Written by `iro pick teach` / `iro arm check`; overrides `pick:` in robot.yaml\n" + yaml.safe_dump(data))
    reversed_ = [j for j, s in signs.items() if s < 0]
    print(f"\nSaved. Reversed joints: {reversed_ or 'none'}. Check with `iro mirror`, then `iro pick teach`.")
    print("If the arm node is running, restart it (iro up / iro mirror) to use the new directions.")
    return 0


if __name__ == "__main__":
    from iro.hw.servos import BusyError
    try:
        sys.exit(main())
    except BusyError as e:
        sys.exit(f"STOPPED: {e}")
