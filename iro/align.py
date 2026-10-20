"""iro arm align: make the RViz arm match the real arm, by eye, live.

Starts the RViz mirror, then you type corrections and watch the model move:
    roll 90        turn the model's wrist roll by +90 deg
    lift -10       shoulder_lift -10 deg     (also: pan, elbow, wrist, grip)
    show           current offsets
    save           write them to config/robot.yaml
    quit
Do the big ones first (roll), then shoulder, elbow, wrist. Put the real arm in the
middle pose (L shape) and compare it from the side and from above.
"""

import re
import subprocess
import sys
import time

import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters

from iro import ROOT, load_config

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ALIASES = {"pan": "shoulder_pan", "lift": "shoulder_lift", "shoulder": "shoulder_lift", "elbow": "elbow_flex",
           "wrist": "wrist_flex", "flex": "wrist_flex", "roll": "wrist_roll", "grip": "gripper", "gripper": "gripper"}


def save(offsets):
    path = ROOT / "config" / "robot.yaml"
    line = "joint_offsets_deg: {" + ", ".join(f"{j}: {round(offsets[j], 2)}" for j in NAMES) + "}"
    text, n = re.subn(r"joint_offsets_deg: \{[^}]*\}", line, path.read_text())
    assert n == 1
    path.write_text(text)


def main():
    cfg_off = load_config()["devices"]["arm"].get("joint_offsets_deg", {})
    offsets = {j: float(cfg_off.get(j, 0.0)) for j in NAMES}
    print(__doc__)
    mirror = subprocess.Popen([str(ROOT / "tools" / "iro"), "mirror"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rclpy.init()
    node = rclpy.create_node("arm_align")
    cli = node.create_client(SetParameters, "/arm/set_parameters")
    print("Starting the mirror and RViz...")
    if not cli.wait_for_service(timeout_sec=30):
        print("The arm node did not start (is the arm powered? is another program using it?).")
        mirror.terminate()
        return 1

    def push():
        req = SetParameters.Request()
        req.parameters = [Parameter(name="joint_offsets_deg", value=ParameterValue(
            type=ParameterType.PARAMETER_DOUBLE_ARRAY, double_array_value=[offsets[j] for j in NAMES]))]
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=3)
        ok = fut.result() and fut.result().results[0].successful
        if not ok:
            print("  (the arm node did not accept it)")

    push()
    try:
        while True:
            cmd = input("align> ").strip().lower().split()
            if not cmd:
                continue
            if cmd[0] in ("quit", "q", "exit"):
                break
            if cmd[0] == "show":
                print("  " + ", ".join(f"{j} {v:+.1f}" for j, v in offsets.items()))
                continue
            if cmd[0] == "save":
                save(offsets)
                print("  saved to config/robot.yaml (used by iro pick, iro up, iro mirror)")
                continue
            joint = ALIASES.get(cmd[0], cmd[0] if cmd[0] in NAMES else None)
            try:
                delta = float(cmd[1])
            except (IndexError, ValueError):
                joint = None
            if not joint:
                print("  e.g.  roll 90  |  lift -10  |  show  |  save  |  quit")
                continue
            offsets[joint] += delta
            push()
            print(f"  {joint} offset now {offsets[joint]:+.1f} deg")
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        if input("Save before leaving? [y/N] ").strip().lower() == "y":
            save(offsets)
            print("  saved")
        mirror.terminate()
        time.sleep(1)
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
