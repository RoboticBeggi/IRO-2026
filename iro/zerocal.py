"""iro arm zero: correct the small zero errors of each joint by touching the table.

LeRobot calibration takes whatever pose you held as "middle", but the URDF's zero
pose is precise (e.g. its upper arm leans 14 deg forward). A few degrees of
difference move the tip by centimetres. Every touch of the flat table must give
the same height, so ~8 touches in different poses pin down the zero error of
shoulder_lift, elbow_flex and wrist_flex (and the table height).

Saves joint_offsets in config/robot.yaml and table_z in config/pick_taught.yaml.
"""

import re
import sys

import numpy as np
import yaml

from iro import ROOT, load_config
from iro.hw.servos import open_bus
from iro.kinematics import JOINTS, Kinematics

TICKS = 4096
FINGER = 0.007          # jaw tips stick out 7 mm past the URDF tip frame (from the meshes)
SOLVE = ["shoulder_lift", "elbow_flex", "wrist_flex"]
MIN_TOUCHES = 6


def fingertip_z(kin, q):
    T = kin.fk(q)
    return T[2, 3] + FINGER * T[2, 2]


def solve(kin, qs, table_guess, prior_table_sigma=0.03, prior_off_sigma=np.radians(20)):
    """Gauss-Newton: offsets (rad) for SOLVE joints + table_z, so every touch lies on the table."""
    idx = [JOINTS.index(j) for j in SOLVE]
    p = np.zeros(len(idx) + 1)
    p[-1] = table_guess

    def residuals(p):
        r = []
        for q in qs:
            qq = np.array(q, float)
            qq[idx] += p[:-1]
            r.append(fingertip_z(kin, qq) - p[-1])
        # weak priors keep the solution sensible when the touches do not vary enough
        r += list(p[:-1] / prior_off_sigma * 0.002)
        r.append((p[-1] - table_guess) / prior_table_sigma * 0.002)
        return np.array(r)

    for _ in range(30):
        r = residuals(p)
        J = np.zeros((len(r), len(p)))
        for i in range(len(p)):
            dp = np.zeros(len(p))
            dp[i] = 1e-6
            J[:, i] = (residuals(p + dp) - r) / 1e-6
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        p += step
        if np.linalg.norm(step) < 1e-7:
            break
    fit = residuals(p)[:len(qs)]
    return dict(zip(SOLVE, p[:-1])), float(p[-1]), float(np.sqrt(np.mean(fit**2)))


def main():
    cfg = load_config()
    d = cfg["devices"]["arm"]
    signs = d.get("joint_signs", {})
    # start from the raw servo angles: any previous offsets are re-measured, not stacked
    kin = Kinematics()
    print(__doc__)
    qs = []
    with open_bus(d["port"], d["servo_ids"]) as bus:
        input("Hold the arm: it goes limp when you press Enter. ")
        bus.disable_torque()
        print("Touch the table with the closed jaw TIPS, gripper roughly pointing down.\n"
              f"Vary each touch: near / far, left / right, wrist more or less bent. At least {MIN_TOUCHES}.")
        while True:
            a = input(f"  touch {len(qs) + 1}: tips on the table, Enter to record ('d' = done): ").strip().lower()
            if a == "d":
                if len(qs) >= MIN_TOUCHES:
                    break
                print(f"  need at least {MIN_TOUCHES}")
                continue
            raw = bus.sync_read("Present_Position", normalize=False)
            qs.append([signs.get(j, 1) * (raw[j] - TICKS / 2) * 2 * np.pi / TICKS for j in JOINTS])
            tip = kin.fk(qs[-1])[:3, 3]
            print(f"     recorded (tip {tip[0] * 100:.0f} cm out, {tip[1] * 100:+.0f} cm side)")
        input("Hold the arm: it stiffens where it is when you press Enter. ")
        now = bus.sync_read("Present_Position", normalize=False)
        bus.sync_write("Goal_Position", now, normalize=False)
        bus.enable_torque()

    before = np.array([fingertip_z(kin, q) for q in qs])
    offsets, table_z, rms = solve(kin, qs, -cfg["pick"]["mount_height"])
    print(f"\nBefore: touches spread {np.ptp(before) * 100:.1f} cm in height (should be 0)")
    print(f"After:  spread {rms * 1000:.1f} mm RMS, table at {table_z * 100:.1f} cm "
          f"(mount height says {-cfg['pick']['mount_height'] * 100:.1f})")
    for j, o in offsets.items():
        print(f"  {j:14} zero error {np.degrees(o):+.1f} deg")
    if rms > 0.005:
        print("Fit is poor (> 5 mm). Redo with more varied touches; make sure only the jaw tips touch. Nothing saved.")
        return 1

    path = ROOT / "config" / "robot.yaml"
    text = path.read_text()
    full = {j: round(float(np.degrees(offsets.get(j, 0.0))), 2) for j in JOINTS}
    line = "joint_offsets_deg: {" + ", ".join(f"{j}: {v}" for j, v in full.items()) + "}"
    text, n = re.subn(r"joint_offsets_deg: \{[^}]*\}", line, text)
    assert n == 1, "joint_offsets_deg line not found in robot.yaml"
    path.write_text(text)
    taught = ROOT / "config" / "pick_taught.yaml"
    data = yaml.safe_load(taught.read_text()) if taught.exists() else {}
    data["table_z"] = round(table_z, 4)
    taught.write_text("# Written by `iro pick teach` / `iro arm check` / `iro arm zero`; overrides `pick:` in robot.yaml\n"
                      + yaml.safe_dump(data))
    print("Saved. Restart `iro mirror` / `iro up` to use it. Next: iro pick teach")
    return 0


if __name__ == "__main__":
    sys.exit(main())
