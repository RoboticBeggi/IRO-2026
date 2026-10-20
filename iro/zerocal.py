"""iro arm zero: correct the small zero errors of each joint by touching the table.

LeRobot calibration takes whatever pose you held as "middle", but the URDF's zero
pose is precise (e.g. its upper arm leans 14 deg forward). A few degrees of
difference move the tip by centimetres. Every touch of the flat table must give
the same height, so ~8 touches in different poses pin down the zero error of
shoulder_lift, elbow_flex and wrist_flex (and the table height).

Saves joint_offsets in config/robot.yaml and table_z in config/pick_taught.yaml.
"""

import json
import re
import sys
import time

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


MAX_OFF = np.radians(10)   # fine-tune only: `iro arm align` fixes anything bigger by eye


def solve(kin, qs, table_guess):
    """Gauss-Newton with box limits: corrections (rad, within +-MAX_OFF) for SOLVE joints and
    table_z, so every touch lies on the table. Returns (offsets, table_z, rms, per-touch fit)."""
    idx = [JOINTS.index(j) for j in SOLVE]
    p = np.zeros(len(idx) + 1)
    p[-1] = table_guess

    def residuals(p, with_prior=True):
        r = []
        for q in qs:
            qq = np.array(q, float)
            qq[idx] += p[:-1]
            r.append(fingertip_z(kin, qq) - p[-1])
        if with_prior:
            # Touches only measure height, so elbow and wrist errors can trade off against each
            # other; a gentle pull toward zero (1 mm per 5 deg) picks the smallest correction.
            r += list(p[1:-1] / np.radians(5) * 0.001)
        return np.array(r)

    lam, cost = 1e-3, np.sum(residuals(p) ** 2)
    for _ in range(200):           # Levenberg-Marquardt: damped steps, kept only if they help
        r = residuals(p)
        J = np.zeros((len(r), len(p)))
        for i in range(len(p)):
            dp = np.zeros(len(p))
            dp[i] = 1e-6
            J[:, i] = (residuals(p + dp) - r) / 1e-6
        A, g = J.T @ J, J.T @ r
        step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
        new = p + step
        new[:-1] = np.clip(new[:-1], -MAX_OFF, MAX_OFF)
        new_cost = np.sum(residuals(new) ** 2)
        if new_cost < cost:
            done = cost - new_cost < 1e-14
            p, cost, lam = new, new_cost, lam / 3
            if done:
                break
        else:
            lam *= 4
            if lam > 1e8:
                break
    fit = residuals(p, with_prior=False)
    return dict(zip(SOLVE, p[:-1])), float(p[-1]), float(np.sqrt(np.mean(fit**2))), fit


def robust_solve(kin, qs, table_guess, bad_mm=4.0):
    """Leave-one-out: judge each touch against a fit made without it; drop the worst while it is
    clearly off (side of a jaw touched, arm slipped), at most 25% of the touches."""
    keep = list(range(len(qs)))
    while len(keep) > max(MIN_TOUCHES, int(0.75 * len(qs))):
        loo = []
        for i in keep:
            others = [qs[j] for j in keep if j != i]
            offs, table, _, _ = solve(kin, others, table_guess)
            idx = [JOINTS.index(j) for j in SOLVE]
            qq = np.array(qs[i], float)
            qq[idx] += [offs[j] for j in SOLVE]
            loo.append(abs(fingertip_z(kin, qq) - table))
        worst = int(np.argmax(loo))
        if loo[worst] * 1000 < bad_mm:
            break
        keep.pop(worst)
    offs, table, rms, _ = solve(kin, [qs[i] for i in keep], table_guess)
    return offs, table, rms, sorted(set(range(len(qs))) - set(keep))


def main():
    cfg = load_config()
    d = cfg["devices"]["arm"]
    signs = d.get("joint_signs", {})
    # Start from the saved offsets (from `iro arm align`); this only fine-tunes lift/elbow/wrist
    base = {j: np.radians(v) for j, v in d.get("joint_offsets_deg", {}).items()}
    log = ROOT / "arm" / "calibration" / f"touches_{time.strftime('%Y%m%d-%H%M%S')}.json"
    kin = Kinematics()
    if len(sys.argv) > 1:   # re-solve saved touches: iro arm zero arm/calibration/touches_*.json
        saved = json.loads(open(sys.argv[1]).read())
        raws = saved["raw"]
        qs = [[signs.get(j, 1) * (r[j] - TICKS / 2) * 2 * np.pi / TICKS + base.get(j, 0.0) for j in JOINTS] for r in raws]
        print(f"Re-solving {len(qs)} saved touches from {sys.argv[1]}")
        return finish(cfg, kin, qs, base, log=None)
    print(__doc__)
    qs, raws = [], []
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
            qs.append([signs.get(j, 1) * (raw[j] - TICKS / 2) * 2 * np.pi / TICKS + base.get(j, 0.0) for j in JOINTS])
            raws.append(raw)  # saved at once: no touch is ever lost
            log.write_text(json.dumps({"base_offsets_deg": d.get("joint_offsets_deg", {}), "signs": signs, "raw": raws}))
            tip = kin.fk(qs[-1])[:3, 3]
            print(f"     recorded (tip {tip[0] * 100:.0f} cm out, {tip[1] * 100:+.0f} cm side)")
        input("Hold the arm: it stiffens where it is when you press Enter. ")
        now = bus.sync_read("Present_Position", normalize=False)
        bus.sync_write("Goal_Position", now, normalize=False)
        bus.enable_torque()
    return finish(cfg, kin, qs, base, log)


def finish(cfg, kin, qs, base, log):
    before = np.array([fingertip_z(kin, q) for q in qs])
    offsets, table_z, rms, dropped = robust_solve(kin, qs, -cfg["pick"]["mount_height"])
    if dropped:
        print(f"Ignored touches that did not fit: {[i + 1 for i in dropped]}")
    print(f"\nBefore: touches spread {np.ptp(before) * 100:.1f} cm in height (should be 0)")
    print(f"After:  spread {rms * 1000:.1f} mm RMS, table at {table_z * 100:.1f} cm "
          f"(mount height says {-cfg['pick']['mount_height'] * 100:.1f})")
    for j, o in offsets.items():
        print(f"  {j:14} correction {np.degrees(o):+.1f} deg (total offset {np.degrees(base.get(j, 0.0) + o):+.1f})")
    if rms > 0.005:
        print(f"Fit is poor (> 5 mm). Offsets not saved; your touches are kept in {log.relative_to(ROOT) if log else 'that file'}.\n"
              "Run `iro arm align` first if the RViz arm does not match the real one, then redo.")
        return 1

    path = ROOT / "config" / "robot.yaml"
    text = path.read_text()
    full = {j: round(float(np.degrees(base.get(j, 0.0) + offsets.get(j, 0.0))), 2) for j in JOINTS + ["gripper"]}
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
