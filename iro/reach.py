"""iro arm reach: measure how far (and how close) the arm can really reach, by hand.

    iro arm reach              torque off; you move the arm around its limits; it records
    iro arm reach --from RUN   re-make config/reach.yaml from a recorded session (no arm)

Why by hand: every move so far kept the gripper straight down, which uses only part of what the
six servos can do. Your hands find the real limits (cables, the base, the floor) faster and more
safely than the arm probing on its own.

While you move it, the screen shows the tip's distance from the base, its height above the floor
and how far the gripper leans from vertical (+ = leaning outward). Every pose is recorded 20x a
second. Things to try:
  - straight out along the floor, as far as it goes: gripper straight down, then leaning outward
  - the same at several heights (floor, 5, 10, 15 cm)
  - pulled in as close to the base as it goes (near the floor and higher up)
  - left and right (base turned) to check it is the same all round
Type a word + Enter at any time to mark the current pose (e.g. "far", "near"), q + Enter to finish.

Saves config/reach.yaml: for each 2 cm height band, the furthest and closest tip distance seen,
with the lean and the joints used there. Picking uses it to know what it can reach and how.
"""

import sys
import threading
import time

import numpy as np
import yaml

from iro import ROOT

REACH = ROOT / "config" / "reach.yaml"
BAND = 0.02          # m: height bands
RATE = 20.0          # samples per second


def lean_deg(T):
    """Gripper lean from vertical in the plane through the base (deg; + = pointing outward)."""
    z = T[:3, 2]                                   # the tip's pointing direction
    u = T[:2, 3] / (np.linalg.norm(T[:2, 3]) or 1.0)
    return float(np.degrees(np.arctan2(z[:2] @ u, -z[2])))


def summarise(rows, table_z):
    """rows: (t, q[5] rad, tip xyz). Returns the reach.yaml dict."""
    q = np.array([r[1] for r in rows])
    tip = np.array([r[2] for r in rows])
    lean = np.array([r[3] for r in rows])
    h = tip[:, 2] - table_z
    rad = np.hypot(tip[:, 0], tip[:, 1])
    ok = h > -0.005                                # below the floor = calibration error: ignore
    bands = []
    for lo in np.arange(0.0, h[ok].max() + BAND, BAND) if ok.any() else []:
        m = ok & (h >= lo) & (h < lo + BAND)
        if m.sum() < 3:
            continue
        # 98th / 2nd percentile, not max / min: one jerk of the hand is not a reach
        far = np.percentile(rad[m], 98)
        near = np.percentile(rad[m], 2)
        i_far = np.flatnonzero(m)[np.argmin(np.abs(rad[m] - far))]
        i_near = np.flatnonzero(m)[np.argmin(np.abs(rad[m] - near))]
        vert = m & (np.abs(lean) < 10)             # gripper (nearly) straight down
        bands.append({
            "height_cm": [round(float(lo) * 100, 1), round(float(lo + BAND) * 100, 1)],
            "far_cm": round(float(far) * 100, 1), "far_lean_deg": round(float(lean[i_far]), 0),
            "far_joints_deg": np.round(np.degrees(q[i_far]), 1).tolist(),
            "near_cm": round(float(near) * 100, 1), "near_lean_deg": round(float(lean[i_near]), 0),
            "far_straight_down_cm": round(float(np.percentile(rad[vert], 98)) * 100, 1) if vert.sum() >= 3 else None,
            "samples": int(m.sum()),
        })
    from iro.kinematics import JOINTS
    return {
        "date": time.strftime("%Y-%m-%d %H:%M"), "table_z": float(table_z), "samples": len(rows),
        "bands": bands,
        "joint_range_seen_deg": {j: [round(float(np.degrees(q[:, i].min())), 1), round(float(np.degrees(q[:, i].max())), 1)]
                                 for i, j in enumerate(JOINTS)},
        "note": "measured by hand (iro arm reach); radii from the base axis, heights above the floor",
    }


def show(summary):
    print(f"\n{'height':>10} {'far':>7} {'lean':>6} {'far, straight down':>19} {'near':>7}")
    for b in summary["bands"]:
        fs = b["far_straight_down_cm"]
        print(f"{b['height_cm'][0]:4.0f}-{b['height_cm'][1]:<3.0f}cm {b['far_cm']:6.1f}cm {b['far_lean_deg']:+5.0f}° "
              f"{(f'{fs:.1f}cm' if fs else '-'):>19} {b['near_cm']:6.1f}cm")
    print("joint range used (deg):", ", ".join(f"{j} {a:.0f}..{b:.0f}" for j, (a, b) in summary["joint_range_seen_deg"].items()))


def from_run(path):
    import csv
    import json
    from iro.pick import pick_config
    rows = []
    with open(path / "reach.csv") as fh:
        for r in csv.DictReader(fh):
            rows.append((float(r["t"]), np.radians(json.loads(r["q_deg"])), np.array(json.loads(r["tip_m"])),
                         float(r["lean_deg"])))
    return summarise(rows, pick_config()["table_z"])


def cmd_reach():
    from iro.motion import Arm
    from iro.pick import pick_config
    from iro.record import run
    table_z = pick_config()["table_z"]
    a = input("The arm goes LIMP (torque off) so you can move it. Hold it now. Ready? [y/N] ").strip().lower()
    if a != "y":
        return 1
    r = run("arm-reach")
    rows, marks = [], []
    lines = []

    def reader():
        for line in sys.stdin:
            lines.append(line.strip())
            if line.strip().lower() == "q":
                return

    with Arm(dry_run=True) as arm:                 # dry_run: torque is never switched on here
        arm.bus.disable_torque()
        threading.Thread(target=reader, daemon=True).start()
        print(__doc__.split("Things to try:")[1].split("Saves")[0])
        t0 = time.time()
        best_far = 0.0
        while True:
            try:
                q = arm.joints()
            except ConnectionError:                  # a garbled read now and then: skip it
                continue
            T = arm.kin.fk(q)
            tip, ln = T[:3, 3], lean_deg(T)
            t = time.time() - t0
            rows.append((t, q[:5], tip, ln))
            r.log("reach", t=round(t, 3), q_deg=np.round(np.degrees(q[:5]), 2).tolist(),
                  tip_m=np.round(tip, 4).tolist(), lean_deg=round(ln, 1))
            rad, h = np.hypot(*tip[:2]), tip[2] - table_z
            best_far = max(best_far, rad if h > -0.005 else 0)
            print(f"\r tip {rad * 100:5.1f} cm out, {h * 100:5.1f} cm up, lean {ln:+4.0f}°   "
                  f"(furthest so far {best_far * 100:.1f} cm)   ", end="", flush=True)
            while lines:
                w = lines.pop(0)
                if w.lower() == "q":
                    break
                marks.append({"label": w or f"mark{len(marks) + 1}", "radius_cm": round(float(rad) * 100, 1),
                              "height_cm": round(float(h) * 100, 1), "lean_deg": round(float(ln), 0),
                              "joints_deg": np.round(np.degrees(q[:5]), 1).tolist()})
                r.note(f"mark {marks[-1]}")
                print(f"\n  marked '{marks[-1]['label']}': {rad * 100:.1f} cm out, {h * 100:.1f} cm up, lean {ln:+.0f}°")
            else:
                time.sleep(1 / RATE)
                continue
            break
    print("\nPut the arm down in a resting pose (it stays limp).")
    if len(rows) < 20:
        print("Too little recorded; nothing saved.")
        return 1
    s = summarise(rows, table_z)
    s["marks"] = marks
    s["recording"] = str(r.dir)
    REACH.write_text(yaml.safe_dump(s, sort_keys=False))
    show(s)
    print(f"\nSaved {REACH.relative_to(ROOT)}  (recording: {r.dir})")
    return 0


def main():
    a = sys.argv[1:]
    if a[:1] == ["--from"]:
        from iro.record import RUNS
        p = RUNS / a[1] if len(a) > 1 and not a[1].startswith("/") else __import__("pathlib").Path(a[1])
        s = from_run(p)
        s["recording"] = str(p)
        REACH.write_text(yaml.safe_dump(s, sort_keys=False))
        show(s)
        return 0
    return cmd_reach()


if __name__ == "__main__":
    sys.exit(main())
