"""iro measure: sizes of objects, and of the gripper's opening, without a ruler.

    iro measure                every object on the floor in front of the robot: L x W x H (OAK-D)
    iro measure NAME           measure one object and save it to objects/NAME/size.yaml
                               (--looks N: turn it between looks; the median is kept)
    iro measure gripper        the jaws' real opening (mm) at each gripper %, seen by the OAK-D:
                               saves config/gripper.yaml (max opening, % for any width)
    iro measure feel [NAME]    the gripper closes on what is between its jaws and reads the width
                               from the servo (a second, independent measurement)

Floor objects (the depth-camera "box measurement" method): the floor plane is fitted, everything
standing higher than 6 mm above it is mapped onto a top-down grid (5 mm cells), each separate blob
is one object: its footprint is the smallest rotated rectangle around it (length x width) and its
height is the 98th percentile above the floor. The OAK-D measures depth from ~22 cm, so objects
must be in the far part of its view (~25-60 cm in front of it). Plain white sponges get depth from
the IR dot projector. Every run is recorded (iro review).
"""

import sys
import time

import cv2
import numpy as np
import yaml

from iro import ROOT

CELL = 0.005               # m, top-down grid
MIN_H = 0.006              # m above the floor to count as an object
NEAR, FAR = 0.22, 0.60     # m from the OAK-D: depth starts at ~20 cm; the arm reaches ~15-40 cm
NOISE = 0.02               # threshold grows 2 cm per metre: the floor's 99% noise was 4 mm at 25 cm,
                           # 12 mm at 55 cm (21 Oct; stereo error grows with distance squared)
GRIPPER = ROOT / "config" / "gripper.yaml"


# ------------------------------------------------------------------ objects on the floor

def floor_frame(n):
    """Two directions along the floor (u roughly the camera's right, v away from it)."""
    u = np.array([1.0, 0.0, 0.0]) - n[0] * n
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def floor_bend(uv, h, rounds=3):
    """Smooth quadratic correction of the floor height (the measured floor bends up by ~7 mm at
    65 cm, 21 Oct), fitted to the points near the floor, robustly. Returns the correction per point."""
    A = np.c_[np.ones(len(uv)), uv, uv ** 2, uv[:, 0] * uv[:, 1]]
    sel = np.abs(h) < 0.02
    coef = np.zeros(A.shape[1])
    for _ in range(rounds):
        if sel.sum() < 200:
            break
        coef = np.linalg.lstsq(A[sel], h[sel], rcond=None)[0]
        res = h - A @ coef
        sel = np.abs(res) < max(0.003, 2.5 * np.std(res[sel]))
    return A @ coef


def floor_objects(oak, depth, fl, min_cells=12):
    """Objects standing on the floor: list of dicts (length, width, height in m; centre in camera
    coordinates; yaw on the floor; picture box), biggest first."""
    n, d = fl[0], fl[1]
    P = oak.points(depth, depth > 0)
    rng = np.linalg.norm(P, axis=1)
    P, rng = P[(rng > NEAR) & (rng < FAR)], rng[(rng > NEAR) & (rng < FAR)]   # where depth is reliable
    u, v = floor_frame(n)
    uv = np.c_[P @ u, P @ v]
    h = P @ n + d                                          # height above the flat floor ...
    h = h - floor_bend(uv, h)                              # ... corrected for its gentle bend
    keep = (h > MIN_H + NOISE * (rng - NEAR)) & (h < 0.4)
    P, h, uv, rng = P[keep], h[keep], uv[keep], rng[keep]
    if len(P) < 30:
        return []
    lo = uv.min(0)
    ij = np.floor((uv - lo) / CELL).astype(int)
    grid = np.zeros(ij.max(0) + 1, np.uint16)
    np.add.at(grid, (ij[:, 0], ij[:, 1]), 1)
    occ = cv2.morphologyEx((grid >= 2).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    k, lab = cv2.connectedComponents(occ, connectivity=8)
    cell_of = lab[ij[:, 0], ij[:, 1]]
    K = oak.K
    out = []
    for c in range(1, k):
        sel = cell_of == c
        if (lab == c).sum() < min_cells or sel.sum() < 50:
            continue
        if np.linalg.norm(P[sel], axis=1).max() > FAR - 0.03:
            continue                      # runs out of the measured range: furniture, or only partly seen
        hc = h[sel]
        height = float(np.percentile(hc, 98))
        # edge "flying pixels" sit low between object and floor: footprint from the upper part
        top = sel & (h > max(MIN_H, 0.3 * height))
        pts = uv[top] if top.sum() > 30 else uv[sel]
        (cx, cy), (a, b), ang = cv2.minAreaRect(pts.astype(np.float32))
        length, width = max(a, b), min(a, b)
        yaw = np.radians(ang if a >= b else ang + 90)
        centre = cx * u + cy * v - d * n                       # on the floor, camera coordinates
        Pc = P[sel]
        px = np.c_[K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2], K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2]]
        corners = cv2.boxPoints(((cx, cy), (a, b), ang))
        out.append({"length": float(length), "width": float(width), "height": height, "yaw_floor": float(yaw),
                    "centre_cam": centre, "box": [*np.percentile(px, 1, 0), *np.percentile(px, 99, 0)],
                    "corners_uv": corners, "points": int(sel.sum()), "uvh": (u, v, n, d)})
    return sorted(out, key=lambda o: -o["length"] * o["width"])


def draw_objects(img, objs, K):
    vis = img.copy()
    for i, o in enumerate(objs):
        u, v, n, d = o["uvh"]
        for hgt in (0.0, o["height"]):                             # footprint and top face
            pts = [cu * u + cv * v - d * n + hgt * n for cu, cv in o["corners_uv"]]
            px = np.array([[K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]] for p in pts], np.int32)
            cv2.polylines(vis, [px], True, (0, 255, 0) if hgt else (0, 160, 255), 2)
        x1, y1 = int(o["box"][0]), int(o["box"][1])
        cv2.putText(vis, f"{i + 1}: {o['length'] * 100:.1f} x {o['width'] * 100:.1f} x {o['height'] * 100:.1f} cm",
                    (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return vis


def look(oak, fl, n=3):
    """Several depth pictures; each object's sizes are the median over them."""
    shots = []
    for _ in range(n):
        img, depth = oak.read(5)
        shots.append(floor_objects(oak, depth, fl))
    base = shots[-1]
    for o in base:
        same = [min(s, key=lambda q: np.linalg.norm(q["centre_cam"] - o["centre_cam"])) for s in shots if s]
        same = [q for q in same if np.linalg.norm(q["centre_cam"] - o["centre_cam"]) < 0.03]
        for k in ("length", "width", "height"):
            o[k] = float(np.median([q[k] for q in same]))
        o["spread_mm"] = float(max(np.ptp([q["length"] for q in same]), np.ptp([q["width"] for q in same]),
                                   np.ptp([q["height"] for q in same])) * 1000)
    return img, base


def cmd_objects(name=None, looks=1):
    from iro.efad import floor_plane, load_calib
    from iro.oak import Oak
    from iro.record import run
    from iro.stream import feed
    r = run(f"measure-{name}" if name else "measure")
    f = feed()
    oak = Oak()
    try:
        _, depth = oak.read(5)
        fl = floor_plane(oak, depth)
        if fl is None:
            print("The OAK-D does not see enough floor.")
            return 1
        T = load_calib()
        results = []
        for k in range(looks):
            if looks > 1:
                input(f"Look {k + 1}/{looks}: put/turn it on the floor 25-60 cm in front of the OAK-D, Enter. ")
            img, objs = look(oak, fl)
            vis = draw_objects(img, objs, oak.K)
            r.snap(f"look{k + 1}", vis)
            r.video("measure", vis)
            f.show(vis, f"{len(objs)} object(s) measured")
            if not objs:
                print("  nothing standing on the floor in the measured area (25-60 cm in front of the OAK-D)")
                r.note("nothing found")
                continue
            for i, o in enumerate(objs):
                where = ""
                if T is not None:
                    c = T[:3, :3] @ o["centre_cam"] + T[:3, 3]
                    where = f"  at ({c[0] * 100:.0f}, {c[1] * 100:+.0f}) cm from the arm"
                line = (f"  {i + 1}: {o['length'] * 100:.1f} x {o['width'] * 100:.1f} x {o['height'] * 100:.1f} cm "
                        f"(L x W x H, looks agree within {o['spread_mm']:.0f} mm){where}")
                print(line)
                r.note(line)
                r.log("objects", look=k + 1, index=i + 1, length_m=o["length"], width_m=o["width"],
                      height_m=o["height"], spread_mm=o["spread_mm"], points=o["points"])
            if name:
                pick = objs[0]
                if len(objs) > 1:
                    print("  More than one object: DRAG a box around yours on the live page.")
                    f.show(vis, f"which one is {name}? drag a box around it")
                    b = f.ask_box(img.shape, timeout=120)
                    if b is not None:
                        c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                        pick = min(objs, key=lambda o: np.linalg.norm(c - np.array([(o["box"][0] + o["box"][2]) / 2,
                                                                                    (o["box"][1] + o["box"][3]) / 2])))
                results.append([pick["length"], pick["width"], pick["height"]])
        if name and results:
            L, W, H = np.median(np.array(results), 0)
            from iro.objects import OBJ_DIR
            out = OBJ_DIR / name / "size.yaml"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(yaml.safe_dump({"length_m": round(float(L), 4), "width_m": round(float(W), 4),
                                           "height_m": round(float(H), 4), "looks": len(results),
                                           "method": "OAK-D floor dimensioning", "recording": str(r.dir),
                                           "date": time.strftime("%Y-%m-%d %H:%M")}))
            print(f"Saved {name}: {L * 100:.1f} x {W * 100:.1f} x {H * 100:.1f} cm -> {out.relative_to(ROOT)}")
    finally:
        oak.close()
    return 0


def object_size(name):
    """(length, width, height) in m of a measured object, or None."""
    from iro.objects import OBJ_DIR
    f = OBJ_DIR / name / "size.yaml"
    if not f.exists():
        return None
    s = yaml.safe_load(f.read_text())
    return s["length_m"], s["width_m"], s["height_m"]


# ------------------------------------------------------------------ the gripper's opening

def gripper_raw(arm, pct):
    c = arm.cal["gripper"]
    return int(c["range_min"] + pct / 100 * (c["range_max"] - c["range_min"]))


def set_gripper(arm, pct, wait=0.8):
    arm.bus.write("Goal_Position", "gripper", gripper_raw(arm, pct), normalize=False)
    time.sleep(wait)


def jaw_gap_px(mask, tip_y):
    """Inner gap between the two jaws just above their tips (median over a band of rows), in
    pixels, plus the rows used; None if the jaws do not show as two prongs."""
    gaps = []
    for y in range(max(0, tip_y - 22), max(0, tip_y - 6)):
        row = mask[y] > 0
        edges = np.flatnonzero(np.diff(np.r_[0, row.astype(np.int8), 0]))
        runs = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]
        runs = [rr for rr in runs if rr[1] - rr[0] >= 3]
        if len(runs) >= 2:
            runs.sort(key=lambda rr: rr[0])
            # the two widest runs are the jaws
            a, b = sorted(sorted(runs, key=lambda rr: rr[0] - rr[1])[:2], key=lambda rr: rr[0])
            gaps.append(b[0] - a[1])
        elif len(runs) == 1:
            gaps.append(0)
    return (float(np.median(gaps)), len(gaps)) if gaps else (None, 0)


def cmd_gripper():
    """Opening curve of the gripper, measured by the OAK-D: % -> mm, max opening."""
    from iro.efad import arm_mask, jaw_direction, load_calib, park_and_shoot
    from iro.motion import Arm
    from iro.oak import Oak
    from iro.pick import pick_config
    from iro.record import run
    from iro.stream import feed
    T = load_calib()
    if T is None:
        print("Calibrate the OAK-D first: iro efad calibrate")
        return 1
    cfg = pick_config()
    r = run("measure-gripper")
    f = feed()
    oak = Oak()
    K = oak.K
    print("The gripper opens step by step in front of the OAK-D, which measures the gap. Keep clear.")
    table = cfg["table_z"]
    where = np.array([0.30, 0.0, table + 0.06])        # in view, upright (efad calibration grid)
    steps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    rows = []
    try:
        with Arm() as arm:
            level = arm.level_roll()
            # wrist roll that opens the jaws ACROSS the OAK-D's view (not toward it)
            view = T[:3, 3] - where
            view /= np.linalg.norm(view)
            best = None
            for dr in np.radians(np.arange(-90, 91, 15)):
                roll = level + dr
                if not (arm.kin.limits[4][0] <= roll <= arm.kin.limits[4][1]):
                    continue
                q, err = arm.kin.ik_down(where, np.r_[arm.joints()[:4], roll])
                if err > 0.004:
                    continue
                score = abs(np.dot(jaw_direction(arm.kin, q), view))      # 0 = across the view
                if best is None or score < best[0]:
                    best = (score, roll)
            roll = best[1] if best else level
            bg = park_and_shoot(arm, oak, roll)
            r.snap("empty_floor", bg)
            arm.move_tip(list(where), roll=roll)
            time.sleep(0.5)
            q = arm.joints()
            tipT = arm.kin.fk(q)
            tip_cam = T[:3, :3].T @ (tipT[:3, 3] - T[:3, 3])
            axis_cam = T[:3, :3].T @ jaw_direction(arm.kin, q)
            shrink = float(np.hypot(axis_cam[0], axis_cam[1]))          # how much of the opening faces the camera
            mm_per_px = tip_cam[2] / K[0, 0] * 1000 / max(shrink, 0.3)
            r.note(f"tip {tip_cam[2] * 100:.1f} cm from the OAK-D, opening faces it {shrink:.0%}, {mm_per_px:.2f} mm/px")
            for pct in steps:
                set_gripper(arm, pct)
                img, _ = oak.read(3)
                m = arm_mask(img, bg)
                ys = np.nonzero(m)[0]
                if len(ys) < 300:
                    print(f"  {pct:3d}%: arm not seen")
                    continue
                gap, used = jaw_gap_px(m, int(ys.max()))
                vis = img.copy()
                vis[m > 0] = (vis[m > 0] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)
                txt = f"{pct}%: gap {gap * mm_per_px:.1f} mm" if gap is not None else f"{pct}%: jaws not separated"
                cv2.putText(vis, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                r.video("gripper", vis)
                r.snap(f"open{pct:03d}", vis)
                f.show(vis, txt)
                if gap is not None:
                    rows.append((pct, gap * mm_per_px))
                    r.log("opening", pct=pct, gap_px=gap, gap_mm=gap * mm_per_px, rows=used)
                print("  " + txt)
            set_gripper(arm, 0)
            arm.move_tip([0.20, 0.0, table + 0.15])
    finally:
        oak.close()
    if len(rows) < 5:
        print(f"Too few readings. Recording: {r.dir} (iro review)")
        return 1
    pct, mm = np.array(rows).T
    mm = np.maximum.accumulate(mm)                         # opening only grows with %
    GRIPPER.write_text(yaml.safe_dump({"pct": pct.round(1).tolist(), "opening_mm": mm.round(1).tolist(),
                                       "max_open_mm": round(float(mm.max()), 1),
                                       "measured_at_cm": round(float(tip_cam[2] * 100), 1),
                                       "recording": str(r.dir), "date": time.strftime("%Y-%m-%d %H:%M")}))
    print(f"Max opening {mm.max():.0f} mm. Saved {GRIPPER.relative_to(ROOT)} (opening at every %).")
    return 0


def opening_mm(pct):
    """Jaw opening (mm) at a gripper % (from iro measure gripper), or None."""
    if not GRIPPER.exists():
        return None
    g = yaml.safe_load(GRIPPER.read_text())
    return float(np.interp(pct, g["pct"], g["opening_mm"]))


def pct_for(mm):
    """Gripper % that opens the jaws to `mm`, or None."""
    if not GRIPPER.exists():
        return None
    g = yaml.safe_load(GRIPPER.read_text())
    return float(np.interp(mm, g["opening_mm"], g["pct"]))


def cmd_feel(name=None):
    """Close gently on what is between the jaws; the width is where the jaw stops."""
    from iro.motion import Arm
    from iro.record import run
    if not GRIPPER.exists():
        print("Measure the gripper first: iro measure gripper")
        return 1
    r = run(f"measure-feel-{name}" if name else "measure-feel")
    with Arm() as arm:
        input("Hold the object between the open jaws (jaws open now), then Enter. ")
        set_gripper(arm, 100)
        input("Ready: Enter to close gently. ")
        c = arm.cal["gripper"]
        last, stop_pct = None, None
        for pct in np.arange(100, -1, -2):                 # step closed; stop where the jaw stops
            arm.bus.write("Goal_Position", "gripper", gripper_raw(arm, pct), normalize=False)
            time.sleep(0.08)
            pos = arm.bus.read("Present_Position", "gripper", normalize=False)
            load = abs(arm.bus.read("Present_Load", "gripper", normalize=False)) / 10
            now = 100 * (pos - c["range_min"]) / (c["range_max"] - c["range_min"])
            r.log("feel", command_pct=pct, jaw_pct=now, load_pct=load)
            if now - pct > 6 or load > 35:                 # commanded past where it is: touching
                stop_pct = now
                break
            last = now
        stop_pct = stop_pct if stop_pct is not None else (last or 0)
        w = opening_mm(stop_pct)
        set_gripper(arm, stop_pct + 8)                     # let go of the squeeze
    print(f"The jaws stopped at {stop_pct:.0f}%: the object is {w:.0f} mm wide there "
          "(soft things like sponges read a little narrower: they squash)")
    r.note(f"jaw stopped at {stop_pct:.1f}% -> {w:.1f} mm")
    return 0


def main():
    a = sys.argv[1:]
    looks = int(a[a.index("--looks") + 1]) if "--looks" in a else 1
    a = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] != "--looks")]
    if a and a[0] == "gripper":
        return cmd_gripper()
    if a and a[0] == "feel":
        return cmd_feel(a[1] if len(a) > 1 else None)
    if a and a[0] in ("-h", "help"):
        print(__doc__)
        return 0
    return cmd_objects(a[0] if a else None, looks)


if __name__ == "__main__":
    sys.exit(main())
