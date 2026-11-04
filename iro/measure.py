"""iro measure: sizes of objects, and of the gripper's opening, without a ruler.

    iro measure                every object on the floor in front of the robot: L x W x H (OAK-D)
    iro measure NAME           measure one object and save it to objects/NAME/size.yaml
                               (--looks N: turn it between looks; sizes are combined per side)
    iro measure NAME --from RUN [RUN..]   re-measure from recorded looks (no camera needed)
    iro measure gripper        the jaws' real opening (mm) at each gripper %, from the arm's own 3D
                               model + servo calibration: saves config/gripper.yaml
    iro measure feel [NAME]    the gripper as a caliper: width at FIRST CONTACT, and how soft it is
                               (a second, independent measurement; saved with NAME)

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
        # trim stray points (1% each side along the blob's own axes) before the tight rectangle
        mu = pts.mean(0)
        ev = np.linalg.eigh(np.cov((pts - mu).T))[1]
        pr = (pts - mu) @ ev
        lo_, hi_ = np.percentile(pr, 1, 0), np.percentile(pr, 99, 0)
        pts = pts[np.all((pr >= lo_) & (pr <= hi_), 1)]
        (cx, cy), (a, b), ang = cv2.minAreaRect(pts.astype(np.float32))
        length, width = max(a, b), min(a, b)
        yaw = np.radians(ang if a >= b else ang + 90)
        # which footprint side runs along the camera's line of sight: its far end is hidden behind
        # the object, so that size is only a LOWER BOUND (single-view dimensioning limit)
        view = np.array([u[2], v[2]])
        view /= np.linalg.norm(view) or 1
        side_a = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
        a_along = abs(side_a @ view) > np.sqrt(0.5)
        hidden = "length" if (a_along == (a >= b)) else "width"
        centre = cx * u + cy * v - d * n                       # on the floor, camera coordinates
        Pc = P[sel]
        px = np.c_[K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2], K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2]]
        corners = cv2.boxPoints(((cx, cy), (a, b), ang))
        out.append({"length": float(length), "width": float(width), "height": height, "yaw_floor": float(yaw),
                    "hidden": hidden,
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


def zone_overlay(img, K, fl):
    """Tint the floor where objects can be measured (NEAR..FAR from the OAK-D)."""
    h, w = img.shape[:2]
    xs, ys = np.meshgrid((np.arange(w) - K[0, 2]) / K[0, 0], (np.arange(h) - K[1, 2]) / K[1, 1])
    rays = np.dstack([xs, ys, np.ones_like(xs)])
    den = rays @ fl[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -fl[1] / den
    rng = t * np.linalg.norm(rays, axis=2)
    zone = (den < 0) & (rng > NEAR) & (rng < FAR)
    out = img.copy()
    out[zone] = (out[zone] * 0.75 + np.array([255, 120, 0]) * 0.25).astype(np.uint8)
    edge = cv2.morphologyEx(zone.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    out[edge] = (255, 120, 0)
    cv2.putText(out, f"measuring zone {NEAR * 100:.0f}-{FAR * 100:.0f} cm", (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 160, 60), 1)
    return out


def look(oak, fl, n=3):
    """Several depth pictures; each object's sizes are the median over them."""
    shots = []
    for _ in range(n):
        img, depth = oak.read(5)
        shots.append(floor_objects(oak, depth, fl))
    base = shots[-1]
    look.raw = (img, depth)                       # last raw picture, for the recording
    for o in base:
        same = [min(s, key=lambda q: np.linalg.norm(q["centre_cam"] - o["centre_cam"])) for s in shots if s]
        same = [q for q in same if np.linalg.norm(q["centre_cam"] - o["centre_cam"]) < 0.03]
        for k in ("length", "width", "height"):
            o[k] = float(np.median([q[k] for q in same]))
        o["spread_mm"] = float(max(np.ptp([q["length"] for q in same]), np.ptp([q["width"] for q in same]),
                                   np.ptp([q["height"] for q in same])) * 1000)
    return img, base


def fuse(results):
    """Combine looks. Each look: its three sizes largest first, each (metres, only_a_lower_bound).
    Per rank: the median of the side-on (reliable) values. End-on "at least" values are used only
    when no side-on value exists: they are noisy both ways (22 Oct, the tape's thickness: end-on
    1.4-3.3 cm, side-on 2.4 cm, gripper 2.65 cm). Returns [(metres, only_at_least)]."""
    final = []
    for k in range(3):
        good = [res[k][0] for res in results if not res[k][1]]
        low = [res[k][0] for res in results if res[k][1]]
        final.append((float(np.median(good)), False) if good else (float(max(low)), True))
    return final


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
                live, _ = oak.read(1)
                f.show(zone_overlay(live, oak.K, fl), "put it inside the blue zone")
                input(f"Look {k + 1}/{looks}: put/turn it inside the BLUE zone on the live page "
                      "(turn it so a different side faces the camera each time), Enter. ")
            img, objs = look(oak, fl)
            vis = draw_objects(zone_overlay(img, oak.K, fl), objs, oak.K)
            raw_img, raw_depth = look.raw            # raw data: any later method can re-measure this look
            np.savez_compressed(r.dir / f"look{k + 1}_raw.npz", image=raw_img, depth=raw_depth, K=oak.K,
                                floor_n=fl[0], floor_d=fl[1])
            r.snap(f"look{k + 1}", vis)
            r.video("measure", vis)
            f.show(vis, f"{len(objs)} object(s) measured")
            if not objs:
                print("  nothing standing on the floor inside the blue zone on the live page (25-60 cm from the OAK-D)")
                r.note("nothing found")
                continue
            for i, o in enumerate(objs):
                where = ""
                if T is not None:
                    c = T[:3, :3] @ o["centre_cam"] + T[:3, 3]
                    where = f"  at ({c[0] * 100:.0f}, {c[1] * 100:+.0f}) cm from the arm"
                hl = "at least " if o["hidden"] == "length" else ""
                hw = "at least " if o["hidden"] == "width" else ""
                line = (f"  {i + 1}: {hl}{o['length'] * 100:.1f} x {hw}{o['width'] * 100:.1f} x {o['height'] * 100:.1f} cm "
                        f"(L x W x H; 'at least' = its back is hidden from the camera; frames agree within "
                        f"{o['spread_mm']:.0f} mm){where}")
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
                results.append(sorted([(pick["length"], pick["hidden"] == "length"),
                                       (pick["width"], pick["hidden"] == "width"),
                                       (pick["height"], False)], reverse=True))
        if name and results:
            final = fuse(results)
            L, W, H = (f[0] for f in final)
            low = [n_ for n_, f in zip(("largest", "middle", "smallest"), final) if f[1]]
            if low:
                print(f"  Note: the {', '.join(low)} side was never seen side-on (its back hidden): it is AT LEAST "
                      "that. Turn the object so that side faces the camera, and measure again.")
            from iro.objects import OBJ_DIR
            out = OBJ_DIR / name / "size.yaml"
            out.parent.mkdir(parents=True, exist_ok=True)
            old = yaml.safe_load(out.read_text()) if out.exists() else {}
            old.update({"length_m": round(float(L), 4), "width_m": round(float(W), 4),
                        "height_m": round(float(H), 4), "looks": len(results),
                        "sizes_are": "largest, middle, smallest side (not tied to how it lies)",
                        "at_least": [n_ for n_, f in zip(("largest", "middle", "smallest"), final) if f[1]],
                        "method": "OAK-D floor dimensioning", "recording": str(r.dir),
                        "date": time.strftime("%Y-%m-%d %H:%M")})
            out.write_text(yaml.safe_dump(old))
            print(f"Saved {name}: {L * 100:.1f} x {W * 100:.1f} x {H * 100:.1f} cm -> {out.relative_to(ROOT)}")
    finally:
        oak.close()
    return 0


class _Lens:
    """Stand-in for the OAK-D when re-measuring a recorded look: just the lens numbers."""

    def __init__(self, K):
        self.K = K

    def points(self, depth, mask):
        from iro.oak import Oak
        return Oak.points(self, depth, mask)


def remeasure(name, runs):
    """Re-measure NAME from recorded looks (lookN_raw.npz in each run) with today's method, no
    camera needed, and save it. The biggest object in each look is taken (as when measuring)."""
    from iro.objects import OBJ_DIR
    from iro.record import RUNS
    results = []
    for rn in runs:
        d = RUNS / rn if (RUNS / rn).exists() else __import__("pathlib").Path(rn)
        for f in sorted(d.glob("look*_raw.npz")):
            z = np.load(f)
            objs = floor_objects(_Lens(z["K"]), z["depth"], (z["floor_n"], float(z["floor_d"])))
            if not objs:
                print(f"  {d.name}/{f.name}: nothing found")
                continue
            o = objs[0]
            results.append(sorted([(o["length"], o["hidden"] == "length"), (o["width"], o["hidden"] == "width"),
                                   (o["height"], False)], reverse=True))
            print(f"  {d.name}/{f.name}: " + " x ".join(f"{'at least ' if lb else ''}{v * 100:.1f}" for v, lb in results[-1]) + " cm")
    if not results:
        return 1
    final = fuse(results)
    out = OBJ_DIR / name / "size.yaml"
    size = yaml.safe_load(out.read_text()) if out.exists() else {}
    size.update({"length_m": round(final[0][0], 4), "width_m": round(final[1][0], 4), "height_m": round(final[2][0], 4),
                 "looks": len(results), "sizes_are": "largest, middle, smallest side (not tied to how it lies)",
                 "at_least": [n_ for n_, f in zip(("largest", "middle", "smallest"), final) if f[1]],
                 "method": "OAK-D floor dimensioning (re-measured from recordings)", "recording": [str(r) for r in runs],
                 "date": time.strftime("%Y-%m-%d %H:%M")})
    out.write_text(yaml.safe_dump(size))
    print(f"{name}: " + " x ".join(f"{'at least ' if lb else ''}{v * 100:.1f}" for v, lb in final)
          + f" cm (largest x middle x smallest, {len(results)} looks) -> {out.relative_to(ROOT)}")
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


def _stl(path):
    """Vertices of an STL file (binary or text)."""
    import struct
    b = open(path, "rb").read()
    if b[:5] == b"solid" and b"facet" in b[:300]:
        return np.array([list(map(float, ln.split()[1:4])) for ln in b.decode(errors="ignore").splitlines()
                         if ln.strip().startswith("vertex")])
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return a["v"].reshape(-1, 3).astype(float)


def jaw_model():
    """(fixed-jaw vertices, function hinge angle -> moving-jaw vertices), in the gripper_link frame,
    from the SO-101 URDF and its meshes (the same model RViz shows)."""
    import xml.etree.ElementTree as ET
    from iro.kinematics import URDF, _rpy, _tf
    root = ET.parse(URDF).getroot()
    assets = URDF.parent / "assets"

    def origin(el):
        o = el.find("origin")
        xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
        return _tf(_rpy(*rpy), xyz)

    def mesh(link, name):
        for vis in root.find(f"link[@name='{link}']").findall("visual"):
            m = vis.find("geometry/mesh")
            if m is not None and name in m.get("filename"):
                T = origin(vis)
                return _stl(assets / m.get("filename").split("/")[-1]) @ T[:3, :3].T + T[:3, 3]
        raise FileNotFoundError(name)

    fixed = mesh("gripper_link", "wrist_roll_follower")          # this part carries the fixed jaw
    jaw = mesh("moving_jaw_so101_v1_link", "moving_jaw")
    hinge = origin(root.find("joint[@name='gripper']"))

    def moving(theta):
        c, s_ = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        return jaw @ (hinge[:3, :3] @ R).T + hinge[:3, 3]
    return fixed, moving


def fingertip_gap(fixed, jaw, tip_len=0.015):
    """Smallest distance between the jaws over the last `tip_len` of the SHORTER jaw (where an
    object is held), and how far the moving jaw's tip falls short of the fixed one (m)."""
    out_f, out_m = -fixed[:, 2], -jaw[:, 2]                      # distance out along the gripper
    start = min(out_f.max(), out_m.max()) - tip_len
    a, b = fixed[out_f > start], jaw[out_m > start]
    a, b = a[::max(1, len(a) // 2500)], b[::max(1, len(b) // 2500)]
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min()
    return float(d), float(out_f.max() - out_m.max())


def gripper_curve(arm_cal):
    """Opening at every gripper %, from the jaw model and the servo calibration."""
    fixed, moving = jaw_model()
    c = arm_cal["gripper"]
    rows = []
    for pct in range(0, 101, 2):
        raw = c["range_min"] + pct / 100 * (c["range_max"] - c["range_min"])
        theta = (raw - 2048) * 2 * np.pi / 4096                 # same mapping as iro.motion (gripper offset 0)
        gap, short = fingertip_gap(fixed, moving(theta))
        rows.append((pct, np.degrees(theta), gap * 1000, short * 1000))
    return rows


def cmd_gripper():
    """The jaws' real opening at every gripper %, from the arm's own 3D model + servo calibration.
    21 Oct: the model's jaws touch at -11 deg, the calibrated fully-closed servo position is
    -12.5 deg: they agree. (Measuring the gap in the OAK-D picture was tried first: the moving jaw
    swings on a hinge, out of view when wide open, so the picture could not show the gap.)"""
    import json
    from iro.record import run
    r = run("measure-gripper")
    cal = json.load(open(ROOT / "arm" / "calibration" / "robots" / "so_follower" / "iro_follower.json"))
    rows = gripper_curve(cal)
    usable = [row for row in rows if row[3] < 0.020 * 1000]      # fingertips still within 2 cm of each other
    closed = min(rows, key=lambda row: row[2])
    max_row = max(usable, key=lambda row: row[2])
    for pct, th, gap, short in rows[::5]:
        print(f"  {pct:3d}%  hinge {th:6.1f} deg  opening {gap:5.1f} mm" + ("" if short < 20 else "  (jaw swung away: not usable)"))
    print(f"Closed at {closed[0]}% ({closed[2]:.1f} mm). Widest usable opening {max_row[2]:.0f} mm at {max_row[0]}%.")
    # chart for the recording
    img = np.full((300, 520, 3), 255, np.uint8)
    for x in range(0, 101, 10):
        cv2.line(img, (40 + x * 4, 20), (40 + x * 4, 270), (230, 230, 230), 1)
        cv2.putText(img, str(x), (32 + x * 4, 288), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
    pts = np.array([[40 + row[0] * 4, 270 - row[2] * 2.5] for row in rows], np.int32)
    cv2.polylines(img, [pts], False, (200, 100, 0), 2)
    cv2.putText(img, f"jaw opening (mm) vs gripper %, widest usable {max_row[2]:.0f} mm", (40, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    r.snap("opening_curve", img)
    for pct, th, gap, short in rows:
        r.log("opening", pct=pct, hinge_deg=th, opening_mm=gap, tip_mismatch_mm=short)
    keep = [row for row in rows if row[0] <= max_row[0]]
    GRIPPER.write_text(yaml.safe_dump({
        "pct": [row[0] for row in keep], "opening_mm": [round(row[2], 1) for row in keep],
        "max_open_mm": round(max_row[2], 1), "max_open_pct": max_row[0],
        "method": "SO-101 URDF jaw meshes + servo calibration (closed at -12.5 deg; model -11 deg)",
        "date": time.strftime("%Y-%m-%d %H:%M")}))
    print(f"Saved {GRIPPER.relative_to(ROOT)}. Check it on a real object: iro measure feel")
    return 0


def opening_mm(pct):
    """Jaw opening (mm) at a gripper % (from iro measure gripper), or None."""
    if not GRIPPER.exists():
        return None
    g = yaml.safe_load(GRIPPER.read_text())
    return float(np.interp(pct, g["pct"], g["opening_mm"]))


def max_open_m(default=0.07):
    """Widest object (m) the jaws close around with both fingertips (iro measure gripper)."""
    if not GRIPPER.exists():
        return default
    return yaml.safe_load(GRIPPER.read_text())["max_open_mm"] / 1000


def pct_for(mm):
    """Gripper % that opens the jaws to `mm`, or None."""
    if not GRIPPER.exists():
        return None
    g = yaml.safe_load(GRIPPER.read_text())
    return float(np.interp(mm, g["opening_mm"], g["pct"]))


RIGID_MM_PER_LOAD = 0.22   # how far the jaws still give per % of load on a HARD object (the tape,
                           # 21 Oct: 0.21-0.25): the gripper's own flex. Much more = a soft object.


HOLD_PCT = 4               # carrying: aim this % past first contact (~6 mm into the gripper's own flex)


def find_contact(cmd, jaw, load, skip=4, base_min=4, final=False):
    """Index of first contact while closing, or None (also None while it cannot be sure yet).
    - the first `skip` readings are the jaw speeding up (lag 0 -> 1.5 %, load 2 -> 8 %): ignored
      (22 Oct: counting them as contact "measured" 71 mm without touching anything)
    - the free-closing baseline is the median of at least `base_min` steady readings before it
    - contact: the jaw starts trailing the command, or the load rises, 2 readings in a row
    - confirmed only if the jaw is then held back MORE (lag +2 % within 3 readings): a real object
      stops the jaw (the tape: lag grew to +7 %); a wobble does not
    - final=True (closing has stopped, e.g. at the load limit): confirm with the readings there are
      (22 Oct: a tape lying on the floor hit the load limit 3 readings after contact; waiting for a
      4th reading reported "nothing" every time)"""
    lag = np.asarray(jaw, float) - np.asarray(cmd, float)
    load = np.asarray(load, float)
    for i in range(skip + base_min, len(lag) - 1):
        lag0, load0 = float(np.median(lag[skip:i])), float(np.median(load[skip:i]))
        if (lag[i] > lag0 + 0.5 and lag[i + 1] > lag0 + 0.5) or (load[i] > load0 + 5 and load[i + 1] > load0 + 5):
            later = lag[i:i + 4]
            if later.max() > lag0 + 2:
                return i
            if len(later) < 4 and not final:
                return None                            # not enough readings yet to confirm
    return None


def close_on(arm, start, r, step=2, hold=False):
    """Close from `start` % until FIRST CONTACT (find_contact). Returns None if nothing was gripped
    (the jaws end closed), else {touch_pct, width_mm, give_mm, softness, feel}. hold=True: a gentle
    carrying grip (HOLD_PCT past contact); else the jaws are left where they stopped."""
    c = arm.cal["gripper"]
    cmd, jaw, load = [], [], []
    hit = None
    for pct in np.arange(start, -1, -step):
        arm.bus.write("Goal_Position", "gripper", gripper_raw(arm, pct), normalize=False)
        time.sleep(0.08)
        pos = arm.bus.read("Present_Position", "gripper", normalize=False)
        ld = abs(arm.bus.read("Present_Load", "gripper", normalize=False)) / 10
        cmd.append(float(pct))
        jaw.append(100 * (pos - c["range_min"]) / (c["range_max"] - c["range_min"]))
        load.append(ld)
        r.log("grip", command_pct=pct, jaw_pct=jaw[-1], load_pct=ld)
        hit = find_contact(cmd, jaw, load)
        if hit is not None or ld > 35:                    # confirmed, or squeezing hard: stop
            break
    if hit is None:
        hit = find_contact(cmd, jaw, load, final=True)    # stopped early: judge what was read
    if hit is None:
        return None
    touch = jaw[hit - 1]                                  # last reading before the jaw was held back
    width = opening_mm(touch)
    give = width - opening_mm(jaw[-1])
    per_load = give / max(load[-1] - load[hit - 1], 1)
    soft = per_load / RIGID_MM_PER_LOAD
    feel = "hard" if soft < 1.6 else ("a bit soft" if soft < 4 else "soft (squashes)")
    r.note(f"contact {touch:.2f}% -> {width:.1f} mm; give {give:.1f} mm, {per_load:.3f} mm/% load, softness x{soft:.1f}")
    # never leave the jaw commanded far past the object: the servo pushes at full force and trips
    # its overload protection (22 Oct: the first real pick stopped lifting with "overloaded")
    arm.bus.write("Goal_Position", "gripper", gripper_raw(arm, max(0.0, touch - HOLD_PCT) if hold else jaw[-1]),
                  normalize=False)
    return {"touch_pct": touch, "width_mm": width, "give_mm": give, "softness": soft, "feel": feel,
            "held_pct": jaw[-1]}


def cmd_feel(name=None):
    """The gripper as a caliper: close slowly, note the jaw position at FIRST CONTACT (the width),
    squeeze a little more to feel how soft it is, let go. 21 Oct: the first version reported where
    the jaw finally stopped, after squeezing 7 mm into a hard tape (20 mm instead of 26.5)."""
    from iro.motion import Arm
    from iro.record import run
    if not GRIPPER.exists():
        print("Measure the gripper first: iro measure gripper")
        return 1
    g = yaml.safe_load(GRIPPER.read_text())
    start = min(100, g["max_open_pct"] + 6)                # widest opening with the fingertips in line
    r = run(f"measure-feel-{name}" if name else "measure-feel")
    with Arm() as arm:
        set_gripper(arm, start)
        input(f"Hold the object between the jaws (open {opening_mm(start):.0f} mm now), then Enter to close gently. ")
        g_ = close_on(arm, start, r)
        if g_ is None:
            set_gripper(arm, start)
            print("Nothing was gripped: the jaws closed without being held back by anything. Nothing saved. "
                  "Hold the object between the fingertips and try again.")
            r.note("no contact: nothing saved")
            return 1
        set_gripper(arm, g_["touch_pct"] + 6)              # let go of the squeeze
    width, soft, feel = g_["width_mm"], g_["softness"], g_["feel"]
    print(f"First contact at {g_['touch_pct']:.1f}%: the object is {width:.1f} mm wide between the fingertips. "
          f"It gave {g_['give_mm']:.1f} mm when squeezed ({soft:.1f}x the gripper's own flex): {feel}.")
    if name:
        from iro.objects import OBJ_DIR
        f = OBJ_DIR / name / "size.yaml"
        f.parent.mkdir(parents=True, exist_ok=True)
        size = yaml.safe_load(f.read_text()) if f.exists() else {}
        grips = size.get("grip_widths_m", []) + [round(width / 1000, 4)]
        size.update({"grip_widths_m": grips[-10:], "grip_width_m": round(float(np.median(grips[-10:])), 4),
                     "softness": round(soft, 2), "feel": feel})
        f.write_text(yaml.safe_dump(size))
        dims = [size[k] for k in ("length_m", "width_m", "height_m") if k in size]
        if dims:
            near = min(dims, key=lambda d: abs(d - width / 1000))
            print(f"  The OAK-D measured {near * 1000:.0f} mm for the closest side: "
                  f"{'they agree' if abs(near * 1000 - width) < 5 else 'they DISAGREE'} ({abs(near * 1000 - width):.0f} mm apart).")
        print(f"  Saved to {f.relative_to(ROOT)} (median of the last {len(grips[-10:])} feels: {np.median(grips[-10:]) * 1000:.1f} mm)")
    return 0


def main():
    a = sys.argv[1:]
    looks = int(a[a.index("--looks") + 1]) if "--looks" in a else 1
    a = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] != "--looks")]
    a = a[:1] if "--from" in sys.argv else a
    if a and a[0] == "gripper":
        return cmd_gripper()
    if a and a[0] == "feel":
        return cmd_feel(a[1] if len(a) > 1 else None)
    if a and a[0] in ("-h", "help"):
        print(__doc__)
        return 0
    if a and "--from" in sys.argv:
        return remeasure(a[0], sys.argv[sys.argv.index("--from") + 1:])
    return cmd_objects(a[0] if a else None, looks)


if __name__ == "__main__":
    sys.exit(main())
