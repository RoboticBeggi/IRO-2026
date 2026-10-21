"""EFAD - Embedded Foundational Augmented Detection: the robot's perception, and picking with it.

    iro efad see            OAK-D live view: names + 3D position of remembered objects
    iro efad wrist          wrist-camera live view (was `iro vision`)
    iro efad show NAME      teach an object to the OAK-D (colour) in ~15 s, no training
    iro efad calibrate      find where the OAK-D sits relative to the arm (NO marker: the gripper
                            opens/closes in front of it and is found by itself; the floor double-checks)
    iro efad calibrate --test   first: only checks the OAK-D finds the gripper (12 poses, saves nothing)
    iro efad check          has the OAK-D moved since? (re-measures the floor, ~3 s)
Every run is recorded (~/iro-runs, iro review): calibration saves a 4-panel picture per pose.
    iro efad pick NAME      find NAME in 3D and pick it up      (--dry-run, --drop)

How the pieces work together:
  OAK-D Pro W (colour + IR-assisted stereo depth): finds every thing, names it with the instant
    memory (iro object show), and measures its 3D centre, top height and long axis
  geometry: gripper straight down over the centre, jaws turned to close across the NARROW side,
    gripping in the upper part of the object (sponges squash, so this is forgiving)
  wrist camera (global shutter): confirms it is the right object right before the grasp
  servos: the 'is something in the jaws' check after closing
No training: an object remembered with `iro object show` can be picked straight away.
"""

import sys
import time

import cv2
import numpy as np
import yaml

from iro import ROOT

CALIB = ROOT / "config" / "efad_calibration.yaml"
SNAP = ROOT.parent / "efad-last.jpg"


def load_calib():
    if not CALIB.exists():
        return None
    c = yaml.safe_load(CALIB.read_text())
    return np.array(c["T_base_cam"])


def to_base(T, p):
    return T[:3, :3] @ np.asarray(p) + T[:3, 3]


# --- calibration WITHOUT markers -------------------------------------------------
#
# 1. The floor: a plane fit to the OAK-D's depth gives its height, tilt and roll above the floor
#    (tested 21 Oct: floor flat to 1.3 mm, height 11.8 cm vs 11.5 cm by tape measure).
# 2. The gripper finds itself: at each arm pose the jaw opens and closes; the only thing that
#    changes between the two pictures is the moving jaw, so the change, with its depth, is a
#    3D point on the gripper. 12+ poses -> where the OAK-D sits relative to the arm.
# 3. The floor fixes tilt, roll and height exactly (solve_with_floor); the gripper sightings
#    only solve the rest. Checked: the sightings alone must put the camera within 2.5 cm.
# `iro efad check` re-measures the floor at any time and says if the OAK-D mount has moved.

def floor_plane(oak, depth, iters=300):
    """(unit normal pointing up from the floor, distance) in camera coordinates, flatness rms (m),
    or None. Uses the lower half of the picture (mostly floor) and RANSAC."""
    h, w = depth.shape
    mask = np.zeros(depth.shape, bool)
    mask[h // 2:, :] = True
    P = oak.points(depth, mask)[::5]
    if len(P) < 500:
        return None
    rng, best = np.random.default_rng(0), None
    for _ in range(iters):
        s3 = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s3[1] - s3[0], s3[2] - s3[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - s3[0]) @ n) < 0.01
        if best is None or inl.sum() > best.sum():
            best = inl
    Q = P[best]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c, full_matrices=False)[2][2]   # full_matrices=True: a 16k x 16k matrix, 2 GB
    if n[1] > 0:                       # camera y points down: the floor's "up" has negative y
        n = -n
    return n, float(-n @ c), float(np.sqrt(np.mean(((Q - c) @ n) ** 2)))


def mount_from_floor(n, d):
    """Height above the floor (m), tilt down and roll (deg) of the camera, from the floor plane."""
    return abs(d), float(np.degrees(np.arcsin(abs(n[2])))), float(np.degrees(np.arctan2(n[0], -n[1])))


MIN_RANGE = 0.22            # m: the OAK-D Pro W measures depth from ~20 cm (400p, extended disparity)


def jaw_change(oak, a, b):
    """What changed between two (colour, depth) shots. Returns (point or None, blob mask, info)."""
    (ia, da), (ib, db) = a, b
    ga, gb = cv2.cvtColor(ia, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ib, cv2.COLOR_BGR2GRAY)
    col = cv2.absdiff(cv2.GaussianBlur(ga, (5, 5), 0), cv2.GaussianBlur(gb, (5, 5), 0)) > 30
    both = (da > 0) & (db > 0)
    dep = both & (np.abs(da.astype(np.float32) - db.astype(np.float32)) > 15)
    # the jaw must be SEEN to move (colour); depth-only change is flicker (21 Oct: a far-floor
    # flicker gave a false "found at 115 cm" with 0 changed colour pixels)
    m = cv2.morphologyEx(col.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    info = {"colour_px": int(col.sum()), "depth_px": int(dep.sum())}
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        info["why"] = "nothing changed (gripper out of view?)"
        return None, m > 0, info
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    blob = lab == k
    info["blob_px"] = int(st[k, cv2.CC_STAT_AREA])
    info["blob_box"] = [int(v) for v in st[k, :4]]
    if info["blob_px"] < 120:
        info["why"] = f"change too small ({info['blob_px']} px)"
        return None, blob, info
    ys, xs = np.nonzero(blob)
    info["pix"] = [float(xs.mean()), float(ys.mean())]          # where the jaw moved, in the picture
    info["why"] = ""
    # depth is a bonus: the lower half of the picture is closer than the ~20 cm depth needs
    near = np.where((da > 0) & ((db == 0) | (da < db)), da, db)     # the jaw is the NEARER surface
    info["depth_valid"] = round(float(((near > 0) & blob).sum() / blob.sum()), 2)
    P = oak.points(near, blob)
    p = None
    if len(P) >= 40 and info["depth_valid"] > 0.5:
        P = P[P[:, 2] < np.percentile(P[:, 2], 50) + 0.03]
        q = np.median(P, 0)
        if np.linalg.norm(q) >= MIN_RANGE:
            p = q
            info["dist_m"] = round(float(np.linalg.norm(q)), 3)
    return p, blob, info


def panel(shots, mask, text):
    """2x2 picture for the recording: open | closed / what changed | depth."""
    (ia, da), (ib, _) = shots[0], shots[1]
    ov = ib.copy()
    ov[mask] = (ov[mask] * 0.3 + np.array([0, 255, 0]) * 0.7).astype(np.uint8)
    dv = cv2.applyColorMap(np.uint8(np.clip(da / 1500.0, 0, 1) * 255), cv2.COLORMAP_TURBO)
    dv[da == 0] = 0
    tiles = [cv2.resize(x, (320, 200)) for x in (ia, ib, ov, dv)]
    for t, lab in zip(tiles, ("open", "closed", "changed", "depth (black = none)")):
        cv2.putText(t, lab, (6, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    out = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.putText(out, text[:80], (6, 216), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return out


def find_jaw(arm, oak):
    """Where the moving jaw is: difference of open vs closed pictures, done twice (open-close-open);
    both must agree within 12 px. Returns (pixel or None, 3D camera point if depth was valid, panel
    picture, info)."""
    shots = []
    for opened in (True, False, True):
        arm.gripper(opened, wait=0.8)
        oak.read(1)                               # drop a picture taken while it was still moving
        shots.append(oak.read(3))
    p1, m1, i1 = jaw_change(oak, shots[0], shots[1])
    p2, m2, i2 = jaw_change(oak, shots[1], shots[2])
    info = {**i1, "second": i2.get("why", "")}
    pix, p = None, None
    if "pix" not in i1 or "pix" not in i2:
        info["why"] = i1["why"] or i2["why"]
    elif np.hypot(*np.subtract(i1["pix"], i2["pix"])) > 12:
        info["why"] = f"two looks disagree by {np.hypot(*np.subtract(i1['pix'], i2['pix'])):.0f} px (something else moved?)"
    else:
        pix = list(np.mean([i1["pix"], i2["pix"]], 0))
        if p1 is not None and p2 is not None and np.linalg.norm(p1 - p2) < 0.015:
            p = (p1 + p2) / 2
    info["pix"] = pix
    txt = ("NOT found: " + info["why"]) if pix is None else (
        f"FOUND at pixel ({pix[0]:.0f},{pix[1]:.0f})" + (f", depth {np.linalg.norm(p) * 100:.0f} cm" if p is not None else ", no depth (too close)"))
    pic = panel(shots, m1 | m2, txt)
    if pix is not None:
        cv2.drawMarker(pic, (int(pix[0] / 2), int(pix[1] / 2)), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
    return pix, p, pic, info


def solve_calibration(cam_pts, grip_T):
    """Camera->base transform AND where the seen point sits on the gripper, from pairs of
    (point seen by the camera, gripper pose from the arm). Alternates Kabsch / least squares."""
    cam_pts = np.asarray(cam_pts)
    off = np.zeros(3)
    for _ in range(100):
        base_pts = np.array([T[:3, :3] @ off + T[:3, 3] for T in grip_T])
        mc, mb = cam_pts.mean(0), base_pts.mean(0)
        U, _, Vt = np.linalg.svd((cam_pts - mc).T @ (base_pts - mb))
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        t = mb - R @ mc
        # offset: least squares  R_i off = R p_i + t - t_i
        A = np.vstack([T[:3, :3] for T in grip_T])
        b = np.concatenate([R @ p + t - T[:3, 3] for p, T in zip(cam_pts, grip_T)])
        off_new = np.linalg.lstsq(A, b, rcond=None)[0]
        if np.linalg.norm(off_new - off) < 1e-7:
            off = off_new
            break
        off = off_new
    T_bc = np.eye(4)
    T_bc[:3, :3], T_bc[:3, 3] = R, t
    res = [np.linalg.norm(R @ p + t - (Tg[:3, :3] @ off + Tg[:3, 3])) for p, Tg in zip(cam_pts, grip_T)]
    return T_bc, off, float(np.sqrt(np.mean(np.square(res)))), res


def solve_with_floor(cam_pts, grip_T, n_cam, d_cam, table_z, iters=200):
    """Like solve_calibration, but the floor fixes tilt, roll and height exactly: the floor normal
    seen by the camera must become the arm's +z and the floor must sit at table_z. Only the turn
    about z, the x/y position and the point on the gripper are left for the gripper sightings
    (far fewer unknowns: simulated with 5 mm sighting noise, 4x smaller error)."""
    cam_pts = np.asarray(cam_pts)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n_cam, z)
    c = float(n_cam @ z)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R0 = np.eye(3) + vx + vx @ vx / (1 + c)                 # turns the floor normal onto +z
    tz = table_z + d_cam                                    # floor point -d*n lands at z = table_z
    C = (R0 @ cam_pts.T).T
    off = np.zeros(3)
    for _ in range(iters):
        B = np.array([T[:3, :3] @ off + T[:3, 3] for T in grip_T])
        mc, mb = C[:, :2].mean(0), B[:, :2].mean(0)
        H = (C[:, :2] - mc).T @ (B[:, :2] - mb)
        yaw = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        txy = mb - (Rz[:2, :2] @ mc)
        R, t = Rz @ R0, np.r_[txy, tz]
        A = np.vstack([T[:3, :3] for T in grip_T])
        b = np.concatenate([R @ p + t - T[:3, 3] for p, T in zip(cam_pts, grip_T)])
        new = np.linalg.lstsq(A, b, rcond=None)[0]
        done = np.linalg.norm(new - off) < 1e-8
        off = new
        if done:
            break
    T_bc = np.eye(4)
    T_bc[:3, :3], T_bc[:3, 3] = R, t
    res = [np.linalg.norm(R @ p + t - (Tg[:3, :3] @ off + Tg[:3, 3])) for p, Tg in zip(cam_pts, grip_T)]
    return T_bc, off, float(np.sqrt(np.mean(np.square(res)))), res


def floor_rotation(n_cam):
    """Rotation turning the camera's floor normal onto the arm's +z."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n_cam, z)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + float(n_cam @ z))


def solve_pixels(pix, grip_T, K, n_cam, d_cam, table_z, depth_pts=None):
    """Where the OAK-D sits, from WHERE IN THE PICTURE the gripper was seen at known arm poses
    (no depth needed: the gripper is closer than the depth camera's 20 cm in the lower half of
    the picture). The floor fixes tilt, roll and height; this solves the turn about z, the x/y
    position and the seen point on the gripper by Gauss-Newton on the pixel error, from several
    starting turns. depth_pts: optional {index: camera point} where depth was valid (also used).
    Returns (T_base_cam, point on gripper, rms pixel error, per-pose pixel errors)."""
    pix = np.asarray(pix, float)
    R0 = floor_rotation(n_cam)
    tz = table_z + d_cam
    Rg = np.array([T[:3, :3] for T in grip_T])
    tg = np.array([T[:3, 3] for T in grip_T])
    depth_pts = depth_pts or {}

    def pose(x):
        c, s_ = np.cos(x[0]), np.sin(x[0])
        R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) @ R0
        return R, np.array([x[1], x[2], tz])

    def resid(x):
        R, t = pose(x)
        pb = np.einsum("nij,j->ni", Rg, x[3:6]) + tg            # the gripper point, arm frame
        pc = (pb - t) @ R                                         # ... camera frame
        z = np.maximum(pc[:, 2], 0.02)
        r = [(K[0, 0] * pc[:, 0] / z + K[0, 2] - pix[:, 0]), (K[1, 1] * pc[:, 1] / z + K[1, 2] - pix[:, 1])]
        r += [np.where(pc[:, 2] > 0.02, 0, 500.0)]              # behind the camera: heavy penalty
        out = np.concatenate(r)
        if depth_pts:                                            # 1 cm of depth error ~ 10 px
            out = np.concatenate([out] + [1000 * (pc[i] - np.asarray(q)) for i, q in depth_pts.items()])
        return out

    best = None
    for yaw0 in np.radians(np.arange(0, 360, 30)):
        x = np.array([yaw0, 0.0, 0.0, 0.0, 0.0, 0.03])
        lam = 1e-2
        for _ in range(200):
            r = resid(x)
            J = np.empty((len(r), 6))
            for k in range(6):
                dx = np.zeros(6)
                dx[k] = 1e-6
                J[:, k] = (resid(x + dx) - r) / 1e-6
            step = np.linalg.solve(J.T @ J + lam * np.diag(np.diag(J.T @ J) + 1e-9), -J.T @ r)
            if (resid(x + step) ** 2).sum() < (r ** 2).sum():
                x, lam = x + step, lam * 0.3
                if np.linalg.norm(step) < 1e-9:
                    break
            else:
                lam *= 10
                if lam > 1e8:
                    break
        cost = (resid(x) ** 2).sum()
        if best is None or cost < best[0]:
            best = (cost, x)
    x = best[1]
    R, t = pose(x)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    n = len(pix)
    e = resid(x)[:2 * n].reshape(2, n)
    per = np.hypot(e[0], e[1])
    return T, x[3:6], float(np.sqrt(np.mean(per ** 2))), per


def calibration_poses(table):
    """Where the OAK-D can see the gripper AND measure its depth (21 Oct test, recorded): it sits
    low (11.7 cm) tilted 31 deg down, so the gripper is only in view when LOW (higher than ~10 cm
    it is above the picture) and must be >= 22 cm away (depth starts at ~20 cm): low and far
    forward. Serpentine order keeps moves short."""
    out = []
    for i, hh in enumerate((0.03, 0.06, 0.09)):
        xs = (0.26, 0.30, 0.34, 0.38) if i % 2 == 0 else (0.38, 0.34, 0.30, 0.26)
        for j, x in enumerate(xs):
            ys = (-0.10, -0.03, 0.04, 0.11)
            for y in (ys if (i + j) % 2 == 0 else ys[::-1]):
                out.append((x, y, table + hh, 0.0))
    out += [(0.32, 0.0, table + 0.05, np.radians(r)) for r in (-40, 40, -20)]   # roll variety
    return out


def cmd_calibrate(test=False):
    """Markerless calibration (test=True: only find the gripper at a few poses, save nothing)."""
    from iro.motion import Arm
    from iro.oak import Oak
    from iro.pick import pick_config
    from iro.record import run
    from iro.stream import feed
    cfg = pick_config()
    r = run("efad-calibrate-test" if test else "efad-calibrate")
    oak = Oak()
    f = feed()
    img, depth = oak.read(5)
    r.snap("start_oak", img)
    fl = floor_plane(oak, depth)
    if fl is None:
        print("The OAK-D does not see enough floor. Clear the floor in front of the robot and try again.")
        r.note("no floor seen")
        return 1
    h, tilt, roll = mount_from_floor(fl[0], fl[1])
    msg = f"Floor: OAK-D is {h * 100:.1f} cm above it, tilted {tilt:.1f} deg down, roll {roll:+.1f} deg (floor flat to {fl[2] * 1000:.1f} mm)"
    print(msg)
    r.note(msg)
    print("The arm visits poses in front of the robot and opens/closes the gripper at each, so the OAK-D "
          "finds the gripper by itself (no marker). Keep the area clear and nothing else moving.")
    input("Ready? Enter. ")
    pixels, depth_pts, grip_T, tried, seen_any = [], {}, [], 0, False
    with Arm() as arm:
        roll0 = arm.level_roll()
        table = cfg["table_z"]
        poses = calibration_poses(table)
        if test:
            poses = [poses[k] for k in (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33)]
        for i, (x, y, z, dr) in enumerate(poses):
            reached = None
            for lean in (0.0, -0.5, -0.9):     # far poses: lean the gripper forward (no need to point down)
                try:
                    arm.move_tip([x, y, z], roll=roll0 + dr, tilt=lean)
                    reached = lean
                    break
                except Exception as e:     # noqa: BLE001 - try the next lean
                    err = str(e)
            if reached is None:
                print(f"  pose {i + 1}: skipped, cannot reach ({err[:50]})")
                r.log("poses", pose=i + 1, x=x, y=y, z=z, result="unreachable")
                continue
            time.sleep(0.4)
            q = arm.joints()
            tip = arm.kin.fk(q, "gripper_link")[:3, 3]
            pix, p, pic, info = find_jaw(arm, oak)
            tried += 1
            seen_any |= info.get("colour_px", 0) > 300 or info.get("blob_px", 0) >= 120
            label = f"pose {i + 1}/{len(poses)} at ({x * 100:.0f},{y * 100:+.0f},{(z - table) * 100:.0f} cm above floor)"
            r.video("calibrate", pic, [label])
            r.snap(f"pose{i + 1:02d}", pic)
            f.show(pic, f"{label}: {'found' if pix is not None else info['why']}")
            r.log("poses", pose=i + 1, x=x, y=y, z=z, lean=reached, result="found" if pix is not None else info["why"],
                  colour_px=info.get("colour_px"), blob_px=info.get("blob_px"), pix=info.get("pix"),
                  depth_valid=info.get("depth_valid"), dist_m=info.get("dist_m"),
                  cam=None if p is None else np.round(p, 4).tolist(), tip=np.round(tip, 4).tolist(),
                  q_deg=np.round(np.degrees(q), 2).tolist())
            if pix is None:
                print(f"  {label}: not found - {info['why']}")
            else:
                if p is not None:
                    depth_pts[len(pixels)] = p
                pixels.append(pix)
                grip_T.append(arm.kin.fk(q, "gripper_link"))
                print(f"  {label}: found at pixel ({pix[0]:.0f},{pix[1]:.0f})"
                      + (f", depth {info['dist_m'] * 100:.0f} cm" if p is not None else ""))
            if tried == 8 and not seen_any:
                print("\nThe OAK-D saw no gripper movement at all in 8 poses: the arm is not in its view here. "
                      "Stopping. Look at the recording (iro review) to see what the OAK-D sees.")
                r.note("stopped early: no movement seen in 8 poses")
                break
        arm.gripper(False)
        arm.move_tip([0.20, 0.0, table + 0.15])
    oak.close()
    r.note(f"{len(pixels)} of {tried} poses found the gripper ({len(depth_pts)} with depth)")
    np.savez(r.dir / "sightings.npz", pixels=np.array(pixels), grip_T=np.array(grip_T), K=oak.K,
             floor_n=fl[0], floor_d=fl[1], table=table,
             depth_idx=np.array(list(depth_pts), int), depth_pts=np.array(list(depth_pts.values())).reshape(-1, 3))
    if test:
        print(f"\nTest: the gripper was found at {len(pixels)} of {tried} poses "
              f"({len(depth_pts)} also with depth). Recording: {r.dir}")
        return 0 if len(pixels) >= 4 else 1
    if len(pixels) < 10:
        print(f"Only {len(pixels)} poses found the gripper (need 10+). Recording: {r.dir} (iro review)")
        return 1
    T, off, rms, per = solve_pixels(pixels, grip_T, oak.K, fl[0], fl[1], table, depth_pts)
    bad = [i for i, e in enumerate(per) if e > 3 * max(rms, 4.0)]
    if bad and len(pixels) - len(bad) >= 10:       # one bad sighting must not spoil it
        keep = [i for i in range(len(pixels)) if i not in bad]
        dp = {keep.index(i): v for i, v in depth_pts.items() if i in keep}
        T, off, rms, per = solve_pixels([pixels[i] for i in keep], [grip_T[i] for i in keep], oak.K, fl[0], fl[1], table, dp)
        r.note(f"dropped {len(bad)} outlier sightings: poses {bad}")
    out = (f"OAK-D position in the arm frame: {np.round(T[:3, 3] * 100, 1)} cm, looking "
           f"{np.degrees(np.arctan2(T[1, 2], T[0, 2])):+.0f} deg from straight ahead; picture fit {rms:.1f} px "
           f"({len(per)} poses); the jaw point sits {np.round(off * 100, 1)} cm from the gripper joint")
    print("\n" + out)
    r.note(out)
    if rms > 8:
        print("Not saved: the picture fit is worse than 8 px. Recording: " + str(r.dir))
        return 1
    CALIB.write_text(yaml.safe_dump({"T_base_cam": T.round(6).tolist(), "jaw_point": off.round(5).tolist(),
                                     "rms_px": round(rms, 2), "poses": len(per),
                                     "floor_cam": {"normal": fl[0].round(5).tolist(), "d": round(fl[1], 5)},
                                     "recording": str(r.dir),
                                     "date": time.strftime("%Y-%m-%d %H:%M"), "method": "markerless: floor + jaw seen in the picture"}))
    print(f"Saved {CALIB.relative_to(ROOT)}. Next: iro efad see")
    return 0


def cmd_check(quiet=False):
    """Has the OAK-D mount moved since calibration? Compares today's floor with the saved one."""
    from iro.oak import Oak
    if not CALIB.exists():
        print("Not calibrated yet: iro efad calibrate")
        return 1
    saved = yaml.safe_load(CALIB.read_text()).get("floor_cam")
    oak = Oak()
    _, depth = oak.read(5)
    oak.close()
    fl = floor_plane(oak, depth)
    if fl is None or saved is None:
        print("Cannot check: not enough floor in view" if fl is None else "Calibration has no floor record: re-run iro efad calibrate")
        return 1
    h0, t0, r0 = mount_from_floor(np.array(saved["normal"]), saved["d"])
    h, t, r = mount_from_floor(fl[0], fl[1])
    moved = abs(h - h0) > 0.01 or abs(t - t0) > 1.5 or abs(r - r0) > 1.5
    print(f"OAK-D now {h * 100:.1f} cm high, tilt {t:.1f} deg, roll {r:+.1f}  |  at calibration {h0 * 100:.1f} cm, "
          f"{t0:.1f} deg, {r0:+.1f}  ->  {'MOVED: run iro efad calibrate' if moved else 'OK, unchanged'}")
    return 2 if moved else 0


# --- seeing ----------------------------------------------------------------------

class Scene:
    """OAK-D + instant memory: named objects with 3D measurements in the arm frame."""

    def __init__(self):
        from iro.instant import Recognizer
        from iro.oak import Oak
        self.oak = Oak()
        self.rec = Recognizer()
        self.T = load_calib()

    def look(self, n=3):
        img, depth = self.oak.read(n)
        found = []
        for name, margin, _, box in self.rec.recognize(img):
            pts = self.oak.measure(depth, box)
            if pts is None or len(pts) < 30:
                continue
            obj = {"name": name, "margin": margin, "box": box, "pts_cam": pts}
            if self.T is not None:
                pb = (self.T[:3, :3] @ pts.T).T + self.T[:3, 3]
                obj.update(geometry(pb))
            found.append(obj)
        return img, depth, found

    def draw(self, img, found):
        vis = img.copy()
        for o in found:
            x1, y1, x2, y2 = map(int, o["box"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
            txt = o["name"]
            if "centre" in o:
                c = o["centre"] * 100
                txt += f" ({c[0]:.0f},{c[1]:+.0f}) top {o['top'] * 100:+.0f}cm"
            cv2.putText(vis, txt, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        return vis

    def close(self):
        self.oak.close()


def geometry(pb):
    """From an object's 3D points in the arm frame: centre, top height, long-axis angle, size."""
    top = float(np.percentile(pb[:, 2], 90))
    upper = pb[pb[:, 2] > top - 0.03]                      # the top face: its outline gives the shape
    xy = upper[:, :2] if len(upper) > 20 else pb[:, :2]
    centre = np.median(xy, 0)
    cov = np.cov((xy - centre).T)
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, 1]
    proj = (xy - centre) @ np.c_[major, vecs[:, 0]]
    length, width = (np.percentile(proj, 95, 0) - np.percentile(proj, 5, 0))
    return {"centre": np.r_[centre, top], "top": top, "yaw": float(np.arctan2(major[1], major[0])),
            "length": float(length), "width": float(width)}


def cmd_see():
    from iro.stream import feed
    scene = Scene()
    f = feed()
    print("EFAD see: OAK-D live view" + ("" if scene.T is not None else " (not calibrated: positions in camera frame)")
          + ". Ctrl+C to stop.")
    try:
        while True:
            img, _, found = scene.look(1)
            f.show(scene.draw(img, found), ", ".join(o["name"] for o in found) or "no remembered object in view")
    except KeyboardInterrupt:
        pass
    scene.close()
    return 0


# --- picking -------------------------------------------------------------------

def jaw_direction(kin, q):
    """Direction (arm frame, horizontal) in which the jaws close, for joint angles q."""
    j = kin.joints["moving_jaw_so101_v1_link"]
    Tg = kin.fk(q, "gripper_link")
    hinge = Tg[:3, :3] @ (j["origin"][:3, :3] @ j["axis"])
    tool = kin.fk(q)[:3, 2]
    d = np.cross(tool, hinge)
    d[2] = 0
    return d / (np.linalg.norm(d) or 1)


def best_roll(arm, xyz, yaw_major):
    """Wrist roll that makes the jaws close across the object's narrow side (perpendicular to its
    long axis). Tries rolls around jaws-level and keeps the reachable one that matches best."""
    want = np.array([-np.sin(yaw_major), np.cos(yaw_major)])      # across the long axis
    level = arm.level_roll()
    best = None
    for dr in np.radians(np.arange(-90, 91, 15)):
        q, err = arm.kin.ik_down(xyz, np.r_[arm.joints()[:4], level + dr])
        if err > 0.004 or not (arm.kin.limits[4][0] <= level + dr <= arm.kin.limits[4][1]):
            continue
        score = abs(np.dot(jaw_direction(arm.kin, q)[:2], want))   # 1 = perfectly across
        if best is None or score > best[0] + 1e-3 or (abs(score - best[0]) < 1e-3 and abs(dr) < abs(best[1])):
            best = (score, dr)
    return (level + best[1], best[0]) if best else (level, 0.0)


def cmd_pick(name, dry_run=False, drop=False):
    from iro.motion import Arm, MotionError
    from iro.pick import pick_config
    from iro.stream import feed
    cfg = pick_config()
    if load_calib() is None:
        print("EFAD is not calibrated yet: iro efad calibrate (no marker needed)")
        return 1
    scene = Scene()
    f = feed()
    try:
        # 1. find it in 3D (several looks, keep the steadiest)
        seen = []
        for _ in range(5):
            img, _, found = scene.look(3)
            f.show(scene.draw(img, found), f"EFAD looking for {name}")
            seen += [o for o in found if o["name"] == name]
        if not seen:
            cv2.imwrite(str(SNAP), img)
            print(f"EFAD cannot see {name} (OAK-D view: {SNAP}). Is it remembered (iro object show {name})? In view?")
            return 1
        c = np.median([o["centre"] for o in seen], 0)
        yaw = float(np.median([o["yaw"] for o in seen]))
        width = float(np.median([o["width"] for o in seen]))
        length = float(np.median([o["length"] for o in seen]))
        print(f"1. {name} at x {c[0] * 100:.1f}, y {c[1] * 100:+.1f} cm, top {c[2] * 100:+.1f} cm; "
              f"{length * 100:.1f} x {width * 100:.1f} cm, long axis {np.degrees(yaw):.0f} deg")
        if width > cfg.get("gripper_max_open_m", 0.07):
            print(f"   too wide for the gripper ({width * 100:.1f} cm across its narrow side)")
            return 1
        # 2. grasp plan: top-down, jaws across the narrow side, grip in the upper part
        height = c[2] - cfg["table_z"]
        grasp_z = max(cfg["table_z"] + 0.012, c[2] - min(0.025, height / 2))
        pre_z = c[2] + 0.08
        with Arm(dry_run=dry_run) as arm:
            roll, match = best_roll(arm, [c[0], c[1], pre_z], yaw)
            print(f"2. plan: wrist roll {np.degrees(roll - arm.level_roll()):+.0f} deg from level "
                  f"(jaws {match * 100:.0f}% across), grip at {grasp_z * 100:+.1f} cm")
            if dry_run:
                print("   [dry run] no motion")
                return 0
            arm.gripper(open=True)
            arm.move_tip([c[0], c[1], pre_z], roll=roll)
            # 3. wrist camera: is the right thing below us?
            try:
                from iro.objects import Detector, grab, open_camera
                cam = open_camera()
                det = Detector()
                wf = grab(cam)
                wfound, wvis = det.detect(wf, draw=True)
                cam.release()
                names = [n for n, *_ in wfound]
                f.show(wvis, f"wrist camera: {', '.join(names) or 'nothing recognised'}")
                print(f"3. wrist camera sees: {', '.join(names) or 'nothing it remembers'}")
            except Exception as e:     # noqa: BLE001 - the confirmation is a bonus, not a must
                print(f"3. wrist camera check skipped ({str(e)[:60]})")
            # 4. grasp
            arm.move_tip([c[0], c[1], grasp_z], duration=1.5, roll=roll)
            arm.gripper(open=False, wait=1.2)
            held = arm.gripper_pct()
            arm.move_tip([c[0], c[1], cfg["lift_z"]], duration=1.5, roll=roll)
            if held < cfg["gripper_closed_pct"] + 3:
                print(f"4. missed: the jaws closed fully ({held:.0f}%)")
                return 1
            print(f"4. holding {name} (jaws at {held:.0f}%)")
            if drop:
                arm.move_tip([c[0], c[1], grasp_z + 0.01], duration=1.5, roll=roll)
                arm.gripper(open=True)
                arm.move_tip([c[0], c[1], pre_z], roll=roll)
                print("   put it back")
        return 0
    except MotionError as e:
        print(f"STOPPED: {e}")
        return 2
    finally:
        scene.close()


def main():
    a = sys.argv[1:] or ["help"]
    if a[0] == "see":
        return cmd_see()
    if a[0] == "wrist":
        from iro.vision import main as wrist
        return wrist()
    if a[0] == "show" and len(a) >= 2:
        from iro.instant import cmd_show
        return cmd_show(a[1], oak=True)
    if a[0] == "check":
        return cmd_check()
    if a[0] == "calibrate":
        return cmd_calibrate(test="--test" in a)
    if a[0] == "pick" and len(a) >= 2:
        return cmd_pick(a[1], dry_run="--dry-run" in a, drop="--drop" in a)
    print(__doc__)
    return 0 if a[0] == "help" else 1


if __name__ == "__main__":
    sys.exit(main())
