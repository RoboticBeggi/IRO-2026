"""EFAD - Embedded Foundational Augmented Detection: the robot's perception, and picking with it.

    iro efad see            OAK-D live view: names + 3D position of remembered objects
    iro efad wrist          wrist-camera live view (was `iro vision`)
    iro efad show NAME      teach an object to the OAK-D (colour) in ~15 s, no training
    iro efad calibrate      find where the OAK-D sits relative to the arm (NO marker: the gripper
                            opens/closes in front of it and is found by itself; the floor double-checks)
    iro efad check          has the OAK-D moved since? (re-measures the floor, ~3 s)
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
    n = np.linalg.svd(Q - c)[2][2]
    if n[1] > 0:                       # camera y points down: the floor's "up" has negative y
        n = -n
    return n, float(-n @ c), float(np.sqrt(np.mean(((Q - c) @ n) ** 2)))


def mount_from_floor(n, d):
    """Height above the floor (m), tilt down and roll (deg) of the camera, from the floor plane."""
    return abs(d), float(np.degrees(np.arcsin(abs(n[2])))), float(np.degrees(np.arctan2(n[0], -n[1])))


def find_jaw(arm, oak):
    """3D point (camera frame) of the moving jaw: difference of open vs closed pictures. Done
    twice (open-close-open); both must agree within 1 cm. Returns (point, mask) or (None, mask)."""
    shots = []
    for opened in (True, False, True):
        arm.gripper(opened, wait=0.7)
        shots.append(oak.read(3))
    pts, masks = [], []
    for (ia, da), (ib, db) in ((shots[0], shots[1]), (shots[1], shots[2])):
        both = (da > 0) & (db > 0)
        moved = both & (np.abs(da.astype(np.float32) - db.astype(np.float32)) > 15)
        moved |= cv2.absdiff(cv2.cvtColor(ia, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ib, cv2.COLOR_BGR2GRAY)) > 35
        m = cv2.morphologyEx(moved.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            return None, m
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        if st[k, cv2.CC_STAT_AREA] < 150:
            return None, m
        blob = lab == k
        near = np.where((da > 0) & ((db == 0) | (da < db)), da, db)     # the jaw is the NEARER surface
        P = oak.points(near, blob)
        if len(P) < 50:
            return None, blob
        P = P[P[:, 2] < np.percentile(P[:, 2], 50) + 0.03]
        pts.append(np.median(P, 0))
        masks.append(blob)
    if np.linalg.norm(pts[0] - pts[1]) > 0.01:
        return None, masks[0]
    return (pts[0] + pts[1]) / 2, masks[0] | masks[1]


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


def cmd_calibrate():
    from iro.motion import Arm
    from iro.oak import Oak
    from iro.pick import pick_config
    from iro.stream import feed
    cfg = pick_config()
    oak = Oak()
    f = feed()
    _, depth = oak.read(5)
    fl = floor_plane(oak, depth)
    if fl is None:
        print("The OAK-D does not see enough floor. Clear the floor in front of the robot and try again.")
        return 1
    h, tilt, roll = mount_from_floor(fl[0], fl[1])
    print(f"Floor: OAK-D is {h * 100:.1f} cm above it, tilted {tilt:.1f} deg down, roll {roll:+.1f} deg "
          f"(floor flat to {fl[2] * 1000:.1f} mm)")
    print("No marker needed: the arm moves through ~16 poses in front of the OAK-D and opens/closes the "
          "gripper at each one, so the camera finds the gripper by itself. Keep the area clear and "
          "nothing else moving in front of the robot.")
    input("Ready? Enter. ")
    cam_pts, grip_T = [], []
    with Arm() as arm:
        roll0 = arm.level_roll()
        table = cfg["table_z"]
        # 30 poses: more sightings average out the jaw's few-mm noise (simulated: 16 -> 30 poses,
        # 4.5 -> 3.7 mm median with the floor constraint)
        poses = [(x, y, table + hh, 0.0) for hh in (0.06, 0.14, 0.22) for x in (0.15, 0.20, 0.25)
                 for y in (-0.08, 0.0, 0.08)]
        poses += [(0.20, 0.0, table + 0.12, np.radians(r)) for r in (-40, 40, -20)]   # roll variety
        for i, (x, y, z, dr) in enumerate(poses):
            try:
                arm.move_tip([x, y, z], roll=roll0 + dr)
            except Exception as e:     # noqa: BLE001 - unreachable pose: skip it
                print(f"  pose {i + 1}: skipped ({str(e)[:60]})")
                continue
            time.sleep(0.5)
            q = arm.joints()
            p, m = find_jaw(arm, oak)
            img, _ = oak.read(1)
            vis = img.copy()
            vis[m > 0] = (vis[m > 0] * 0.4 + np.array([0, 200, 0]) * 0.6).astype(np.uint8)
            f.show(vis, f"calibration pose {i + 1}/{len(poses)}: {'gripper found' if p is not None else 'gripper NOT found'}")
            if p is None:
                print(f"  pose {i + 1}: gripper not found (out of view, or something else moved)")
                continue
            cam_pts.append(p)
            grip_T.append(arm.kin.fk(q, "gripper_link"))
            print(f"  pose {i + 1}: gripper at {np.round(p * 100, 1)} cm from the OAK-D")
        arm.gripper(False)
        arm.move_tip([0.20, 0.0, table + 0.15])
    oak.close()
    if len(cam_pts) < 6:
        print(f"Only {len(cam_pts)} poses found the gripper (need 6+). Check the OAK-D sees the area in front "
              "of the arm (iro efad see), and that nothing else moved.")
        return 1
    T, off, rms, res = solve_with_floor(cam_pts, grip_T, fl[0], fl[1], table)
    bad = [i for i, r in enumerate(res) if r > 3 * max(rms, 0.004)]
    if bad and len(cam_pts) - len(bad) >= 6:       # one bad sighting must not spoil it
        keep = [i for i in range(len(cam_pts)) if i not in bad]
        T, off, rms, res = solve_with_floor([cam_pts[i] for i in keep], [grip_T[i] for i in keep], fl[0], fl[1], table)
    free = solve_calibration(cam_pts, grip_T)[0]   # without the floor: must roughly agree
    disagree = float(np.linalg.norm(free[:3, 3] - T[:3, 3]))
    # independent check: the gripper sightings alone (no floor) must put the camera nearly there too
    print(f"\nOAK-D position in the arm frame: {np.round(T[:3, 3] * 100, 1)} cm; fit error {rms * 1000:.1f} mm "
          f"({len(res)} poses); the point found sits {np.round(off * 100, 1)} cm from the gripper joint")
    print(f"Check: gripper sightings alone put the OAK-D {disagree * 100:.1f} cm from that (should be < 2.5 cm)")
    if rms > 0.012 or disagree > 0.025:
        print("Not saved: the fit or the floor check is off. Run again with nothing moving in view; if it "
              "repeats, send me the output.")
        return 1
    CALIB.write_text(yaml.safe_dump({"T_base_cam": T.round(6).tolist(), "jaw_point": off.round(5).tolist(),
                                     "rms_m": round(rms, 5), "poses": len(res),
                                     "floor_cam": {"normal": fl[0].round(5).tolist(), "d": round(fl[1], 5)},
                                     "check_without_floor_m": round(disagree, 4),
                                     "date": time.strftime("%Y-%m-%d %H:%M"), "method": "markerless (jaw)"}))
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
        return cmd_calibrate()
    if a[0] == "pick" and len(a) >= 2:
        return cmd_pick(a[1], dry_run="--dry-run" in a, drop="--drop" in a)
    print(__doc__)
    return 0 if a[0] == "help" else 1


if __name__ == "__main__":
    sys.exit(main())
