"""EFAD - Embedded Foundational Augmented Detection: the robot's perception, and picking with it.

    iro efad see            OAK-D live view: names + 3D position of remembered objects
    iro efad wrist          wrist-camera live view (was `iro vision`)
    iro efad show NAME      teach an object to the OAK-D (colour) in ~15 s, no training
    iro efad marker         make the calibration marker to print
    iro efad calibrate      find where the OAK-D sits relative to the arm (marker on the gripper)
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
MARKER_ID, MARKER_MM = 0, 40
SNAP = ROOT.parent / "efad-last.jpg"


def aruco():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    return d, cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())


def load_calib():
    if not CALIB.exists():
        return None
    c = yaml.safe_load(CALIB.read_text())
    return np.array(c["T_base_cam"])


def to_base(T, p):
    return T[:3, :3] @ np.asarray(p) + T[:3, 3]


# --- marker + calibration ------------------------------------------------------

def cmd_marker():
    d, _ = aruco()
    px = 600
    img = np.full((px + 200, px + 200), 255, np.uint8)
    img[100:100 + px, 100:100 + px] = cv2.aruco.generateImageMarker(d, MARKER_ID, px)
    cv2.putText(img, f"EFAD marker: print so the BLACK square is {MARKER_MM} mm wide", (20, px + 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    out = ROOT.parent / "efad-marker.png"
    cv2.imwrite(str(out), img)
    print(f"Saved {out}. Print it, check the black square measures {MARKER_MM} mm, and tape it flat on "
          "the side of the gripper that faces the OAK-D (keep the white border).")
    return 0


def find_marker(img, oak):
    """Marker centre in camera coordinates (metres) via its known size, or None."""
    _, det = aruco()
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if ids is None or MARKER_ID not in ids.flatten():
        return None, None
    c = corners[list(ids.flatten()).index(MARKER_ID)].reshape(4, 2)
    s = MARKER_MM / 1000 / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, c.astype(np.float32), oak.K, oak.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return (tvec.flatten() if ok else None), c


def solve_calibration(cam_pts, grip_T):
    """Camera->base transform AND the marker's offset on the gripper, from pairs of
    (marker centre seen by the camera, gripper pose from the arm). Alternates Kabsch / least squares."""
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


def cmd_calibrate():
    from iro.motion import SEEDS, Arm
    from iro.oak import Oak
    from iro.pick import pick_config
    from iro.stream import feed
    cfg = pick_config()
    oak = Oak()
    f = feed()
    print("Tape the printed marker on the gripper side facing the OAK-D. The arm will move through "
          "about 16 poses in front of the robot; keep the area clear.")
    input("Ready? Enter. ")
    cam_pts, grip_T = [], []
    with Arm() as arm:
        roll0 = arm.level_roll()
        table = cfg["table_z"]
        poses = [(x, y, table + h, r) for h in (0.06, 0.14, 0.22) for x in (0.16, 0.24)
                 for y in (-0.08, 0.0, 0.08) for r in (0.0,)]
        poses += [(0.20, 0.0, table + 0.12, np.radians(r)) for r in (-40, 40)]   # some roll variety
        for i, (x, y, z, dr) in enumerate(poses):
            try:
                arm.move_tip([x, y, z], roll=roll0 + dr)
            except Exception as e:     # noqa: BLE001 - unreachable pose: skip it
                print(f"  pose {i + 1}: skipped ({str(e)[:60]})")
                continue
            time.sleep(0.6)
            img, _ = oak.read(2)
            p, c = find_marker(img, oak)
            vis = img.copy()
            if c is not None:
                cv2.polylines(vis, [c.astype(int)], True, (0, 255, 0), 2)
            f.show(vis, f"calibration pose {i + 1}/{len(poses)}: {'marker seen' if p is not None else 'marker NOT seen'}")
            if p is None:
                print(f"  pose {i + 1}: marker not seen")
                continue
            cam_pts.append(p)
            grip_T.append(arm.kin.fk(arm.joints(), "gripper_link"))
            print(f"  pose {i + 1}: marker at {np.round(p * 100, 1)} cm")
        arm.move_tip([0.20, 0.0, table + 0.15])
    oak.close()
    if len(cam_pts) < 6:
        print(f"Only {len(cam_pts)} poses saw the marker (need 6+). Turn the marker toward the OAK-D, check "
              "the light, and run it again.")
        return 1
    T, off, rms, res = solve_calibration(cam_pts, grip_T)
    bad = [i for i, r in enumerate(res) if r > 3 * max(rms, 0.003)]
    if bad and len(cam_pts) - len(bad) >= 6:       # one bad detection must not spoil it
        keep = [i for i in range(len(cam_pts)) if i not in bad]
        T, off, rms, res = solve_calibration([cam_pts[i] for i in keep], [grip_T[i] for i in keep])
    print(f"\nOAK-D position in the arm frame: {np.round(T[:3, 3] * 100, 1)} cm; fit error {rms * 1000:.1f} mm "
          f"({len(res)} poses); marker sits {np.round(off * 100, 1)} cm from the gripper joint")
    if rms > 0.012:
        print("Fit worse than 12 mm: not saved. Check the marker is flat and its printed size is exactly "
              f"{MARKER_MM} mm, then run again.")
        return 1
    CALIB.write_text(yaml.safe_dump({"T_base_cam": T.round(6).tolist(), "marker_offset": off.round(5).tolist(),
                                     "rms_m": round(rms, 5), "poses": len(res),
                                     "date": time.strftime("%Y-%m-%d %H:%M")}))
    print(f"Saved {CALIB.relative_to(ROOT)}. Next: iro efad see")
    return 0


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
        print("EFAD is not calibrated yet: iro efad marker, then iro efad calibrate")
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
    if a[0] == "marker":
        return cmd_marker()
    if a[0] == "calibrate":
        return cmd_calibrate()
    if a[0] == "pick" and len(a) >= 2:
        return cmd_pick(a[1], dry_run="--dry-run" in a, drop="--drop" in a)
    print(__doc__)
    return 0 if a[0] == "help" else 1


if __name__ == "__main__":
    sys.exit(main())
