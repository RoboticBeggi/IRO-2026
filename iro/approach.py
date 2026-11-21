"""Drive the mecanum base up to an object until the arm can pick it, then pick it.

    iro approach NAME [--pick] [--show]   find NAME (OAK-D, by name, any distance it can see),
                                          drive until it sits in the arm's sweet spot, then pick it
                                          (--pick) and show it around (--show)
    iro approach range NAME               how well the OAK-D sees NAME with distance: the robot backs
                                          away from it step by step and measures (needs free floor
                                          behind; the lidar checks and refuses otherwise)

How it works ("stop and go", the standard way for mobile manipulators: the base moves, stops, the
cameras look again, never grasping while the base moves):
  1. FAR EYE: the OAK-D colour picture -> the instant recogniser (YOLOE things + DINOv2 fingerprints,
     iro object show NAME) finds NAME by name. Where it is: from depth (median of the box's depth
     points) or, when depth is thin, from where the box's bottom edge meets the floor (camera ray
     through that pixel down to the floor plane). Both into the arm's frame (EFAD calibration,
     corrected by what earlier picks learnt: iro.oakguide.Correction).
  2. SEARCH: not in view -> turn on the spot 40 deg at a time and look again, up to a full turn.
  3. DRIVE: the object should end up at SWEET = 25 cm straight ahead of the arm base (straight-down
     reach, the OAK-D scout's best range). Each step: turn so it is ahead, and drive at most 40 cm
     toward the spot (lidar-steered, iro.base.move_by_lidar), then look again. Closer than 60 cm:
     the steps shrink and the robot no longer turns, only slides (mecanum).
  4. SAFETY: the lidar watches the direction of travel (people, furniture: it sees at its height,
     not the low object itself); speeds <= 0.15 m/s; the firmware watchdog stops the wheels if this
     program stops.
  5. HAND-OVER: within 4 cm of the spot -> iro pick NAME (OAK-D scout + wrist camera).
"""

import math
import sys
import time

import cv2
import numpy as np
import yaml

SWEET = np.array([0.25, 0.0])      # m, arm frame: where the object should end up
CLOSE_ENOUGH = 0.04                # m from SWEET: hand over to the pick
MAX_STEP = 0.40                    # m per drive step
STOP_AHEAD = 0.25                  # m: lidar return this close in the travel direction -> stop


class FarEye:
    """Find a named object with the OAK-D at any distance it can see. One OAK-D session."""

    def __init__(self, oak=None):
        from iro.efad import load_calib
        from iro.instant import Recognizer
        from iro.oakguide import Correction
        from iro.pick import pick_config
        if oak is None:
            from iro.oak import Oak
            oak = Oak()
        self.oak, self.rec, self.T = oak, Recognizer(), load_calib()
        self.corr, self.table = Correction(), pick_config()["table_z"]
        self.last, self.rejected, self._sig = None, [], {}

    def floor_point(self, u, v):
        """Where the camera ray through pixel (u, v) meets the floor (arm frame xy), or None."""
        K = self.oak.K
        ray = self.T[:3, :3] @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
        o = self.T[:3, 3]
        if ray[2] >= -1e-6:
            return None                            # at or above the horizon
        s = (self.table - o[2]) / ray[2]
        return (o + s * ray)[:2]

    def signature(self, name):
        """objects/NAME/colour.yaml (hue range, min saturation/brightness, min share of the box), or None."""
        if name not in self._sig:
            from iro.objects import OBJ_DIR
            f = OBJ_DIR / name / "colour.yaml"
            self._sig[name] = yaml.safe_load(f.read_text()) if f.exists() else None
        return self._sig[name]

    @staticmethod
    def colour_mask(img, sig):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return ((hsv[:, :, 0] >= sig["hue"][0]) & (hsv[:, :, 0] <= sig["hue"][1]) &
                (hsv[:, :, 1] >= sig["s_min"]) & (hsv[:, :, 2] >= sig["v_min"]))

    def look(self, name, n=1, min_margin=0.0):
        """All sightings of NAME in one look: [{xy, dist, margin, box, how, colour, size}], best first.
        Candidates: what the recogniser names NAME (margin >= min_margin) + blobs of NAME's colour.
        Kept only if (a) enough of the box has NAME's colour, (b) its size from depth fits NAME's
        sizes (+3 cm). The rejected ones are in self.rejected (why) for the recording."""
        from iro.oakguide import taught_sizes
        img, depth = self.oak.read(n)
        sig, sizes = self.signature(name), taught_sizes(name)
        biggest = max(sizes) if sizes else 0.30
        mask = self.colour_mask(img, sig) if sig else None
        cands = [(float(m), b, "named") for nm, m, _, b in self.rec.recognize(img) if nm == name and m >= min_margin]
        if mask is not None:                       # colour blobs: finds it even when the recogniser does not
            k, lab, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                                                                              np.ones((3, 3), np.uint8)))
            for i in range(1, k):
                x, y, w, h, a = st[i]
                if a < 30:
                    continue
                b = [float(x), float(y), float(x + w), float(y + h)]
                if not any(_iou(b, c[1]) > 0.3 for c in cands):
                    cands.append((0.0, b, "colour"))
        out, self.rejected = [], []
        for margin, box, found_by in cands:
            x1, y1, x2, y2 = box
            frac = None
            if mask is not None:
                w, h = x2 - x1, y2 - y1
                inner = mask[int(y1 + 0.25 * h):int(y2 - 0.25 * h) + 1, int(x1 + 0.25 * w):int(x2 - 0.25 * w) + 1]
                frac = float(inner.mean()) if inner.size else 0.0
                if frac < sig["frac_min"]:
                    self.rejected.append((box, f"colour {frac * 100:.0f}%"))
                    continue
            pts = self.oak.measure(depth, box)
            dxy, size = None, None
            if pts is not None and len(pts) >= 30:
                pb = (self.T[:3, :3] @ pts.T).T + self.T[:3, 3]
                dxy = np.median(pb[:, :2], 0)
                height = float(np.percentile(pb[:, 2], 95) - self.table)
                v = dxy / (np.linalg.norm(dxy) or 1)
                lat = pb[:, :2] @ np.array([-v[1], v[0]])
                across = float(np.percentile(lat, 95) - np.percentile(lat, 5))
                size = (across, height)
                if height > biggest + 0.03 or across > biggest + 0.03:
                    self.rejected.append((box, f"too big {across * 100:.0f}x{height * 100:.0f} cm"))
                    continue
            fxy = self.floor_point((x1 + x2) / 2, y2)
            xy = dxy if dxy is not None else fxy
            if xy is None:
                continue
            # physical size of the box from its distance: a thin stripe of the right colour (the
            # keyboard's yellow-green line, 21 Nov) is not the object
            if sizes:
                Z = float(np.median(pts[:, 2])) if pts is not None and len(pts) >= 30 else \
                    float(np.linalg.norm(np.r_[xy, self.table] - self.T[:3, 3]))
                wm, hm = (x2 - x1) * Z / self.oak.K[0, 0], (y2 - y1) * Z / self.oak.K[1, 1]
                small = 0.4 * min(sizes)
                if min(wm, hm) < small or max(wm, hm) < 0.4 * sorted(sizes)[1]:
                    self.rejected.append((box, f"too small {wm * 100:.1f}x{hm * 100:.1f} cm"))
                    continue
            xy = self.corr.point(xy)
            out.append({"xy": xy, "dist": float(np.linalg.norm(xy)), "margin": margin, "box": box,
                        "how": "depth" if dxy is not None else "floor", "found_by": found_by, "colour": frac,
                        "size": size, "depth_xy": None if dxy is None else self.corr.point(dxy),
                        "floor_xy": None if fxy is None else self.corr.point(fxy),
                        "depth_pts": 0 if pts is None else len(pts)})
        # best: named and coloured first, then by how much of it has the colour
        out.sort(key=lambda o: -(o["margin"] + (o["colour"] or 0) + (0.2 if o["found_by"] == "named" else 0)))
        self.last = (img, out)
        return out

    def draw(self, text=""):
        img, out = self.last
        vis = img.copy()
        for box, why in self.rejected:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(vis, why, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        for o in out:
            x1, y1, x2, y2 = map(int, o["box"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{o['dist'] * 100:.0f} cm ({o['how']}, {o['found_by']}) {o['margin']:+.2f}",
                        (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if text:
            cv2.putText(vis, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return vis

    def close(self):
        self.oak.close()


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def clearance(lp, heading, width=0.45, reach=1.5):
    """Free distance (m) from the robot along `heading` (rad, robot frame) in a corridor `width` wide,
    from the newest lidar scan. The lidar sees at its own height (people, furniture)."""
    a, r = lp.raw()
    ok = np.isfinite(r) & (r > 0.12)
    d = (a[ok] - heading + math.pi) % (2 * math.pi) - math.pi
    rr = r[ok]
    ahead, side = rr * np.cos(d), np.abs(rr * np.sin(d))
    m = (ahead > 0) & (side < width / 2) & (ahead < reach)
    return float(ahead[m].min()) if m.any() else float("inf")


# ------------------------------------------------------------------ the range experiment

def cmd_range(name, step=0.15, max_back=0.9):
    from iro.base import Base, LidarOdom, LidarPoints, move_by_lidar
    from iro.oakguide import Scout
    from iro.record import run
    r = run(f"approach-range-{name}")
    sc = Scout()
    o = sc.find(name)
    if o is None:
        sc.close()
        print(f"The OAK-D scout does not see the {name} in front (18-45 cm). Put it there first.")
        return 1
    P0 = o["centre"]
    eye = FarEye(sc.oak)
    lp = LidarPoints()
    home = LidarOdom(lp)                  # robot pose relative to the start, for the true distance
    step_odom = LidarOdom(lp)
    rows = []
    try:
        with Base() as b:
            bearing = math.atan2(P0[1], P0[0])
            if abs(bearing) > math.radians(8):
                move_by_lidar(b, step_odom, 0, 0, bearing)
                home.reset()
                o = sc.find(name)
                P0 = o["centre"] if o is not None else np.array([np.hypot(*P0), 0.0])
            print(f"start: {name} at {P0[0] * 100:.0f}, {P0[1] * 100:+.0f} cm (scout). Backing away {step * 100:.0f} cm at a time.")
            back = 0.0
            while True:
                x, y, th = home.pose()
                c, s_ = math.cos(th), math.sin(th)
                d = P0 - np.array([x, y])
                truth = np.array([c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1]])      # where it really is now
                looks = [eye.look(name) for _ in range(6)]
                hits = [lk[0] for lk in looks if lk]
                row = {"true_cm": float(np.linalg.norm(truth) * 100), "seen": len(hits) / len(looks),
                       "margin": float(np.median([h["margin"] for h in hits])) if hits else None}
                for how in ("depth_xy", "floor_xy"):
                    e = [np.linalg.norm(h[how] - truth) for h in hits if h[how] is not None]
                    row[how + "_err_cm"] = float(np.median(e) * 100) if e else None
                    row[how + "_n"] = len(e)
                row["box_px"] = float(np.median([h["box"][2] - h["box"][0] for h in hits])) if hits else None
                rows.append(row)
                r.log("range", **{k: (-1 if v is None else v) for k, v in row.items()})
                r.snap(f"range_{len(rows):02d}", eye.draw(f"true {row['true_cm']:.0f} cm, seen {row['seen'] * 100:.0f}%")
                       if eye.last else sc.read()[0])
                fmt = lambda v: "  -  " if v is None else f"{v:5.1f}"          # noqa: E731
                print(f"  {row['true_cm']:5.0f} cm: seen {row['seen'] * 100:3.0f}%  margin {fmt(row['margin'])}  "
                      f"error by depth {fmt(row['depth_xy_err_cm'])} cm ({row['depth_xy_n']})  "
                      f"by floor ray {fmt(row['floor_xy_err_cm'])} cm  box {fmt(row['box_px'])} px", flush=True)
                if back + step > max_back + 1e-6:
                    break
                free = clearance(lp, math.pi)
                if free < step + 0.45:
                    print(f"  stopped: only {free * 100:.0f} cm free behind (lidar)")
                    break
                move_by_lidar(b, step_odom, -step, 0.0, 0.0, check=lambda: clearance(lp, math.pi) < 0.40)
                back += step
            # drive back to where it started
            x, y, th = home.pose()
            print(f"  returning {math.hypot(x, y) * 100:.0f} cm to the start")
            move_by_lidar(b, step_odom, -(math.cos(th) * x + math.sin(th) * y), -(-math.sin(th) * x + math.cos(th) * y), -th)
    finally:
        lp.close()
        sc.close()
    return 0


# ------------------------------------------------------------------ approach

def approach(name, b, eye, lp, r, log=print, max_s=90):
    """Drive until NAME sits at SWEET in front of the arm. Returns True when there."""
    from iro.base import LidarOdom, move_by_lidar
    odom = LidarOdom(lp)
    t0 = time.time()
    turned = 0.0
    while time.time() - t0 < max_s:
        hits = [h for _ in range(2) for h in eye.look(name)]
        r.video("approach", eye.draw(f"approach {name}"))
        if not hits:
            if turned >= 2 * math.pi - 1e-3:
                log(f"   not found after a full turn")
                return False
            log("   not in view: turning 40 deg to look")
            move_by_lidar(b, odom, 0, 0, math.radians(40))
            turned += math.radians(40)
            continue
        turned = 0.0
        P = np.median([h["xy"] for h in hits], 0)
        e = P - SWEET
        dist = float(np.linalg.norm(P))
        log(f"   {name} at {P[0] * 100:.0f}, {P[1] * 100:+.0f} cm ({hits[0]['how']}, margin {hits[0]['margin']:+.2f}); "
            f"{np.linalg.norm(e) * 100:.0f} cm from the spot")
        r.note(f"approach: seen at ({P[0]:.3f}, {P[1]:.3f}) by {hits[0]['how']}, {np.linalg.norm(e) * 100:.0f} cm to go")
        if np.linalg.norm(e) < CLOSE_ENOUGH:
            return True
        bearing = math.atan2(P[1], P[0])
        if dist > 0.60 and abs(bearing) > math.radians(10):
            # far: first face it (keeps it in the middle of the picture), then drive
            move_by_lidar(b, odom, 0, 0, bearing)
            continue
        step = e * min(1.0, MAX_STEP / max(np.linalg.norm(e), 1e-6))
        if dist < 0.60:
            step = e * min(1.0, 0.15 / max(np.linalg.norm(e), 1e-6))      # near: small slides only
        heading = math.atan2(step[1], step[0])
        free = clearance(lp, heading)
        if free < STOP_AHEAD + np.linalg.norm(step):
            log(f"   something {free * 100:.0f} cm away in the way (lidar): stopping")
            return False
        move_by_lidar(b, odom, float(step[0]), float(step[1]), 0.0,
                      check=lambda: clearance(lp, heading) < STOP_AHEAD)
    log("   took too long")
    return False


def cmd_approach(name, pick=False, show=False):
    from iro.base import Base, LidarPoints
    from iro.record import run
    r = run(f"approach-{name}")
    eye = FarEye()
    lp = LidarPoints()
    ok = False
    try:
        with Base() as b:
            t0 = time.time()
            ok = approach(name, b, eye, lp, r)
            r.note(f"approach {'done' if ok else 'failed'} in {time.time() - t0:.1f} s")
            print(f"{'In position' if ok else 'Could not get there'} after {time.time() - t0:.0f} s.")
    finally:
        lp.close()
        eye.close()                    # the pick opens the OAK-D again as its scout
    if ok and pick:
        from iro.grasp import cmd_pick
        return cmd_pick(name, show=show)
    return 0 if ok else 1


def main():
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == "range":
        return cmd_range(a[1])
    if a and not a[0].startswith("-"):
        return cmd_approach(a[0], pick="--pick" in a, show="--show" in a)
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
