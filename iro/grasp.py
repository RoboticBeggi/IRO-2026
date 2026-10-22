"""iro pick NAME: pick up a taught object from the floor with the wrist camera (no markers, no
camera calibration).

    iro pick teach NAME      once per object: the arm lowers its open jaws to the floor, you slide
                             the object between them; it closes (contact detection), lets go, rises
                             straight up and remembers where the object then is in the picture
    iro pick NAME            find it from above, centre it, turn the jaws across it, open just
                             wide enough, descend, close until contact, lift   (--drop: put it back)

Needs: `iro track teach NAME` (what it looks like) and `iro measure gripper` (jaw opening curve).
How it works (calibration-free visual servoing, as `iro pick`):
  look pose: gripper straight down LOOK_H above the floor; the wrist camera sees the floor below
  grasp pixel: where the object appears when it is exactly between the jaws (taught), plus the
    direction the jaws close in the picture and the mm per pixel on the floor
  centring: two small test moves measure how the picture moves with the arm (image Jacobian), then
    the arm steps sideways until the object sits on the grasp pixel
  turning: the object's outline (MobileSAM) gives its narrow direction; the wrist roll turns the
    jaws across it (the sign is checked by looking again, and remembered)
  grasping: open to the object's width + 15 mm (iro measure gripper), straight down, close until
    FIRST CONTACT (iro measure close_on), check the width is plausible, lift, check it is still held
Every run is recorded (iro review): a picture per step, the grip log, the decisions.
"""

import sys
import time

import cv2
import numpy as np
import yaml

LOOK_H = 0.15              # m above the floor for looking down
GRASP_H = 0.012            # m above the floor where the fingertips close (half a lying tape)
MARGIN_MM = 15             # open this much wider than the object
SERVO_TOL_PX = 6
SEARCH = [(0.20, 0.0), (0.24, 0.0), (0.20, 0.07), (0.20, -0.07), (0.16, 0.0), (0.24, 0.07), (0.24, -0.07)]


def grasp_file(name):
    from iro.objects import OBJ_DIR
    return OBJ_DIR / name / "grasp.yaml"


def wrap90(a):
    """Angle difference of two undirected lines, in (-90, 90] degrees."""
    return (a + 90) % 180 - 90


class DownEye:
    """Wrist camera + the object's patch matcher + SAM outlines, all recorded."""

    def __init__(self, name, r):
        from iro.lockon import PatchMatcher
        from iro.objects import Refiner, open_camera
        self.name, self.r = name, r
        self.pm = PatchMatcher()
        if not self.pm.load(name):
            raise SystemExit(f"'{name}' is not taught yet: iro track teach {name}")
        self.cam = open_camera()
        self.sam = Refiner()
        self.min_peak = None                       # set from the taught top view when picking

    def frame(self):
        from iro.lockon import fresh
        return fresh(self.cam)

    def find(self, near=None, radius=None, tries=3):
        """(frame, box, centre, peak) of the object, or (frame, None, None, 0)."""
        for _ in range(tries):
            f = self.frame()
            got = self.pm.locate(f, near=near, radius=radius, min_peak=self.min_peak)
            if got is not None:
                box, _, peak = got
                return f, box, np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]), peak
        return f, None, None, 0.0

    def outline(self, f, box):
        """Object mask, narrow-direction angle (deg, picture) and its extent (px) along `axis_deg`."""
        got = self.sam.refine(f, box)
        if got is None:
            return None
        _, mask = got
        ys, xs = np.nonzero(mask)
        pts = np.c_[xs, ys].astype(float)
        c = pts.mean(0)
        vals, vecs = np.linalg.eigh(np.cov((pts - c).T))
        minor = vecs[:, 0]
        return mask, float(np.degrees(np.arctan2(minor[1], minor[0]))), pts

    @staticmethod
    def extent(pts, axis_deg):
        a = np.radians(axis_deg)
        proj = pts @ np.array([np.cos(a), np.sin(a)])
        return float(np.percentile(proj, 99) - np.percentile(proj, 1))

    def show(self, f, text, box=None, target=None, tag=None):
        v = f.copy()
        if box is not None:
            cv2.rectangle(v, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        if target is not None:
            cv2.drawMarker(v, (int(target[0]), int(target[1])), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
        self.cam.flash(v, text, 0.8)
        self.r.video("pick", v, [text])
        if tag:
            self.r.snap(tag, v)
        return v

    def close(self):
        self.cam.release()


def look(arm, xy, roll, h=LOOK_H, duration=None):
    from iro.pick import pick_config
    table = pick_config()["table_z"]
    return arm.move_tip([xy[0], xy[1], table + h], roll=roll, duration=duration)


# ------------------------------------------------------------------ teach

def cmd_teach(name):
    from iro.measure import GRIPPER, close_on, opening_mm, set_gripper
    from iro.motion import Arm
    from iro.record import run
    if not GRIPPER.exists():
        print("Measure the gripper first: iro measure gripper")
        return 1
    g = yaml.safe_load(GRIPPER.read_text())
    open_pct = g["max_open_pct"]
    r = run(f"pick-teach-{name}")
    eye = DownEye(name, r)
    xy = np.array([0.20, 0.0])
    try:
        with Arm() as arm:
            roll = arm.level_roll()
            set_gripper(arm, open_pct)
            look(arm, xy, roll)
            look(arm, xy, roll, h=GRASP_H, duration=2.5)
            while True:
                input(f"Slide the {name} between the open jaws ({opening_mm(open_pct):.0f} mm), centred, lying the way "
                      "you want it picked. Hands away, then Enter. ")
                grip = close_on(arm, open_pct, r)
                if grip is not None:
                    break
                set_gripper(arm, open_pct)
                print("  The jaws closed on nothing. Put it between the fingertips and try again.")
            print(f"  Felt it: {grip['width_mm']:.1f} mm across the jaws ({grip['feel']}).")
            set_gripper(arm, open_pct)                      # let go WITHOUT moving it
            time.sleep(0.4)
            look(arm, xy, roll, duration=2.5)               # straight up
            time.sleep(0.5)
            f, box, c, peak = eye.find()
            if box is None:
                eye.show(f, f"cannot see the {name} from above", tag="teach_notseen")
                print(f"  From above the {name} is not recognised (match too weak). Draw a box around it on the "
                      "live page instead.")
                box = eye.cam.feed.ask_box(f.shape, timeout=120)
                if box is None:
                    return 1
                box = np.array(box, float)
                c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            out = eye.outline(f, box)
            if out is None:
                print("  Could not outline it (SAM). Try again with less clutter around it.")
                return 1
            mask, minor_deg, pts = out
            # the jaws closed across the object: its narrow direction NOW is the jaws' direction
            jaw_deg = minor_deg
            px = eye.extent(pts, jaw_deg)
            mm_per_px = grip["width_mm"] / max(px, 1)
            # learn its top view (it was taught from the side) AND this floor seen from above as
            # "not it" (22 Oct: a plain floor patch matched at 0.17 from 10 cm up)
            eye.pm.add_view(f, mask)
            eye.pm.learn_scene(f, box, mask)
            eye.pm.save(name)
            s_map, _ = eye.pm.score_map(f)
            top_peak = float(s_map[eye.pm.small(mask > 0)].max())
            eye.show(f, f"grasp pixel ({c[0]:.0f},{c[1]:.0f}); jaws close at {jaw_deg:.0f} deg; {mm_per_px:.2f} mm/px",
                     box, c, "teach_result")
    finally:
        eye.close()
    data = {"grasp_pixel": [round(float(c[0]), 1), round(float(c[1]), 1)], "jaw_axis_deg": round(jaw_deg, 1),
            "mm_per_px": round(mm_per_px, 4), "look_h_m": LOOK_H, "grasp_h_m": GRASP_H,
            "taught_width_mm": round(grip["width_mm"], 1), "roll_sign": 1, "top_peak": round(top_peak, 3),
            "recording": str(r.dir), "date": time.strftime("%Y-%m-%d %H:%M")}
    grasp_file(name).write_text(yaml.safe_dump(data))
    r.note(f"taught: {data}")
    print(f"Taught. The {name} sits at pixel {data['grasp_pixel']} when it is between the jaws "
          f"(seen from {LOOK_H * 100:.0f} cm up). Next: iro pick {name}")
    return 0


# ------------------------------------------------------------------ pick

def jacobian(arm, eye, xy, roll, c0, step=0.02):
    """Pixels per metre of tip motion (2x2), from two small test moves."""
    J = np.zeros((2, 2))
    for i in range(2):
        p = xy.copy()
        p[i] += step
        look(arm, p, roll)
        _, box, c, _ = eye.find(near=c0, radius=160)
        if box is None:
            raise RuntimeError("lost the object during a test move")
        J[:, i] = (c - c0) / step
    look(arm, xy, roll)
    if abs(np.linalg.det(J)) < 1e3:
        raise RuntimeError(f"the picture barely moves with the arm (J={np.round(J)})")
    return J


def centre(arm, eye, xy, roll, J, target, c, tries=15):
    """Step sideways until the object sits on the target pixel. Returns (xy, frame, box, c)."""
    Jinv = np.linalg.inv(J)
    for k in range(tries):
        err = c - target
        if np.linalg.norm(err) < SERVO_TOL_PX:
            break
        xy = xy + np.clip(-0.7 * Jinv @ err, -0.03, 0.03)
        look(arm, xy, roll)
        f, box, c, _ = eye.find(near=c, radius=150)
        if box is None:
            raise RuntimeError("lost the object while centring")
        eye.r.log("centre", step=k, err_px=float(np.linalg.norm(err)), x=float(xy[0]), y=float(xy[1]))
    else:
        raise RuntimeError(f"could not centre it in {tries} steps")
    return xy, f, box, c


def cmd_pick(name, drop=False):
    from iro.measure import close_on, max_open_m, pct_for, set_gripper
    from iro.motion import Arm, MotionError
    from iro.record import run
    gf = grasp_file(name)
    if not gf.exists():
        print(f"Teach the grasp first: iro pick teach {name}")
        return 1
    G = yaml.safe_load(gf.read_text())
    target = np.array(G["grasp_pixel"])
    r = run(f"pick-{name}")
    eye = DownEye(name, r)
    if G.get("top_peak"):
        eye.min_peak = 0.5 * G["top_peak"]          # seen from above it must match like it did then
    try:
        with Arm() as arm:
            roll = arm.level_roll()
            set_gripper(arm, 0)
            # 1. find it from above
            for xy in SEARCH:
                xy = np.array(xy)
                look(arm, xy, roll)
                f, box, c, peak = eye.find()
                if box is not None:
                    break
            else:
                eye.show(f, f"{name} not found from above", tag="notfound")
                print(f"1. The {name} is not under the search area (20 +/-7 cm in front, 16-24 cm out).")
                return 1
            eye.show(f, f"1. found {name} (match {peak:.2f})", box, target, "1_found")
            print(f"1. found it at pixel {np.round(c)} (match {peak:.2f})")
            # 2. centre it on the grasp pixel
            J = jacobian(arm, eye, xy, roll, c)
            xy, f, box, c = centre(arm, eye, xy, roll, J, target, c)
            eye.show(f, "2. centred", box, target, "2_centred")
            print(f"2. centred above it at ({xy[0] * 100:.1f}, {xy[1] * 100:+.1f}) cm")
            # 3. turn the jaws across its narrow side
            out = eye.outline(f, box)
            if out is not None:
                _, minor, pts = out
                d = wrap90(minor - G["jaw_axis_deg"])
                if abs(d) > 8:
                    sign = G.get("roll_sign", 1)
                    roll2 = roll + sign * np.radians(d)
                    look(arm, xy, roll2)
                    f, box, c, _ = eye.find(near=c, radius=200)
                    out2 = eye.outline(f, box) if box is not None else None
                    d2 = wrap90(out2[1] - G["jaw_axis_deg"]) if out2 else d
                    if abs(d2) > abs(d) * 0.7:              # turned the wrong way: flip, remember
                        sign = -sign
                        roll2 = roll + sign * np.radians(d)
                        look(arm, xy, roll2)
                        G["roll_sign"] = sign
                        gf.write_text(yaml.safe_dump(G))
                        r.note(f"wrist roll sign flipped to {sign}")
                        f, box, c, _ = eye.find(near=c, radius=200)
                    roll = roll2
                    if box is None:
                        raise RuntimeError("lost the object while turning the jaws")
                    J = jacobian(arm, eye, xy, roll, c)        # the picture is turned now
                    xy, f, box, c = centre(arm, eye, xy, roll, J, target, c)
                    out = eye.outline(f, box)
                    print(f"3. turned the jaws {d:+.0f} deg across it")
                else:
                    print("3. jaws already across it")
            # 4. how wide to open
            width = None
            if out is not None:
                width = eye.extent(out[2], G["jaw_axis_deg"]) * G["mm_per_px"]
            want = min(max_open_m() * 1000, (width or G["taught_width_mm"]) + MARGIN_MM)
            if width and width + 5 > max_open_m() * 1000:
                print(f"4. it is {width:.0f} mm across the jaws: too wide for the gripper ({max_open_m() * 1000:.0f} mm)")
                return 1
            open_pct = pct_for(want)
            set_gripper(arm, open_pct)
            eye.show(f, f"4. opening {want:.0f} mm for {width or 0:.0f} mm", box, target, "4_open")
            print(f"4. it is about {width or G['taught_width_mm']:.0f} mm across: opening {want:.0f} mm")
            # 5. straight down, close until contact
            look(arm, xy, roll, h=G["grasp_h_m"], duration=2.5)
            grip = close_on(arm, open_pct, r, hold=True)
            if grip is None:
                set_gripper(arm, open_pct)
                look(arm, xy, roll, duration=2.0)
                f2, *_ = eye.find()
                eye.show(f2, "5. MISSED: closed on nothing", tag="5_missed")
                print("5. Missed: the jaws closed on nothing. (Recording shows where it was.)")
                return 1
            expect = width or G["taught_width_mm"]
            print(f"5. gripped: {grip['width_mm']:.0f} mm between the fingertips (expected ~{expect:.0f}), {grip['feel']}")
            if abs(grip["width_mm"] - expect) > 20:
                print("   (that width does not match: it may be holding it by a corner)")
            # 6. lift and check it is still held
            look(arm, xy, roll, h=0.10, duration=2.0)
            time.sleep(0.3)
            from iro.measure import opening_mm
            c_ = arm.cal["gripper"]
            now = 100 * (arm.bus.read("Present_Position", "gripper", normalize=False) - c_["range_min"]) / (
                c_["range_max"] - c_["range_min"])
            if opening_mm(now) < grip["width_mm"] - 10:
                print("6. Dropped it while lifting (the jaws closed further).")
                r.note("dropped while lifting")
                return 1
            f, *_ = eye.find()
            eye.show(f, f"6. holding the {name}", tag="6_lifted")
            print(f"6. Holding the {name}, 10 cm up.")
            r.note(f"picked: width {grip['width_mm']:.1f} mm at ({xy[0]:.3f}, {xy[1]:.3f}), roll {np.degrees(roll):.0f}")
            if drop:
                look(arm, xy, roll, h=G["grasp_h_m"] + 0.005, duration=2.0)
                set_gripper(arm, open_pct)
                look(arm, xy, roll, duration=2.0)
                print("   Put it back down.")
    except (RuntimeError, MotionError) as e:
        print(f"STOPPED: {e}. The arm holds its pose.")
        r.note(f"stopped: {e}")
        return 1
    finally:
        eye.close()
    return 0


def main():
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == "teach":
        return cmd_teach(a[1])
    if a and not a[0].startswith("-"):
        return cmd_pick(a[0], drop="--drop" in a)
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
