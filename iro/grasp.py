"""iro pick NAME: pick up a taught object from the floor with the wrist camera (no markers, no
camera calibration).

    iro pick teach NAME      once per object: the arm lowers its open jaws to the floor, you slide
                             the object between them; it closes (contact detection), lets go, rises
                             straight up and remembers where the object then is in the picture
    iro pick NAME            the OAK-D finds it on the floor, the arm goes straight above it with
                             the jaws turned across it, the wrist camera fine-centres, down, close
                             until contact, lift   (--drop: put it back;  --to=X,Y: carry it to that
                             spot, cm from the arm base). No OAK-D: searches with the wrist camera.
    iro pick calibrate NAME  once (or after the wrist camera was moved): how the wrist picture moves
                             with the gripper, measured on NAME lying still (~20 s)
    iro pick scout           what the OAK-D sees on the floor now

Needs: `iro track teach NAME` (what it looks like) and `iro measure gripper` (jaw opening curve).
How it works (calibration-free visual servoing, as `iro pick`):
  look pose: gripper straight down LOOK_H above the floor; the wrist camera sees the floor AHEAD
    of the jaws (not between them), so it looks from a taught offset (in the gripper's frame)
  grasp pixel: where the object appears from that offset when the jaws are exactly over it
    (taught), plus the direction the jaws close in the picture and the mm per pixel
  centring: two small test moves measure how the picture moves with the arm (image Jacobian), then
    the arm steps sideways until the object sits on the grasp pixel
  turning: the object's outline (MobileSAM) gives its narrow direction; the wrist roll turns the
    jaws across it while the camera ORBITS around the object (it sits 12 cm off the roll axis, so
    turning in place swings the view away); the sign is checked by looking again, and remembered
  grasping: open fully (the last centimetres are blind), straight down (leaning outward when it is
    far: iro arm reach), close until FIRST CONTACT (iro measure close_on), check the width matches
    the taught one, lift, squeeze-test that it is still held. The OAK-D films it all from the side.
Every run is recorded (iro review): a picture per step, the grip log, the decisions.
"""

import sys
import time

import cv2
import numpy as np
import yaml

from iro import ROOT

LOOK_H = 0.15              # m above the floor for looking down
GRASP_H = 0.012            # m above the floor where the fingertips close (half a lying tape)
SERVO_TOL_PX = 8           # outline centre; the arm itself repeats to ~3 mm (~6 px)
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

    def still_frame(self, max_wait=0.7, tol=1.5):
        """A frame taken when the PICTURE has stopped moving: two frames in a row less than `tol` px
        apart (phase correlation). The joint encoders say "still" while the arm's plastic links and
        the camera on the wrist still wobble after a quick move: a picture taken then was 28 px off
        with all joints within 3 counts (15 Nov)."""
        from iro.lockon import gray_small, picture_shift
        a = self.frame()
        ga, t0 = gray_small(a), time.time()
        while time.time() - t0 < max_wait:
            b = self.cam.latest()
            gb = gray_small(b)
            if np.linalg.norm(picture_shift(ga, gb, b.shape[1])[0]) < tol:
                return b
            a, ga = b, gb
        return a

    def find(self, near=None, radius=None, tries=3, still=False):
        """(frame, box, centre, peak) of the object, or (frame, None, None, 0)."""
        for k in range(tries):
            f = self.still_frame() if still and k == 0 else self.frame()
            got = self.pm.locate(f, near=near, radius=radius, min_peak=self.min_peak)
            if got is not None:
                box, _, peak = got
                return f, box, np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]), peak
        return f, None, None, 0.0

    def outline(self, f, box):
        """Object mask, narrow-direction angle (deg, picture) and its extent (px) along `axis_deg`."""
        # raw SAM outline from the box (Refiner.refine rejects outlines < 70 % of the box, which
        # throws away the right answer for a loose hand-drawn box: 22 Oct teach)
        # A white object on a light floor is low contrast: with a loose box SAM sometimes outlines
        # only the dark strap (9 % of the box). Try: box; box + a point at its centre; the patch
        # matcher's own map inside the box. First one of a plausible size wins.
        box = [float(v) for v in box]
        area_box = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        mask = None
        for kw in ({"bboxes": [box]}, {"bboxes": [box], "points": [[cx, cy]], "labels": [1]}):
            res = self.sam.sam(f, verbose=False, **kw)[0]
            if res.masks is not None:
                m = res.masks.data[0].cpu().numpy().astype(np.uint8)
                if 0.15 * area_box < m.sum() < 1.1 * area_box:
                    mask = m
                    break
        if mask is None:
            s_map, _ = self.pm.score_map(f)
            m = cv2.resize((s_map > 0).astype(np.uint8), (f.shape[1], f.shape[0]), interpolation=cv2.INTER_NEAREST)
            inbox = np.zeros_like(m)
            inbox[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1
            m = m & inbox
            n, lab, st, _ = cv2.connectedComponentsWithStats(m)
            if n > 1:
                k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                m = (lab == k).astype(np.uint8)
                if 0.15 * area_box < m.sum() < 1.1 * area_box:
                    mask = m
                    self.r.note("outline from the patch matcher (SAM did not fit)")
        if mask is None:
            return None
        ys, xs = np.nonzero(mask)
        pts = np.c_[xs, ys].astype(float)
        c = pts.mean(0)
        vals, vecs = np.linalg.eigh(np.cov((pts - c).T))
        minor = vecs[:, 0]
        return mask, float(np.degrees(np.arctan2(minor[1], minor[0]))), pts

    def precise_centre(self, f, box):
        """Centre of the object's outline (sub-pixel), or None. The matcher's box moves in ~10 px
        jumps (its patch grid): centring on it never settled within 6 px (22 Oct)."""
        out = self.outline(f, box)
        return None if out is None else core_centre(out[0])

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


def look(arm, xy, roll, h=LOOK_H, duration=None, tilt=0.0, settle=True):
    from iro.pick import pick_config
    table = pick_config()["table_z"]
    return arm.move_tip([xy[0], xy[1], table + h], roll=roll, duration=duration, tilt=tilt, settle=settle)


OVER_H = (LOOK_H, 0.13, 0.11, 0.10)    # m: heights to come in over the object from (a standing tape is 7.6 cm)


def max_lean_deg():
    """The most the gripper leaned outward near the floor when you moved it by hand (iro arm reach)."""
    from iro.reach import REACH
    if not REACH.exists():
        return 45.0
    b = [x for x in yaml.safe_load(REACH.read_text())["bands"] if x["height_cm"][0] < 6]
    return float(max([x["far_lean_deg"] for x in b] + [0.0])) or 45.0


def plan_reach(arm, xy, roll, h_grasp, step_deg=5, quick=False):
    """(over height, tilt rad) for coming down on floor point xy, or None if out of reach.
    Straight down reaches ~27 cm out at 15 cm up; leaning outward reaches ~40 cm at the floor
    (measured by hand, 22 Oct: config/reach.yaml). Least lean first: leaning tilts the jaws, which
    can brush a tall object. One tilt for the whole way down, so the jaws do not swing near it."""
    from iro.motion import SEEDS
    from iro.pick import pick_config
    table = pick_config()["table_z"]

    def ok(h, tilt):
        for s in [arm.joints()] + [np.array(x) for x in SEEDS]:
            s = np.array(s, float)
            s[4] = roll
            if arm.kin.ik_down([xy[0], xy[1], table + h], s, tilt=tilt, iters=40)[1] < 0.003:
                return True
        return False

    degs = range(0, int(max_lean_deg()) + 1, step_deg)
    if quick:                                      # reachable at all? (the full search takes ~10 s far out)
        degs = list(range(0, int(max_lean_deg()) + 1, 10))   # 22 Oct: [0, 30, 64] missed a 40 cm tape that 45 reaches
    for deg in degs:
        t = -np.radians(deg)                       # ik_down: negative = leaning outward
        if not ok(h_grasp, t) or not ok(0.10, t):
            continue
        for h in OVER_H:
            if ok(h, t):
                return h, t
    return None


def heading(arm):
    """Direction (rad, on the floor) of the gripper's x axis: the wrist camera turns with it."""
    x = arm.kin.fk(arm.joints(), "gripper_link")[:3, 0]
    return float(np.arctan2(x[1], x[0]))


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def orbit_xy(arm, P, roll, off, h=LOOK_H, iters=20):
    """Where to look from so the floor point P sits on the grasp pixel with the wrist at `roll`.
    The camera is 12 cm off the roll axis: turning the wrist in place swings the view in a circle
    (22 Oct 16:28: the tape ended up 290 px away and was "lost"). So the camera orbits around P
    instead: look point = P + offset turned by the gripper's heading AT that pose (solved by IK
    without moving; the pan angle changes with the look point, hence a few iterations)."""
    from iro.pick import pick_config
    table = pick_config()["table_z"]
    seed = arm.joints()
    seed[4] = roll
    xy = np.asarray(P, float) + rot(heading(arm)) @ off
    for _ in range(iters):
        q, _ = arm.kin.ik_down([xy[0], xy[1], table + h], seed, iters=40)
        x = arm.kin.fk(q, "gripper_link")[:3, 0]
        new = np.asarray(P, float) + rot(np.arctan2(x[1], x[0])) @ off
        if np.linalg.norm(new - xy) < 2e-4:
            return new
        xy = 0.5 * (xy + new)                  # damped: undamped it oscillated (17 mm off at +30 deg)
        seed = q
    return xy


def floor_jacobian(arm, eye, xy, roll, step=0.02):
    """How the whole picture (the floor) shifts per metre of tip motion (2x2), from two small test
    moves and phase correlation: needs no object, only floor texture."""
    from iro.lockon import gray_small, picture_shift
    look(arm, xy, roll)
    f0 = eye.frame()
    J = np.zeros((2, 2))
    for i in range(2):
        p = xy.copy()
        p[i] += step
        look(arm, p, roll)
        f1 = eye.frame()
        J[:, i] = picture_shift(gray_small(f0), gray_small(f1), f0.shape[1])[0] / step
    look(arm, xy, roll)
    return J


class SideCam:
    """The OAK-D films the whole pick from the side (feed `oak` in the recording). The wrist camera
    cannot see the jaws' last centimetres (it looks ahead of them), so this is the only view of
    what knocks the object over (22 Oct: a standing tape was tipped flat by the descent).
    Best effort: no OAK-D, no side view; picking goes on."""

    def __init__(self, r, fps=6, scout=None):
        import threading
        self.r, self.stage, self.ok, self.stop_, self.scout = r, "start", False, False, scout
        try:
            if scout is not None:                  # one OAK-D: the scout owns it, reads are locked
                self.oak = scout
            else:
                from iro.oak import Oak
                self.oak = Oak(ir=0.0)
            self.ok = True
        except Exception as e:                     # noqa: BLE001 - side view is optional
            r.note(f"no OAK-D side view: {str(e)[:80]}")
            return
        self.t = threading.Thread(target=self._loop, args=(1 / fps,), daemon=True)
        self.t.start()

    def _loop(self, dt):
        while not self.stop_:
            try:
                img, _ = self.oak.read(1)
                self.r.video("oak", img, [self.stage])
            except Exception as e:                 # noqa: BLE001 - keep picking without it
                self.r.note(f"OAK-D side view stopped: {str(e)[:80]}")
                return
            time.sleep(dt)

    def close(self):
        if self.ok:
            self.stop_ = True
            self.t.join(timeout=3)
            if self.scout is None:
                self.oak.close()


def holding(arm, grip, r, squeeze=12):
    """Squeeze test: aim `squeeze` % further closed for a moment. Holding it, the jaws stay near the
    grip width; empty, they close ~18 mm further. (22 Oct: the old test, "did the jaws close
    further while lifting", said "holding" over a tape left on the floor: the carrying grip
    only aims 4 % past contact, so empty jaws barely move.)"""
    from iro.measure import HOLD_PCT, gripper_raw, opening_mm
    c = arm.cal["gripper"]
    arm.bus.write("Goal_Position", "gripper", gripper_raw(arm, max(0.0, grip["touch_pct"] - squeeze)), normalize=False)
    time.sleep(0.6)
    now = 100 * (arm.bus.read("Present_Position", "gripper", normalize=False) - c["range_min"]) / (
        c["range_max"] - c["range_min"])
    arm.bus.write("Goal_Position", "gripper", gripper_raw(arm, max(0.0, grip["touch_pct"] - HOLD_PCT)), normalize=False)
    mm = opening_mm(now)
    empty = opening_mm(max(0.0, grip["touch_pct"] - squeeze))     # where empty jaws would end up
    # halfway between "held" and "empty": a held tape squeezed to 18-20 mm from 26-27 (3 picks,
    # 22 Oct), empty ~5 mm; the first cut-off (grip - 9 mm) was only 1 mm from a real hold
    r.note(f"squeeze test: jaws at {mm:.1f} mm (gripped at {grip['width_mm']:.1f}, empty would be {empty:.1f})")
    return mm > (grip["width_mm"] + empty) / 2, mm


# ------------------------------------------------------------------ teach

def cmd_teach(name):
    """22 Oct: the wrist camera looks AHEAD of the jaws: straight above the object, the object is below
    the bottom of the picture (the first teach "learnt" the tape's strap instead). So after letting go,
    the arm backs off, in the direction that slides the picture up, until the whole object is in view
    and you confirm it. That offset (in the gripper's own frame: the camera turns with the wrist) is
    remembered: picking looks from the same offset, then moves over by it and descends."""
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
    xy = np.array([0.24, 0.0])                      # far enough out to back off up to ~13 cm (22 Oct test)
    try:
        with Arm() as arm:
            roll = arm.level_roll()
            set_gripper(arm, open_pct)
            look(arm, xy, roll)
            J = floor_jacobian(arm, eye, xy, roll)       # before the object is there: plain floor
            r.note(f"floor Jacobian px/m {np.round(J).tolist()}")
            up = np.linalg.solve(J, [0.0, -1.0])         # tip move that slides the picture UP
            up /= np.linalg.norm(up)
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
            psi = heading(arm)
            chosen, off = None, np.zeros(2)
            for k in range(0, 6):                           # back off 0, 3, 6 ... 15 cm until it is in view
                off = k * 0.03 * up
                try:
                    look(arm, xy + off, roll)
                except Exception as e:                      # noqa: BLE001 - out of reach: stop backing off
                    r.note(f"offset {k * 3} cm unreachable: {e}")
                    break
                time.sleep(0.3)
                f, box, c, peak = eye.find()
                h_, w_ = f.shape[:2]
                inside = box is not None and box[0] > 8 and box[1] > 8 and box[2] < w_ - 8 and box[3] < h_ - 8
                eye.show(f, f"offset {k * 3} cm: " + (f"is this the {name}? (match {peak:.2f})" if inside else "not fully in view"),
                         box if inside else None, tag=f"teach_offset{k * 3:02d}")
                if not inside:
                    continue
                a_ = input(f"  {k * 3} cm back: is the GREEN box on the {name} (live page)? Enter = yes, n = look further, "
                           "d = draw it: ").strip().lower()
                if a_ == "d":
                    b_ = eye.cam.feed.ask_box(f.shape, timeout=120)
                    if b_ is not None:
                        box = np.array(b_, float)
                        c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                        chosen = (off, f, box, c)
                        break
                elif a_ != "n":
                    chosen = (off, f, box, c)
                    break
            if chosen is None:
                print(f"  Not found automatically. Drag a box around the {name} on the live page (2 min).")
                f = eye.frame()
                b_ = eye.cam.feed.ask_box(f.shape, timeout=120)
                if b_ is None:
                    return 1
                box = np.array(b_, float)
                chosen = (off, f, box, np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]))
            off, f, box, c = chosen
            eye.r.snap("teach_raw", f)                     # raw picture: can be re-done from the recording
            np.save(eye.r.dir / "teach_box.npy", np.asarray(box, float))
            out = eye.outline(f, box)
            if out is None:
                print("  Could not outline it (SAM): the outline spilled over the floor or was a speck. Draw the box "
                      "snug around the object and try again.")
                return 1
            mask, minor_deg, pts = out
            c = pts.mean(0)                                 # outline centre: what picking centres on
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
            off_local = rot(-psi) @ off                     # in the gripper's own frame
            eye.show(f, f"grasp pixel ({c[0]:.0f},{c[1]:.0f}) from {np.linalg.norm(off) * 100:.0f} cm back; "
                        f"jaws {jaw_deg:.0f} deg; {mm_per_px:.2f} mm/px", box, c, "teach_result")
    finally:
        eye.close()
    data = {"grasp_pixel": [round(float(c[0]), 1), round(float(c[1]), 1)], "jaw_axis_deg": round(jaw_deg, 1),
            "look_offset_gripper_m": [round(float(v), 4) for v in off_local],
            "mm_per_px": round(mm_per_px, 4), "look_h_m": LOOK_H, "grasp_h_m": GRASP_H,
            "taught_width_mm": round(grip["width_mm"], 1), "roll_sign": 1, "top_peak": round(top_peak, 3),
            "recording": str(r.dir), "date": time.strftime("%Y-%m-%d %H:%M")}
    grasp_file(name).write_text(yaml.safe_dump(data))
    r.note(f"taught: {data}")
    print(f"Taught. Seen from {np.linalg.norm(off) * 100:.0f} cm back, the {name} sits at pixel {data['grasp_pixel']} "
          f"when the jaws are over it. Next: iro pick {name}")
    return 0


# ------------------------------------------------------------------ pick

def core_centre(mask):
    """Centre of the THICK part of an outline: the pixels deeper inside than half the deepest one.
    A strap, a handle or a cut-off corner moves the plain average of the outline; not this."""
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(d > 0.5 * d.max())
    return np.array([xs.mean(), ys.mean()])


WRIST = ROOT / "config" / "wrist_camera.yaml"


def wrist_model():
    """(Jg, Mz) measured by `iro pick calibrate`, or None. Jg (2x2): how far (px) the object moves in
    the wrist picture per metre the gripper moves, in the GRIPPER's own frame at the look pose, so it
    holds for every look point and wrist roll: J in the arm frame = Jg @ rot(-heading). Mz (2): px per
    metre the gripper rises. 15 Nov: measuring J before every centring from the floor's texture
    (two test moves, ~2.5 s) gave rubbish on a plain wooden floor (7 tries, entries differing 3x and
    in sign); measured once on the object it fits to ~5 px."""
    if not WRIST.exists():
        return None
    d = yaml.safe_load(WRIST.read_text())
    return np.array(d["Jg"]), np.array(d["Mz"])


def see(arm, eye, near=None, radius=None):
    """The object in a picture taken with the arm standing still: (frame, box, centre, peak), the
    centre being the outline's thick core. The wrist camera looks at a slant: a picture taken while
    the joints still creep (up to 0.4 s after a move) is up to 100 px off (15 Nov)."""
    arm.wait_still()
    f, box, c, peak = eye.find(near=near, radius=radius, still=True)
    if box is None and near is not None:
        f, box, c, peak = eye.find()
    if box is not None:
        pc = eye.precise_centre(f, box)
        c = pc if pc is not None else c
    return f, box, c, peak


def cmd_calibrate(name):
    """iro pick calibrate NAME: measure the wrist camera's motion model on an object lying still in
    front of the robot (found by the OAK-D). 10 small moves, ~20 s. Saves config/wrist_camera.yaml."""
    from iro.motion import Arm
    from iro.oakguide import Scout
    from iro.record import run
    gf = grasp_file(name)
    if not gf.exists():
        print(f"Teach the grasp first: iro pick teach {name}")
        return 1
    G = yaml.safe_load(gf.read_text())
    off = np.array(G["look_offset_gripper_m"])
    scout = Scout()
    o = scout.find(name)
    scout.close()
    if o is None:
        print(f"The OAK-D does not see the {name} on the floor (18-45 cm in front of the arm).")
        return 1
    r = run(f"pick-calibrate-{name}")
    eye = DownEye(name, r)
    eye.min_peak = 0.5 * G.get("top_peak", 0) or None
    try:
        with Arm() as arm:
            roll = arm.level_roll()
            xy0 = orbit_xy(arm, o["centre"], roll, off)
            look(arm, xy0, roll)
            f, box, c0, _ = see(arm, eye)
            if box is None:
                print(f"The wrist camera does not see the {name} from above it.")
                return 1
            hd = heading(arm)
            # bring it near the picture's middle first: the fit is best away from the edges
            D, C = [], []
            moves = [(0.012, 0, 0), (-0.012, 0, 0), (0, 0.012, 0), (0, -0.012, 0), (0.01, 0.01, 0), (-0.01, -0.01, 0),
                     (0, 0, 0.02), (0, 0, -0.02), (0.01, -0.01, 0.01), (0, 0, 0)]
            for dx, dy, dz in moves:
                d = rot(hd) @ np.array([dx, dy])                       # gripper frame -> arm frame
                look(arm, xy0 + d, roll, h=LOOK_H + dz)
                f, box, c, _ = see(arm, eye, near=c0, radius=250)
                if box is None:
                    continue
                D.append([dx, dy, dz])
                C.append(c - c0)
            look(arm, xy0, roll)
        D, C = np.array(D), np.array(C)
        if len(D) < 7:
            print(f"Only {len(D)} of {len(moves)} sightings: not enough. Is the {name} well inside the picture?")
            return 1
        M = np.linalg.lstsq(np.c_[D, np.ones(len(D))], C, rcond=None)[0][:3].T          # 2x3
        e = np.linalg.norm(np.c_[D, np.ones(len(D))] @ np.linalg.lstsq(np.c_[D, np.ones(len(D))], C, rcond=None)[0] - C, axis=1)
        data = {"Jg": np.round(M[:, :2], 1).tolist(), "Mz": np.round(M[:, 2], 1).tolist(), "look_h_m": LOOK_H,
                "object": name, "object_height_m": round(float(o["height"]), 3), "fit_px_median": round(float(np.median(e)), 1),
                "fit_px_worst": round(float(e.max()), 1), "sightings": len(D), "recording": str(r.dir),
                "date": time.strftime("%Y-%m-%d %H:%M")}
        r.note(f"wrist camera model: {data}")
        if np.median(e) > 12:
            print(f"The fit is poor (median {np.median(e):.0f} px): not saved. Was the {name} moved, or near the picture's edge?")
            return 1
        WRIST.write_text("# Written by `iro pick calibrate`: how the wrist picture moves with the gripper (see grasp.wrist_model)\n"
                         + yaml.safe_dump(data))
        print(f"Saved {WRIST.name}: sideways {np.abs(M[:, :2]).max():.0f} px/m, up/down {np.linalg.norm(M[:, 2]):.0f} px/m; "
              f"fit {np.median(e):.1f} px (worst {e.max():.1f}).")
        return 0
    finally:
        eye.close()


def centre2(arm, eye, xy, roll, Jg, target, c, tries=6, scale=1.0, keep=None):
    """Centre the object on the target pixel with the stored camera model: usually ONE step.
    J = scale * Jg @ rot(-heading). The scale starts at 1 and is corrected from what each step really
    did (a taller object is nearer the camera and moves more px per metre), within 0.5-2x.
    Returns (xy, frame, box, c)."""
    f = box = None
    for k in range(tries):
        err = c - target
        if np.linalg.norm(err) < SERVO_TOL_PX and box is not None:
            if keep is not None:
                keep["scale"] = scale
            return xy, f, box, c
        t_step = time.time()
        J = scale * Jg @ rot(-heading(arm))
        step = np.clip(-np.linalg.solve(J, err), -0.04, 0.04) if np.linalg.norm(err) >= SERVO_TOL_PX else np.zeros(2)
        if np.any(step):
            xy = xy + step
            look(arm, xy, roll)
        expect = c + J @ step
        f, box, c2, _ = see(arm, eye, near=expect, radius=150)
        if box is None:
            raise RuntimeError("lost the object while centring")
        moved, pred = c2 - c, J @ step
        if np.linalg.norm(pred) > 20:
            scale = float(np.clip(scale * (moved @ pred) / (pred @ pred), 0.5, 2.0))
        eye.r.log("centre", step=k, err_px=float(np.linalg.norm(err)), x=float(xy[0]), y=float(xy[1]),
                  after_px=float(np.linalg.norm(c2 - target)), scale=scale, step_s=time.time() - t_step)
        c = c2
    raise RuntimeError(f"could not centre it in {tries} steps")


def centre(arm, eye, xy, roll, J, target, c, tries=15):
    """Step sideways until the object's OUTLINE centre sits on the target pixel. (xy, frame, box, c)."""
    Jinv = np.linalg.inv(J)
    f = box = None
    for k in range(tries):
        err = c - target
        if np.linalg.norm(err) < SERVO_TOL_PX:
            if box is None:                             # centred already: no step, so no picture yet
                f, box, c2, _ = eye.find(near=c, radius=150)
                if box is None:
                    f, box, c2, _ = eye.find()
                if box is None:
                    raise RuntimeError("lost the object while centring")
                pc = eye.precise_centre(f, box)
                c = pc if pc is not None else c2
                if np.linalg.norm(c - target) >= SERVO_TOL_PX:
                    continue                            # the fresh picture says it is off after all: step
            break
        step = np.clip(-0.7 * Jinv @ err, -0.03, 0.03)
        xy = xy + step
        look(arm, xy, roll)
        # search where the step should have moved it: a big step moves it > 150 px (15 Nov: 4 of 7
        # picks "lost" a tape that was in plain view, 160 px from where it had been)
        expect = c + J @ step
        f, box, c, _ = eye.find(near=expect, radius=150)
        if box is None:
            f, box, c, _ = eye.find()                   # anywhere in the picture before giving up
            if box is not None:
                eye.r.note(f"centring: found {np.linalg.norm(c - expect):.0f} px from where expected")
        if box is None:
            raise RuntimeError("lost the object while centring")
        pc = eye.precise_centre(f, box)
        if pc is not None:
            c = pc
        eye.r.log("centre", step=k, err_px=float(np.linalg.norm(err)), x=float(xy[0]), y=float(xy[1]))
    else:
        raise RuntimeError(f"could not centre it in {tries} steps")
    return xy, f, box, c


def log_pose(arm, r, stage, **extra):
    """Joints + tip in the recording at every step (to measure offset errors afterwards)."""
    q = arm.joints()
    tip = arm.kin.fk(q)[:3, 3]
    r.log("poses", stage=stage, q_deg=np.round(np.degrees(q), 2).tolist(), tip_m=np.round(tip, 4).tolist(), **extra)


def tallest_m(name, default=0.08):
    """The object's largest side (objects/NAME/size.yaml): how high it can be if it stands up."""
    from iro.objects import OBJ_DIR
    f = OBJ_DIR / name / "size.yaml"
    if not f.exists():
        return default
    d = yaml.safe_load(f.read_text()) or {}
    sizes = [d.get("height_m"), d.get("length_m"), d.get("width_m")] + ((d.get("ruler") or {}).get("sizes_m") or [])
    sizes = [v for v in sizes if isinstance(v, (int, float))]
    return max(sizes) if sizes else default


LIFT_V = 0.12        # m/s top tip speed lifting / carrying a held object
PLACE_V = 0.06       # m/s top speed setting it down (minimum-jerk: touches down at ~0)


def pick_table():
    from iro.pick import pick_config
    return pick_config()["table_z"]


def put_down(arm, xy, roll, grasp_h, open_pct, over_h, tilt=0.0):
    """Set a held object down at floor point xy (straight down, soft), let go, rise.
    Gentle on purpose: all the way down to the height it was gripped at (let go 4 mm up, a tape
    standing on its rim rocked and fell over, 15 Nov), jaws fully open and still before rising, and
    the first 3 cm up slowly (the open jaws pass right beside it)."""
    from iro.measure import set_gripper
    table = pick_table()
    if np.linalg.norm(arm.tip()[:2] - xy) > 0.004:
        look(arm, xy, roll, h=max(0.10, arm.tip()[2] - table), tilt=tilt, settle=False)
    arm.move_line([xy[0], xy[1], table + grasp_h + 0.001], speed=PLACE_V, tilt=tilt, roll=roll)
    time.sleep(0.15)                                     # let it stand still before letting go
    set_gripper(arm, open_pct, wait=1.0)
    time.sleep(0.2)
    arm.move_line([xy[0], xy[1], table + grasp_h + 0.03], speed=0.03, tilt=tilt, roll=roll, settle=False)
    arm.move_line([xy[0], xy[1], table + over_h], speed=FAST_V, tilt=tilt, roll=roll, settle=False)


FAST_V = 0.15        # m/s top tip speed above the object (free air)
CAREFUL_V = 0.06     # m/s top speed near it; minimum-jerk slows to 0 at the end: soft landing


def descend(arm, xy, roll, h_to, r, top_m=0.08, block_mm=8, tilt=0.0):
    """Down to h_to as two SMOOTH straight-line moves (arm.move_line), no stops on the way.
    FAST while the jaws are higher than the object could possibly be (top_m + 2 cm: nothing to touch);
    CAREFUL below that, watched ~50x a second for a jaw landing on something (22 Oct: a standing
    tape was knocked over): BLOCKED = the tip has STOPPED going down while its command keeps going
    down (over the last 0.25 s the command fell >= 6 mm, the tip < 1.5 mm), or the tip ends more than
    block_mm above where it was sent. Then it backs up 3 cm.
    Not "the tip is more than X behind its command": near full stretch (33 cm out, leaning) the
    joints lag 8 mm behind in free air and that rule stopped two good picks (15 Nov).
    15 Nov: 1 cm steps took 22.5 s, stepped at 2 speeds 13 s, this ~5 s. Returns True when it reached h_to."""
    table = pick_table()
    careful_from = h_to + top_m + 0.02
    t0 = time.time()
    if arm.tip()[2] - table > careful_from + 0.005:
        arm.move_line([xy[0], xy[1], table + careful_from], speed=FAST_V, tilt=tilt, roll=roll, settle=False)
    t1 = time.time()
    hist, worst = [], [0.0]

    def watch(k, q_cmd, q_now):
        z_cmd, z_now = arm.kin.fk(q_cmd)[2, 3], arm.kin.fk(q_now)[2, 3]
        now = time.time()
        hist.append((now, z_cmd, z_now))
        worst[0] = max(worst[0], z_now - z_cmd)
        while len(hist) > 2 and now - hist[1][0] >= 0.25:
            hist.pop(0)
        if now - hist[0][0] < 0.22:
            return False
        # ... and it must be ABOVE its command: a tip hanging a little below it (gravity, far out)
        # waits for the command to catch up at the start, which is no blockage (15 Nov, free air)
        return hist[0][1] - z_cmd >= 0.006 and hist[0][2] - z_now < 0.0015 and z_now - z_cmd > 0.005

    reached = arm.move_line([xy[0], xy[1], table + h_to], speed=CAREFUL_V, tilt=tilt, roll=roll, watch=watch)
    got = arm.tip()[2] - table
    if reached and got - h_to > block_mm / 1000:
        reached = False
    if not reached:
        r.note(f"blocked at {got * 100:.1f} cm (the tip stopped going down; it was {worst[0] * 1000:.0f} mm behind at most)")
        look(arm, xy, roll, h=got + 0.03, duration=0.8, tilt=tilt)
        return False
    r.note(f"down in {time.time() - t0:.1f} s (fast {t1 - t0:.1f} s, careful below {careful_from * 100:.1f} cm; "
           f"the tip was {worst[0] * 1000:.0f} mm behind at most)")
    return True


def roll_across(arm, P, yaw, h=LOOK_H):
    """(wrist roll, how well 0-1) that makes the jaws close ACROSS an object's long side (direction
    `yaw` on the floor) when the gripper is straight down over floor point P. From the arm's own
    model (which way the jaws close for given joints), no picture needed. The jaw direction turns
    1:1 with the roll when the gripper points down; the nearest of the two equivalent rolls
    (180 deg apart) inside the joint's range is taken."""
    from iro.efad import jaw_direction
    from iro.pick import pick_config
    xyz = [P[0], P[1], pick_config()["table_z"] + h]
    level = arm.level_roll()
    lo, hi = arm.kin.limits[4]

    def jaw_angle(roll):
        seed = arm.joints()
        seed[4] = roll
        q, _ = arm.kin.ik_down(xyz, seed, iters=60)
        d = jaw_direction(arm.kin, q)
        return np.arctan2(d[1], d[0])
    a0 = jaw_angle(level)
    sgn = 1.0 if np.cos(jaw_angle(level + 0.2) - a0 - 0.2) > np.cos(jaw_angle(level + 0.2) - a0 + 0.2) else -1.0
    want = yaw + np.pi / 2                                   # across the long side
    d = (want - a0 + np.pi / 2) % np.pi - np.pi / 2          # jaws are a line: (-90, 90] deg
    cands = [level + sgn * (d + k * np.pi) for k in (0, 1, -1)]
    cands = [c for c in cands if lo + 0.05 <= c <= hi - 0.05]
    if not cands:
        return level, 0.0
    roll = min(cands, key=lambda c: abs(c - level))
    return roll, float(abs(np.cos(jaw_angle(roll) - want)))


def find_from(arm, eye, starts, roll):
    """Look from each start pose until the object is seen. (xy, frame, box, centre, peak) or None."""
    for xy in starts:
        xy = np.array(xy, float)
        try:
            look(arm, xy, roll)
        except Exception:                              # noqa: BLE001 - out of reach: next one
            continue
        f, box, c, peak = eye.find()
        if box is not None:
            return xy, f, box, c, peak
    return None


def cmd_pick(name, drop=False, attempts=3, to=None):
    """Find, centre, turn, open, descend feeling for blockages, close until contact, lift, check.
    A miss is looked at again (did it move?) and retried, up to `attempts` times."""
    from iro.measure import close_on, max_open_m, pct_for, set_gripper
    from iro.motion import Arm, MotionError
    from iro.record import run
    gf = grasp_file(name)
    if not gf.exists():
        print(f"Teach the grasp first: iro pick teach {name}")
        return 1
    G = yaml.safe_load(gf.read_text())
    target = np.array(G["grasp_pixel"])
    off = np.array(G.get("look_offset_gripper_m", [0.0, 0.0]))
    r = run(f"pick-{name}")
    t_start = time.time()
    # the OAK-D (the scout) starts up in the background while the wrist camera's models load
    import threading
    box_ = {}

    def open_scout():
        try:
            import os
            if os.environ.get("IRO_NO_SCOUT"):         # test the wrist-only search: IRO_NO_SCOUT=1 iro pick NAME
                raise RuntimeError("switched off (IRO_NO_SCOUT)")
            from iro.oakguide import Scout
            box_["scout"] = Scout()
        except Exception as e:                     # noqa: BLE001 - without it: the old search from above
            box_["why"] = str(e)[:100]
    th = threading.Thread(target=open_scout, daemon=True)
    th.start()
    eye = DownEye(name, r)
    try:                                           # first use of each model is slow (0.7 s + 1-2 s): do it now
        f0 = eye.frame()
        eye.pm.locate(f0)
        eye.sam.sam(f0, verbose=False, bboxes=[[200.0, 150.0, 440.0, 330.0]])
    except Exception:                              # noqa: BLE001 - only a warm-up
        pass
    th.join(timeout=25)
    scout, model = box_.get("scout"), wrist_model()
    if scout is None:
        r.note(f"no OAK-D scout ({box_.get('why', 'still starting')}): searching with the wrist camera")
    elif model is None:
        r.note("no wrist camera model (iro pick calibrate NAME): searching with the wrist camera")
    side = SideCam(r, scout=scout)
    r.note(f"ready in {time.time() - t_start:.1f} s")
    if G.get("top_peak"):
        eye.min_peak = 0.5 * G["top_peak"]          # seen from above it must match like it did then
    starts = list(SEARCH)
    try:
        with Arm() as arm:
            level = arm.level_roll()
            for attempt in range(1, attempts + 1):
                roll = level
                set_gripper(arm, 0)
                tag = f"a{attempt}_"
                # 0. the OAK-D scout: where it is and which way it lies, before the arm moves. Then
                #    straight to the look pose that puts it on the grasp pixel, jaws already turned.
                o = None
                if scout is not None and model is not None:
                    try:
                        o = scout.find(name)
                    except Exception as e:                 # noqa: BLE001 - camera hiccup: search instead
                        r.note(f"scout failed: {str(e)[:80]}")
                    if o is not None:
                        P = o["centre"]
                        roll_g, across = roll_across(arm, P, o["yaw"])
                        r.snap(tag + "0_scout", scout.draw(pick=o))
                        r.note(f"scout: {name} at ({P[0]:.3f}, {P[1]:.3f}), {o['length'] * 100:.1f} x {o['width'] * 100:.1f} x "
                               f"{o['height'] * 100:.1f} cm, long side {np.degrees(o['yaw']):+.0f} deg; roll "
                               f"{np.degrees(roll_g - level):+.0f} deg (jaws {across * 100:.0f}% across)")
                        try:
                            xy = orbit_xy(arm, P, roll_g, off)
                            look(arm, xy, roll_g)
                            f, box, c, peak = see(arm, eye, near=target, radius=220)
                        except MotionError as e:
                            r.note(f"scout look pose not reachable ({e}): searching")
                            box = None
                        if box is None:
                            r.note("the wrist camera did not confirm the scout's object: searching")
                            o = None
                if o is not None:
                    roll = roll_g
                    eye.show(f, f"1. found {name} via the OAK-D (match {peak:.2f}, {np.linalg.norm(c - target):.0f} px off)",
                             box, target, tag + "1_found")
                    print(f"{'' if attempt == 1 else f'[try {attempt}] '}1. the OAK-D found it at {P[0] * 100:.0f}, "
                          f"{P[1] * 100:+.0f} cm; wrist camera confirms (match {peak:.2f}, {np.linalg.norm(c - target):.0f} px from the mark)")
                    r.note(f"aim: {np.linalg.norm(c - target):.0f} px from the grasp pixel at first sight")
                    kept = {}
                    xy, f, box, c = centre2(arm, eye, xy, roll, model[0], target, c, scale=G.get("centre_scale", 1.0), keep=kept)
                    r.note(f"centred to {np.linalg.norm(c - target):.0f} px")
                    eye.show(f, "2. centred", box, target, tag + "2_centred")
                    log_pose(arm, r, "centred", attempt=attempt)
                    P0 = xy - rot(heading(arm)) @ off
                    if plan_reach(arm, P0, roll, G["grasp_h_m"], quick=True) is None:
                        print(f"2. It is {np.linalg.norm(P0) * 100:.0f} cm from the base: out of reach. Move it closer.")
                        r.note(f"out of reach at {np.round(P0, 3).tolist()}")
                        return 1
                    out = eye.outline(f, box)
                if o is None:
                    # 1. find it from above (no scout, or it did not see it)
                    got = find_from(arm, eye, starts, roll)
                    if got is None:
                        f = eye.frame()
                        eye.show(f, f"{name} not found from above", tag=tag + "notfound")
                        print(f"1. The {name} is not in the search area (looked from {len(starts)} places).")
                        return 1
                    xy, f, box, c, peak = got
                    pc = eye.precise_centre(f, box)
                    c = pc if pc is not None else c
                    eye.show(f, f"1. found {name} (match {peak:.2f})", box, target, tag + "1_found")
                    print(f"{'' if attempt == 1 else f'[try {attempt}] '}1. found it (match {peak:.2f})")
                    # 2. centre it on the grasp pixel
                    # steering from the floor texture: an object at the picture's edge cannot be lost by a
                    # test move (22 Oct: a knocked-over tape at the bottom edge was)
                    J = floor_jacobian(arm, eye, xy, roll)
                    xy, f, box, c = centre(arm, eye, xy, roll, J, target, c)
                    eye.show(f, "2. centred", box, target, tag + "2_centred")
                    log_pose(arm, r, "centred", attempt=attempt)
                    P0 = xy - rot(heading(arm)) @ off
                    if plan_reach(arm, P0, roll, G["grasp_h_m"], quick=True) is None:
                        dist = np.linalg.norm(P0) * 100
                        eye.show(f, f"{dist:.0f} cm out: out of reach", box, target, tag + "2_toofar")
                        print(f"2. It is {dist:.0f} cm from the base: too far even leaning the gripper "
                              f"{max_lean_deg():.0f} deg (iro arm reach). Move it closer (under ~35 cm) and run again.")
                        r.note(f"out of reach at {np.round(P0, 3).tolist()}")
                        return 1
                    # 3. turn the jaws across its narrow side
                    out = eye.outline(f, box)
                    if out is not None:
                        d = wrap90(out[1] - G["jaw_axis_deg"])
                        if abs(d) > 8:
                            P = xy - rot(heading(arm)) @ off        # the object's floor point: fixed while turning
                            sign = G.get("roll_sign", 1)

                            def turn(s):
                                r2 = roll + s * np.radians(d)
                                xy2 = orbit_xy(arm, P, r2, off)
                                look(arm, xy2, r2)
                                got_ = eye.find(near=target, radius=200)
                                if got_[1] is None:
                                    got_ = eye.find()           # anywhere in the picture (see centre())
                                r.note(f"turned {s * d:+.0f} deg: look ({xy2[0]:.3f}, {xy2[1]:.3f}), "
                                       f"{'seen' if got_[1] is not None else 'NOT seen'}")
                                return r2, xy2, got_

                            roll2, xy2, (f, box, c, _) = turn(sign)
                            out2 = eye.outline(f, box) if box is not None else None
                            d2 = wrap90(out2[1] - G["jaw_axis_deg"]) if out2 else None
                            if d2 is None or abs(d2) > abs(d) * 0.7:   # turned the wrong way (or unsure): try the other
                                roll3, xy3, got3 = turn(-sign)
                                out3 = eye.outline(got3[0], got3[1]) if got3[1] is not None else None
                                d3 = wrap90(out3[1] - G["jaw_axis_deg"]) if out3 else None
                                if d3 is not None and (d2 is None or abs(d3) < abs(d2)):
                                    sign = -sign
                                    roll2, xy2, (f, box, c, _) = roll3, xy3, got3
                                    G["roll_sign"] = sign
                                    gf.write_text(yaml.safe_dump(G))
                                    r.note(f"wrist roll sign flipped to {sign} (left {abs(d3):.0f} deg vs {abs(d2 or 90):.0f})")
                                elif d2 is not None:
                                    roll2, xy2, (f, box, c, _) = turn(sign)    # the first way was the better one
                            roll, xy = roll2, xy2
                            if box is None:
                                eye.show(f, "3. lost it after turning", tag=tag + "3_lost")
                                raise RuntimeError("lost the object while turning the jaws")
                            J = floor_jacobian(arm, eye, xy, roll)      # the picture is turned now
                            xy, f, box, c = centre(arm, eye, xy, roll, J, target, c)
                            out = eye.outline(f, box)
                            print(f"3. turned the jaws {d:+.0f} deg across it")
                # 4. how wide to open: room to spare, the last centimetres are blind
                width = eye.extent(out[2], G["jaw_axis_deg"]) * G["mm_per_px"] if out is not None else None
                if o is not None and o.get("completed") and across > 0.94:
                    width = o["width"] * 1000        # true size: the wrist's slanted view misjudges it (15 Nov)
                w = width or G["taught_width_mm"]
                if w + 2 * 8 > max_open_m() * 1000:             # < 8 mm room each side, blind descent: no
                    eye.show(f, f"4. {w:.0f} mm across: too wide to grasp safely", box, target, tag + "4_toowide")
                    print(f"4. Lying like this it is {w:.0f} mm across the jaws; they open {max_open_m() * 1000:.0f} mm: "
                          "less than 8 mm of room each side for a blind descent. Turn it so a narrower side faces up "
                          "(e.g. stand the tape on its edge).")
                    return 1
                # fully open, always (22 Oct: 56-60 mm still tipped a standing tape over; teach also
                # closed from fully open, so the taught grasp pixel matches this jaw position)
                want = max_open_m() * 1000
                open_pct = pct_for(want)
                set_gripper(arm, open_pct, wait=0)          # opens while the arm moves over it
                eye.show(f, f"4. opening {want:.0f} mm for {w:.0f} mm", box, target, tag + "4_open")
                print(f"2-4. centred, about {w:.0f} mm across: opening {want:.0f} mm")
                look_xy = xy.copy()
                # 5. over it (the camera looked from an offset), down feeling for blockages, close
                gxy = look_xy - rot(heading(arm)) @ off
                plan = plan_reach(arm, gxy, roll, G["grasp_h_m"])
                if plan is None:
                    print(f"5. Over it is {np.linalg.norm(gxy) * 100:.0f} cm out: out of reach. Move it closer.")
                    r.note(f"grasp point out of reach {np.round(gxy, 3).tolist()}")
                    return 1
                over_h, tilt = plan
                if tilt:
                    print(f"   leaning the gripper {np.degrees(-tilt):.0f} deg outward to reach it "
                          f"({np.linalg.norm(gxy) * 100:.0f} cm out)")
                r.note(f"over at {over_h * 100:.0f} cm, lean {np.degrees(-tilt):.0f} deg")
                side.stage = f"{attempt}: over it, going down (lean {np.degrees(-tilt):.0f})"
                look(arm, gxy, roll, h=over_h, tilt=tilt, settle=False)
                log_pose(arm, r, "over", attempt=attempt, grasp_xy=np.round(gxy, 4).tolist())
                reached = descend(arm, gxy, roll, G["grasp_h_m"], r, top_m=tallest_m(name), tilt=tilt)
                side.stage = f"{attempt}: closing" if reached else f"{attempt}: blocked"
                # closing starts from fully open: the contact detector needs ~8 free readings before
                # the touch (15 Nov: started 12 mm from the object it missed a real grip)
                grip = close_on(arm, open_pct, r, hold=True) if reached else None
                log_pose(arm, r, "closed" if grip else ("blocked" if not reached else "missed"), attempt=attempt)
                if grip is None:
                    set_gripper(arm, open_pct)
                    why = "a jaw landed on it (stopped before pushing it over)" if not reached else "closed on nothing"
                    print(f"5. Missed: {why}.")
                    # straight up first (a sideways move this low sweeps the jaws through it), then
                    # look again from where we looked before: did it move?
                    look(arm, gxy, roll, h=over_h, duration=1.5, tilt=tilt)
                    look(arm, look_xy, level, duration=1.5)
                    f2, box2, c2, _ = eye.find()
                    if box2 is None:
                        eye.show(f2, "5. missed; no longer where it was", tag=tag + "5_missed")
                        print("   It is no longer where it was (knocked away?). Searching again.")
                    else:
                        moved = np.linalg.norm(c2 - c)
                        eye.show(f2, f"5. missed; it moved {moved:.0f} px", box2, target, tag + "5_missed")
                        print(f"   Still there (moved {moved:.0f} px in the picture). Trying again.")
                    r.note(f"attempt {attempt}: {why}")
                    starts = [look_xy] + list(SEARCH)
                    continue
                print(f"5. gripped: {grip['width_mm']:.0f} mm between the fingertips (expected ~{w:.0f}), {grip['feel']}")
                taught = G["taught_width_mm"]
                if abs(grip["width_mm"] - taught) > 8:
                    # 22 Oct: 39 mm on a tape taught at 24 mm = it had been tipped flat; lifting failed
                    set_gripper(arm, open_pct)
                    look(arm, gxy, roll, h=over_h, duration=1.5, tilt=tilt)
                    eye.show(eye.frame(), f"5. {grip['width_mm']:.0f} mm, taught {taught:.0f} mm: lying differently",
                             tag=tag + "5_wrongwidth")
                    print(f"   That is {grip['width_mm']:.0f} mm, but it was taught at {taught:.0f} mm: it is not lying "
                          f"the way it was taught (knocked over?). Let go. Stand it up as taught and run again.")
                    r.note(f"width {grip['width_mm']:.1f} vs taught {taught}: not as taught")
                    return 1
                # 6. lift and check it is still held
                side.stage = f"{attempt}: lifting"
                arm.move_line([gxy[0], gxy[1], pick_table() + 0.10], speed=LIFT_V, tilt=tilt, roll=roll)
                held, mm = holding(arm, grip, r)
                if not held:
                    print(f"6. Not holding it after lifting: a squeeze closed the jaws to {mm:.0f} mm (gripped at "
                          f"{grip['width_mm']:.0f}). It slipped out or was never really between the jaws.")
                    r.note("not held after lifting")
                    set_gripper(arm, open_pct)
                    return 1
                log_pose(arm, r, "lifted", attempt=attempt)
                f, *_ = eye.find()
                eye.show(f, f"6. holding the {name}", tag=tag + "6_lifted")
                print(f"6. Holding the {name}, 10 cm up (try {attempt}).")
                if scout is not None:
                    try:
                        left = [q for q in scout.objects(2) if np.linalg.norm(q["centre"] - gxy) < 0.06
                                and q["height"] > 0.015 and q["bottom"] < 0.035]     # hanging in the jaws: bottom ~9 cm
                        r.note("OAK-D: the floor is empty where it stood" if not left else
                               f"OAK-D: something still stands on the floor there ({left[0]['height'] * 100:.1f} cm tall)")
                        if o is not None and not left:
                            # this pick's pair for the scout's self-correction: where the scout (uncorrected)
                            # had it, and where it really was
                            raw = o["front_raw"] + scout.corr.rot().T @ o.get("shift", np.zeros(2)) if o.get("completed") else o["centre_raw"]
                            if kept.get("scale") and abs(kept["scale"] - G.get("centre_scale", 1.0)) > 0.03:
                                G["centre_scale"] = round(0.5 * G.get("centre_scale", kept["scale"]) + 0.5 * kept["scale"], 3)
                                gf.write_text(yaml.safe_dump(G))
                            if scout.corr.add(raw, gxy, name):
                                r.note(f"scout pair saved ({len(scout.corr.pairs)} so far; they now fit to "
                                       f"{scout.corr.residual_mm():.0f} mm)")
                    except Exception:                      # noqa: BLE001 - a bonus check
                        pass
                r.note(f"whole pick: {time.time() - t_start:.1f} s")
                r.note(f"picked on try {attempt}: width {grip['width_mm']:.1f} mm at ({gxy[0]:.3f}, {gxy[1]:.3f})")
                if to is not None:
                    dest = np.asarray(to, float)
                    plan2 = plan_reach(arm, dest, roll, G["grasp_h_m"])
                    if plan2 is None:
                        print(f"   Cannot place it at {dest[0] * 100:.0f}, {dest[1] * 100:+.0f} cm: out of reach. Putting it back.")
                        r.note(f"place target out of reach {np.round(dest, 3).tolist()}")
                        drop = True
                    else:
                        side.stage = f"{attempt}: carrying it"
                        over2, tilt2 = plan2
                        t_c = time.time()
                        look(arm, dest, roll, h=0.10, tilt=tilt2, settle=False)          # carried 10 cm up
                        put_down(arm, dest, roll, G["grasp_h_m"], open_pct, over2, tilt2)
                        r.note(f"placed at ({dest[0]:.3f}, {dest[1]:.3f}) in {time.time() - t_c:.1f} s")
                        print(f"   Placed it at {dest[0] * 100:.0f}, {dest[1] * 100:+.0f} cm.")
                        return 0
                if drop:
                    side.stage = f"{attempt}: putting it down"
                    put_down(arm, gxy, roll, G["grasp_h_m"], open_pct, over_h, tilt)
                    print("   Put it back down.")
                return 0
            print(f"Gave up after {attempts} tries. Recording: {r.dir} (iro review)")
            return 1
    except (RuntimeError, MotionError) as e:
        print(f"STOPPED: {e}. The arm holds its pose.")
        r.note(f"stopped: {e}")
        return 1
    finally:
        eye.close()
        side.close()
        if scout is not None:
            scout.close()


def main():
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == "teach":
        return cmd_teach(a[1])
    if len(a) >= 2 and a[0] == "calibrate":
        return cmd_calibrate(a[1])
    if a and a[0] == "scout":
        from iro.oakguide import cmd_scout
        return cmd_scout()
    if a and not a[0].startswith("-"):
        to = next((x.split("=", 1)[1] for x in a if x.startswith("--to=")), None)
        if to is not None:
            try:
                to = [float(v) / 100 for v in to.split(",")]
                assert len(to) == 2
            except (ValueError, AssertionError):
                print("--to=X,Y in cm from the arm base, e.g. --to=25,-5")
                return 1
        return cmd_pick(a[0], drop="--drop" in a, to=to)
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
