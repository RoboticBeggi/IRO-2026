"""iro track: the arm follows ONE object with the wrist camera and stays locked on it.

    iro track teach NAME       FIRST: guided tour, draw boxes on 10-12 views (iro/lockon.py)
    iro track NAME             follow it
    iro track NAME --watch     vision only, the arm does not move: check the lock first
    iro track NAME --measure   re-measure the steering (after changing the camera mount)
    iro track review [RUN]     contact sheets + summary of a recorded run (default: the last one)
    iro track                  no name: wiggle any object in front of the camera to pick it
    iro track --remember NAME  follow it AND save it as NAME

Every run is RECORDED (iro/record.py): ~/iro-runs/<time>_track-<name>/track.mp4 (annotated:
boxes, search window, match heat map, why sightings were rejected), feed.mp4 (the live page)
and log.csv. ~/iro-runs/latest points to the newest; `iro review` works for every program.
Live view: http://localhost:8080 (green = locked, yellow = coasting, red = lost).
Ctrl+C = stop and hold the pose.

LOCK-IN (tracking by detection: the patch matcher decides, nothing else):
  - the patch matcher (DINOv2, iro/lockon.py) looks for the object in every picture it can
    (~15 per second): ZOOMED into a search window around the prediction while locked, the
    whole picture while lost
  - a sighting must fit the taught object's size, shape and brightness (a dark shadow or a
    long strip under the sofa is rejected) and be strong enough
  - at lock-on, the rest of the picture is learnt as "not it" (this room, this shadow)
  - a Kalman filter smooths the sightings; VitTrack only fills the gaps between two sightings
    and is ignored as soon as the matcher has not confirmed it for 0.4 s
  - while lost, a re-find needs 2 sightings in a row at the same spot

SMOOTH, SAFE CONTROL:
  - the arm moves ONLY while a fresh sighting (< 0.6 s) exists; otherwise it holds still
  - only pan and wrist steer; the other joints keep their commanded pose (commanding the
    measured, gravity-sagged angles is what made the arm creep downwards)
  - commands never run more than 4 deg ahead of the measured joints (no wind-up)
  - deadband with hysteresis (start > 22 px off, stop < 10 px), 15 Hz, gain halves on overshoot
  - wrong steering is caught when it is MEASURED (fresh pictures; + and - nudges must move the
    picture opposite ways), not while tracking: you moving the object away fast looks the same
  - lost = the arm holds still and keeps searching (the whole picture) until Ctrl+C

Failsafes: steps <= 1.5 deg, joint limits, gripper never below 3 cm above the floor,
watchdog (load > 85% x3, 60 C or servo errors, each confirmed by a second read) freezes the arm.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from iro import ROOT

MODEL = ROOT / "models" / "vittrack.onnx"
STEP_MAX = np.radians(1.5)
LAG_MAX = np.radians(4.0)          # a command may lead the measured joint by this much
GAIN = 0.25
DEAD_IN, DEAD_OUT = 22, 10        # px: start correcting above DEAD_IN, settle below DEAD_OUT
CONTROL_HZ = 15
FRESH = 0.6                       # s: the arm only moves while the last sighting is this recent
COAST = 1.5                       # s: after this without a sighting the lock counts as lost
LOAD_STOP, TEMP_STOP = 85, 60
PAN, LIFT, ELBOW, WRIST = 0, 1, 2, 3


def new_tracker(frame, box):
    p = cv2.TrackerVit_Params()
    p.net = str(MODEL)
    t = cv2.TrackerVit_create(p)
    x1, y1, x2, y2 = map(int, box)
    t.init(frame, (x1, y1, max(8, x2 - x1), max(8, y2 - y1)))
    return t


def centre(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def area(b):
    return max(1.0, (b[2] - b[0]) * (b[3] - b[1]))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = area(a) + area(b) - ix * iy
    return ix * iy / u if u > 0 else 0.0


class Kalman:
    """Constant-velocity Kalman filter on the box centre (pixels)."""

    def __init__(self, c):
        self.x = np.array([c[0], c[1], 0.0, 0.0])
        self.P = np.diag([20.0, 20.0, 200.0, 200.0])
        self.Q = np.diag([4.0, 4.0, 60.0, 60.0])
        self.R = np.diag([16.0, 16.0])
        self.t = time.time()

    def predict(self):
        dt = min(0.2, time.time() - self.t)
        self.t = time.time()
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x[:2]

    def update(self, c, r=None):
        H = np.eye(2, 4)
        S = H @ self.P @ H.T + (np.diag([r, r]) if r else self.R)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(c) - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P

    def gate_radius(self, lost_s):
        """How far from the prediction a candidate may be: 3 sigma, growing while lost."""
        return float(np.clip(3 * np.sqrt(self.P[0, 0] + self.P[1, 1]), 50, 120) + 60 * lost_s)


class Lock:
    """One locked target: matcher sightings (the authority) + Kalman + VitTrack for the gaps."""

    def __init__(self, rec, frame, box, name=None, pm=None):
        from iro.instant import crop
        self.rec, self.name = rec, name
        if pm is None:                                   # build the patch matcher from this one view
            from iro.lockon import PatchMatcher, sam_outline
            from iro.objects import Refiner
            pm = PatchMatcher()
            pm.add_view(frame, sam_outline(Refiner(), frame, box))
        self.pm = pm
        pm.learn_scene(frame, box)                       # the room around it right now = "not it"
        self.looks = rec.fp([crop(frame, box)])
        self.box = np.asarray(box, float)
        self.kf = Kalman(centre(box))
        self.tracker = new_tracker(frame, box)
        self.last_seen = time.time()
        self.ego = np.zeros(2)                           # total picture shift caused by our own steering
        self.pending, self.why, self.peak, self.window, self.n_ok = None, "", 0.0, None, 0

    def age(self):
        return time.time() - self.last_seen

    def state(self):
        a = self.age()
        return "LOCKED" if a < FRESH else ("coasting" if a < COAST else "LOST")

    def search_window(self, shape):
        """Zoomed search region around the prediction (4:3, 2.5x the box), or None = whole picture."""
        if self.age() > COAST:
            return None
        h, w = shape[:2]
        bw, bh = self.box[2] - self.box[0], self.box[3] - self.box[1]
        ww = float(np.clip(max(2.5 * bw, 2.5 * bh * 4 / 3, 240), 240, w))
        wh = ww * 3 / 4
        cx, cy = np.clip(self.kf.x[:2], [ww / 2, wh / 2], [w - ww / 2, h - wh / 2])
        return np.array([cx - ww / 2, cy - wh / 2, cx + ww / 2, cy + wh / 2])

    def detect(self, frame, window, ego):
        """Background thread: matcher sighting in the window (or everywhere)."""
        return frame, window, ego, self.pm.locate(frame, window=window)

    def take(self, result):
        """Use a finished sighting. Returns 'ok', 'relock' or None (self.why says why)."""
        frame, window, ego, found = result
        self.window = window
        if found is None:
            self.why, self.pending, self.peak = self.pm.why, None, 0.0
            return None
        box, _, peak = found
        box = box + np.tile(self.ego - ego, 2)           # the picture moved while it was searching
        self.peak = peak
        if window is None:                               # whole-picture search: needs 2 in a row
            if self.pending is None or np.linalg.norm(centre(self.pending) - centre(box)) > 60:
                self.pending, self.why = box, "seen once, waiting for a 2nd sighting"
                return None
            self.pending = None
            self.kf = Kalman(centre(box))
            verdict = "relock"
        else:
            ratio = area(box) / area(self.box)
            if not 0.3 < ratio < 3.3:
                self.why = f"size jumped x{ratio:.1f}"
                return None
            self.kf.update(centre(box), r=9.0)
            verdict = "ok"
        self.box, self.last_seen, self.why = box, time.time(), ""
        self.tracker = new_tracker(frame, box)
        self.n_ok += 1
        if self.n_ok % 5 == 0:
            self.pm.learn_negatives()                    # look-alikes in a CONFIRMED picture = "not it"
        return verdict

    def follow(self, frame, pred):
        """VitTrack between sightings: only trusted shortly after a sighting and near the prediction."""
        ok, b = self.tracker.update(frame)
        score = self.tracker.getTrackingScore() if ok else 0.0
        if not ok or score < 0.5 or self.age() > 0.4:
            return None, score
        tb = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]], float)
        if np.linalg.norm(centre(tb) - pred) > 60:
            return None, score
        self.kf.update(centre(tb), r=36.0)
        return tb, score


class Watchdog:
    def __init__(self, arm):
        self.arm, self.t, self.strain = arm, 0.0, 0

    def check(self):
        """None if fine, else the reason to freeze. Temperature / error readings are confirmed
        by a second read: one garbled packet once read 88 C on a 42 C motor."""
        if time.time() - self.t < 0.5:
            return None
        self.t = time.time()
        from iro.hw.servos import read_health
        h = read_health(self.arm.bus)
        if self._bad(h):
            time.sleep(0.05)
            return self._bad(read_health(self.arm.bus))
        worst = max(h.items(), key=lambda kv: kv[1]["load_pct"])
        self.strain = self.strain + 1 if worst[1]["load_pct"] > LOAD_STOP else 0
        if self.strain >= 3:
            return f"{worst[0]} straining at {worst[1]['load_pct']:.0f}% (blocked, or pushing something?)"
        return None

    @staticmethod
    def _bad(h):
        for j, v in h.items():
            if v["errors"]:
                return f"{j} reports {', '.join(v['errors'])}"
            if v["temp_c"] >= TEMP_STOP:
                return f"{j} at {v['temp_c']} C (let it cool: iro arm off, holding the arm)"
        return None


def write_joints(arm, q):
    """Send joint targets after clamping to every limit. Returns the q actually sent, or None."""
    from iro.kinematics import JOINTS
    q = np.clip(q, arm.kin.limits[:, 0], arm.kin.limits[:, 1])
    raw = {}
    for j, a in zip(JOINTS, q):
        c = arm.cal[j]
        raw[j] = int(np.clip(arm.to_raw(j, a), c["range_min"], c["range_max"]))
    if arm.kin.fk(q)[2, 3] < arm.pick["table_z"] + 0.03:
        return None                                    # would point the gripper into the floor
    arm.bus.sync_write("Goal_Position", raw, normalize=False)
    return q


def wiggle_box(cam, seconds=1.5):
    """Box around what MOVES while the camera is still: wiggle the object to select it.
    Works for any object, even plain white ones the finder does not box well."""
    frames, t0 = [], time.time()
    while time.time() - t0 < seconds:
        frames.append(cv2.GaussianBlur(cv2.cvtColor(cam.latest(), cv2.COLOR_BGR2GRAY), (5, 5), 0))
    if len(frames) < 5:
        return None
    motion = np.zeros(frames[0].shape, np.float32)
    for a, b in zip(frames, frames[1:]):
        motion += cv2.absdiff(a, b) > 20
    mask = (motion > max(2, 0.15 * len(frames))).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    H, W = mask.shape
    if w * h < 0.005 * H * W or w * h > 0.6 * H * W:
        return None
    return (x, y, x + w, y + h)


def choose_target(rec, cam, input_fn=input):
    """Let the person pick the target and SEE what was picked before tracking starts."""
    from iro.instant import centre_thing
    from iro.objects import draw_box
    while True:
        cam.status = "WIGGLE the object in front of the camera (keep the camera still)"
        input_fn("Hold the object in front of the wrist camera. Press Enter, then WIGGLE it gently for 2 s. ")
        box = wiggle_box(cam)
        f = cam.latest()
        how = "wiggle"
        if box is None:                                   # no clear motion: fall back to the middle thing
            box = centre_thing(rec.things(f), f.shape)
            how = "middle of the picture"
        if box is None:
            print("  Could not find it. Try again, wiggle a little more (but keep it in view).")
            continue
        # snap to the finder's tighter box if one sits on the wiggle area
        cands = [b for b in rec.things(f) if iou(b, box) > 0.3 and area(b) < 1.5 * area(box)]
        if cands:
            box = max(cands, key=lambda b: iou(b, box))
        shot = draw_box(f, box, (0, 255, 0), "target?")
        cam.flash(shot, "Is the green box on your object?  Enter = yes,  r = retry", 30)
        a = input_fn(f"  Picked by {how}. Is the GREEN box on your object (live feed)? Enter = yes, r = retry: ")
        cam.hold_until = 0
        if a.strip().lower() != "r":
            return f, box


JCACHE = ROOT / "config" / "track_jacobian.yaml"


def acquire(pm, cam, name, rec_v=None, tries=150):
    """Find the taught object: it must match (and fit its size/shape/brightness) in 3 frames in a
    row at the same spot, then the person confirms it on the live page (or draws the box).
    Returns (frame, box)."""
    from iro.objects import draw_box
    while True:
        cam.status = f"looking for {name}... (show it to the camera)"
        hist = []
        for _ in range(tries):
            f = cam.latest()
            found = pm.locate(f)
            if rec_v:
                rec_v.add(f, pm, None if found is None else found[0], None, ["ACQUIRE", pm.why or f"peak {found[2]:.2f}"])
            if found is None:
                hist = []
                continue
            hist.append((f, found[0]))
            if len(hist) >= 3 and np.linalg.norm(centre(hist[-1][1]) - centre(hist[-3][1])) < 40:
                break
        else:
            f = cam.latest()
            hist = [(f, None)]
        f, box = hist[-1]
        shot = draw_box(f, box, (0, 255, 0), name) if box is not None else f
        if rec_v:
            cv2.imwrite(str(rec_v.dir / "lock_on.jpg"), shot)
        cam.flash(shot, ("Is this the " + name + "?  Enter = yes,  d = draw it,  r = look again") if box is not None
                  else "Not found. d = draw a box around it,  r = look again", 600)
        a = input(f"  {'Found it (live page): Enter = yes, ' if box is not None else 'Not found: '}"
                  "d = draw the box yourself, r = look again: ").strip().lower()
        cam.hold_until = 0
        if a == "d":
            cam.flash(f, "DRAG a box around the " + name, 600)
            box = cam.feed.ask_box(f.shape, timeout=120)
            cam.hold_until = 0
            if box is not None:
                if rec_v:
                    cv2.imwrite(str(rec_v.dir / "lock_on.jpg"), draw_box(f, box, (0, 255, 0), name + " (drawn)"))
                return f, np.asarray(box, float)
        elif a != "r" and box is not None:
            return f, box


def steering(arm, cam, q0, remeasure=False):
    """2x2: picture shift (px) per radian of pan and wrist bend, measured on the whole picture
    (the room does not move, so nothing has to be held still) and cached."""
    import yaml
    from iro.lockon import flow_jacobian
    if JCACHE.exists() and not remeasure:
        return np.array(yaml.safe_load(JCACHE.read_text())["J"])
    print("  measuring the steering (small nudges of pan and wrist)...")
    J, why = flow_jacobian(arm, cam, q0, joints=(PAN, WRIST))
    if J is None:
        print(f"  steering measurement failed: {why}")
        return None
    ok = abs(J[0, 0]) > 1.5 * abs(J[1, 0]) and abs(J[1, 1]) > 1.5 * abs(J[0, 1]) and abs(np.linalg.det(J)) > 1e3
    print(f"  steering: pan moves the picture {np.round(J[:, 0] * np.radians(1), 1)} px/deg, "
          f"wrist {np.round(J[:, 1] * np.radians(1), 1)} px/deg  {'OK' if ok else 'LOOKS WRONG'}")
    if not ok:
        return None
    JCACHE.write_text(yaml.safe_dump({"J": J.round(2).tolist(), "date": time.strftime("%Y-%m-%d %H:%M")}))
    return J


class Recorder:
    """The annotated tracking video (boxes, search window, heat map, why sightings were rejected),
    written into this run's recording (iro/record.py) next to the plain live-page video."""

    def __init__(self, name):
        from iro.record import run
        self.r = run(f"track-{name}")
        self.dir = self.r.dir

    def add(self, frame, pm, box, window, lines, pred=None, vit=None, color=(0, 255, 0)):
        h, w = frame.shape[:2]
        v = frame.copy()
        if window is not None:
            cv2.rectangle(v, tuple(int(x) for x in window[:2]), tuple(int(x) for x in window[2:]), (255, 160, 0), 1)
        if vit is not None:
            cv2.rectangle(v, tuple(int(x) for x in vit[:2]), tuple(int(x) for x in vit[2:]), (255, 255, 0), 1)
        if box is not None:
            cv2.rectangle(v, tuple(int(x) for x in box[:2]), tuple(int(x) for x in box[2:]), color, 2)
        if pred is not None:
            cv2.circle(v, (int(pred[0]), int(pred[1])), 5, (0, 0, 255), -1)
        cv2.drawMarker(v, (w // 2, h // 2), (255, 255, 0), cv2.MARKER_CROSS, 24, 1)
        if pm is not None and pm.last is not None:                  # heat map of the last search, top right
            heat = cv2.applyColorMap(np.uint8(np.clip(pm.last[0] / 0.4, 0, 1) * 255), cv2.COLORMAP_JET)
            v[0:120, w - 160:w] = cv2.resize(heat, (160, 120), interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(v, (w - 160, 0), (w - 1, 119), (255, 255, 255), 1)
        self.r.video("track", v, lines)


def review(which=None):
    """Contact sheets + the tracking summary (states, rejections, joint travel)."""
    from iro.record import RUNS
    from iro.record import review as general
    d = Path(which) if which and Path(which).exists() else ((RUNS / which) if which else (RUNS / "latest").resolve())
    general(str(d))
    log = d / "log.csv"
    if log.exists():
        import csv
        rows = list(csv.DictReader(open(log)))
        if rows:
            states, whys = {}, {}
            for r in rows:
                states[r["state"]] = states.get(r["state"], 0) + 1
                if r["why"]:
                    k = r["why"].split(" (")[0].split(" 0")[0]
                    whys[k] = whys.get(k, 0) + 1
            print(f"  {float(rows[-1]['t']):.1f} s tracked; states: " + ", ".join(f"{k} {100 * v / len(rows):.0f}%" for k, v in states.items()))
            print("  rejections:", ", ".join(f"{k} x{v}" for k, v in sorted(whys.items(), key=lambda kv: -kv[1])[:6]) or "none")
            print(f"  lift {rows[0]['lift_deg']} -> {rows[-1]['lift_deg']} deg, wrist {rows[0]['wrist_deg']} -> "
                  f"{rows[-1]['wrist_deg']} deg, pan {rows[0]['pan_deg']} -> {rows[-1]['pan_deg']} deg")
    return 0


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "teach":
        from iro.lockon import cmd_teach
        return cmd_teach(sys.argv[2])
    if len(sys.argv) > 1 and sys.argv[1] == "review":
        return review(sys.argv[2] if len(sys.argv) > 2 else None)
    p = argparse.ArgumentParser(prog="iro track", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", nargs="?")
    p.add_argument("--remember", metavar="NAME")
    p.add_argument("--measure", action="store_true", help="re-measure the steering (camera mount changed)")
    p.add_argument("--watch", action="store_true", help="vision only: the arm holds still")
    a = p.parse_args()

    from concurrent.futures import ThreadPoolExecutor

    from iro.hw.servos import BusyError
    from iro.instant import Recognizer
    from iro.motion import Arm, MotionError

    rec = Recognizer()
    from iro.objects import open_camera
    cam = open_camera()
    cam.status = "iro track: starting"
    rec_v = Recorder(a.name or a.remember or "object")
    lock = None
    try:
        with Arm() as arm:
            home = np.radians([0.0, 0.0, 0.0, 30.0, 0.0])
            home[4] = arm.level_roll()
            arm.move_joints_repeatable(home)

            # 1. lock on
            if a.name:
                from iro.lockon import PatchMatcher
                pm = PatchMatcher()
                if not pm.load(a.name):
                    print(f"'{a.name}' has not been taught yet: iro track teach {a.name}")
                    return 1
                f, box = acquire(pm, cam, a.name, rec_v)
            else:
                f, box = choose_target(rec, cam)
                pm = None
            lock = Lock(rec, f, box, a.name or a.remember, pm)
            print("Locked on.")
            J = None
            if not a.watch:
                J = steering(arm, cam, arm.joints(), remeasure=a.measure)
                if J is None:
                    print("Steering measurement looked wrong (the picture should shift sideways for pan and "
                          "up/down for the wrist). Point the camera at a textured scene and run with --measure.")
                    return 1
                Jinv = np.linalg.inv(J)
            lock.kf = Kalman(centre(lock.box))
            lock.last_seen = time.time()
            print("Watching only (the arm holds still)." if a.watch else "Tracking. Move it slowly at first.",
                  "Ctrl+C to stop.")

            # 2. follow
            pool, job = ThreadPoolExecutor(1), None
            log = open(rec_v.dir / "log.csv", "w")
            log.write("t,state,seen_age,peak,why,box_w,box_h,cx,cy,err_x,err_y,gain,pan_deg,lift_deg,wrist_deg,vit_score\n")
            t_start = time.time()
            q_cmd = home.copy()                 # steer from the COMMANDED pose, never the gravity-sagged one
            q_meas = arm.joints()
            dog = Watchdog(arm)
            err_f, moving, gain, last_sign = np.zeros(2), False, GAIN, np.zeros(2)
            t_ctrl, t_lost_msg = 0.0, 0.0
            while True:
                reason = dog.check()
                if reason:
                    raise MotionError(f"failsafe: {reason}")
                f = cam.latest()
                h, w = f.shape[:2]
                pred = lock.kf.predict()

                # sightings (background thread) are the authority
                if job is not None and job.done():
                    verdict = lock.take(job.result())
                    if verdict == "relock":
                        print("  found it again")
                    job = None
                if job is None:
                    job = pool.submit(lock.detect, f.copy(), lock.search_window(f.shape), lock.ego.copy())
                vit, vit_score = lock.follow(f, pred)
                state = lock.state()

                # control at CONTROL_HZ, only on a fresh sighting
                if time.time() - t_ctrl >= 1 / CONTROL_HZ:
                    t_ctrl = time.time()
                    q_meas = arm.joints()
                    if state == "LOCKED":
                        err = lock.kf.x[:2] - np.array([w / 2, h / 2])
                        err_f = 0.6 * err_f + 0.4 * err                       # low-pass
                        size = np.linalg.norm(err_f)
                        moving = size > (DEAD_OUT if moving else DEAD_IN)      # hysteresis
                        sign = np.sign(err_f)
                        flips = np.sum((sign != last_sign) & (np.abs(err_f) > DEAD_OUT) & (last_sign != 0))
                        gain = max(0.08, gain * 0.5) if flips else min(GAIN, gain * 1.1)   # overshoot: back off
                        last_sign = sign
                        if moving and J is not None:
                            dq = np.clip(-gain * Jinv @ err_f, -STEP_MAX, STEP_MAX)
                            new = q_cmd.copy()
                            for k, j in enumerate((PAN, WRIST)):
                                new[j] = np.clip(new[j] + dq[k], q_meas[j] - LAG_MAX, q_meas[j] + LAG_MAX)
                            sent = write_joints(arm, new)
                            if sent is not None:
                                shift = J @ (sent - q_cmd)[[PAN, WRIST]]     # the picture will shift by this
                                lock.kf.x[:2] += shift
                                lock.ego += shift
                                q_cmd = sent
                    else:
                        err_f, moving = np.zeros(2), False

                bw, bh = lock.box[2] - lock.box[0], lock.box[3] - lock.box[1]
                log.write(f"{time.time() - t_start:.3f},{state},{lock.age():.2f},{lock.peak:.2f},{lock.why.replace(',', ';')},"
                          f"{bw:.0f},{bh:.0f},{lock.kf.x[0]:.0f},{lock.kf.x[1]:.0f},{err_f[0]:.0f},{err_f[1]:.0f},{gain:.2f},"
                          f"{np.degrees(q_meas[PAN]):.1f},{np.degrees(q_meas[LIFT]):.1f},{np.degrees(q_meas[WRIST]):.1f},{vit_score:.2f}\n")
                color = {"LOCKED": (0, 255, 0), "coasting": (0, 220, 255), "LOST": (0, 0, 255)}[state]
                lines = [f"{state}  seen {lock.age():.1f}s ago  peak {lock.peak:.2f}  gain {gain:.2f}",
                         f"pan {np.degrees(q_meas[PAN]):.1f}  lift {np.degrees(q_meas[LIFT]):.1f}  "
                         f"wrist {np.degrees(q_meas[WRIST]):.1f}  err {err_f[0]:.0f},{err_f[1]:.0f}",
                         lock.why or ""]
                rec_v.add(f, lock.pm, lock.box, lock.window, lines, pred, vit, color)
                if state == "LOST" and time.time() - t_lost_msg > 5:   # keeps searching until Ctrl+C
                    t_lost_msg = time.time()
                    print(f"  lost for {lock.age():.0f} s, still searching (arm holds): {lock.why or 'nothing seen'}")
                cam.flash(live_view(f, lock.box, color, lock.name or "target", w, h), f"{state} | peak {lock.peak:.2f}", 0.2)
    except KeyboardInterrupt:
        print("\nStopped: the arm holds its pose (`iro arm off` = limp).")
    except (MotionError, BusyError) as e:
        print(f"STOPPED: {e}. The arm holds its pose.")
    finally:
        cam.release()
        if a.remember and lock is not None:
            rec.memory.add(a.remember.replace(" ", "_"), lock.looks)
            lock.pm.save(a.remember.replace(" ", "_"))
            print(f"Remembered '{a.remember}' ({len(lock.pm.fg)} object patches): iro track {a.remember}")
    return 0


def live_view(f, box, color, label, w, h):
    from iro.objects import draw_box
    vis = draw_box(f, box, color, label)
    cv2.drawMarker(vis, (w // 2, h // 2), (255, 255, 0), cv2.MARKER_CROSS, 24, 2)
    return vis


if __name__ == "__main__":
    sys.exit(main())
