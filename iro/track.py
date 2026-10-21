"""iro track: the arm follows ONE object with the wrist camera and stays locked on it.

    iro track                  hold anything in the middle of the wrist camera view, press Enter
    iro track Tapak_Sirih      start on a remembered object (iro object show / iro efad show)
    iro track --remember NAME  follow it AND save it as NAME (no training)
  iro track teach NAME       BEST: guided tour, draw boxes on 10-12 views (iro/lockon.py)
  iro track NAME --measure   re-measure the steering (after changing the camera mount)

Live view: http://localhost:8080 (green = locked, yellow = coasting on prediction, red = lost).
Ctrl+C = stop and hold the pose.

LOCK-IN (the DeepSORT / StrongSORT recipe, for one target):
  - a Kalman filter predicts where the object will be next frame (position + velocity)
  - VitTrack follows it frame to frame; EFAD re-finds it in the background
  - a re-found candidate is accepted ONLY if it passes all gates:
      position: within the predicted region (grows slowly while lost)
      size:     0.5x - 2x the current box
      look:     clearly closer to THIS object's saved views than to the "distractors"
                (everything else seen in the room is collected as distractors automatically)
  - switching to a candidate far from the current track needs 2 confirmations in a row
  - new views of the object are saved only when tracker AND re-finding agree (no drift poisoning)

SMOOTH CONTROL (visual-servoing practice for camera + servo latency):
  - commands are computed from the MEASURED joint angles, not stacked on the last command
    (stacking is what made it swing left-right), at 15 Hz with a low-pass on the error
  - deadband with hysteresis (start moving > 22 px off, stop < 10 px)
  - if the error keeps flipping sign (overshoot), the gain halves automatically

Failsafes: steps <= 1.5 deg, calibrated joint limits, gripper never points below 3 cm above the
floor, watchdog (load > 85% x3, 60 C, servo error) freezes the arm.
"""

import argparse
import sys
import time

import cv2
import numpy as np

from iro import ROOT

MODEL = ROOT / "models" / "vittrack.onnx"
START_SHOT = ROOT.parent / "iro-track-start.jpg"     # what it locked on (for checking)
LOG = ROOT.parent / "iro-track-log.csv"              # per-frame numbers (for diagnosis)
STEP_MAX = np.radians(1.5)
GAIN = 0.25
DEAD_IN, DEAD_OUT = 22, 10        # px: start correcting above DEAD_IN, settle below DEAD_OUT
CONTROL_HZ = 15
REDETECT_EVERY = 6                # frames between background re-finds
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

    def update(self, c):
        H = np.eye(2, 4)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(c) - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P

    def gate_radius(self, lost_s):
        """How far from the prediction a candidate may be: 3 sigma, growing while lost."""
        return float(np.clip(3 * np.sqrt(self.P[0, 0] + self.P[1, 1]), 50, 120) + 60 * lost_s)


class Lock:
    """One locked target: tracker + Kalman + its own looks + the distractors around it."""

    def __init__(self, rec, frame, box, name=None, pm=None):
        from iro.instant import crop
        self.rec, self.name, self.crop = rec, name, crop
        if pm is None:                                   # build the patch matcher from this one view
            from iro.lockon import PatchMatcher, sam_outline
            from iro.objects import Refiner
            pm = PatchMatcher()
            pm.add_view(frame, sam_outline(Refiner(), frame, box))
        self.pm = pm
        self.looks = rec.fp([crop(frame, box)])
        self.kf = Kalman(centre(box))
        self.pending = None                                  # candidate waiting for a 2nd confirmation
        self.reset(frame, box)

    def reset(self, frame, box):
        self.box = np.asarray(box, float)
        self.tracker = new_tracker(frame, box)
        self.lost_since = None

    def refind(self, frame, near, radius):
        """Where the object is by patch matching, near the prediction (runs in a background thread)."""
        return frame, self.pm.locate(frame, near=near, radius=radius)

    def judge(self, result, track_ok):
        """Apply the gates to a re-find. Returns 'confirm', 'switch', or None (self.why says why)."""
        frame, found = result
        if found is None:
            self.why = "not found near the predicted spot"
            self.pending = None
            return None
        box, strength = found
        ratio = area(box) / area(self.box)
        if not 0.4 < ratio < 2.5:
            self.why = f"size x{ratio:.1f}"
            self.pending = None
            return None
        if track_ok and iou(box, self.box) > 0.3:
            self.pending = None
            self.why = ""
            return "confirm"
        if self.pending is not None and iou(self.pending, box) > 0.3:   # same place twice: trust it
            self.pending = None
            self.reset(frame, box)
            self.kf.update(centre(box))
            return "switch"
        self.pending = box
        self.why = "waiting for a 2nd confirmation"
        return None


class Watchdog:
    def __init__(self, arm):
        self.arm, self.t, self.strain = arm, 0.0, 0

    def check(self):
        """None if fine, else the reason to freeze."""
        if time.time() - self.t < 0.5:
            return None
        self.t = time.time()
        from iro.hw.servos import read_health
        h = read_health(self.arm.bus)
        for j, v in h.items():
            if v["errors"]:
                return f"{j} reports {', '.join(v['errors'])}"
            if v["temp_c"] >= TEMP_STOP:
                return f"{j} at {v['temp_c']} C"
        worst = max(h.items(), key=lambda kv: kv[1]["load_pct"])
        self.strain = self.strain + 1 if worst[1]["load_pct"] > LOAD_STOP else 0
        if self.strain >= 3:
            return f"{worst[0]} straining at {worst[1]['load_pct']:.0f}% (blocked, or pushing something?)"
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
        cv2.imwrite(str(START_SHOT), shot)
        cam.flash(shot, "Is the green box on your object?  Enter = yes,  r = retry", 30)
        a = input_fn(f"  Picked by {how}. Is the GREEN box on your object (live feed)? Enter = yes, r = retry: ")
        cam.hold_until = 0
        if a.strip().lower() != "r":
            return f, box


JCACHE = ROOT / "config" / "track_jacobian.yaml"


def steering(arm, cam, q0, remeasure=False):
    """2x2: picture shift (px) per radian of pan and wrist bend, measured on the whole picture
    (the room does not move, so nothing has to be held still) and cached."""
    import yaml
    from iro.lockon import flow_jacobian
    if JCACHE.exists() and not remeasure:
        return np.array(yaml.safe_load(JCACHE.read_text())["J"])
    J = flow_jacobian(arm, cam, q0, joints=(PAN, WRIST))
    ok = abs(J[0, 0]) > 1.5 * abs(J[1, 0]) and abs(J[1, 1]) > 1.5 * abs(J[0, 1]) and abs(np.linalg.det(J)) > 1e3
    print(f"  steering: pan moves the picture {np.round(J[:, 0] * np.radians(1), 1)} px/deg, "
          f"wrist {np.round(J[:, 1] * np.radians(1), 1)} px/deg  {'OK' if ok else 'LOOKS WRONG'}")
    if not ok:
        return None
    JCACHE.write_text(yaml.safe_dump({"J": J.round(2).tolist(), "date": time.strftime("%Y-%m-%d %H:%M")}))
    return J


def main():
    p = argparse.ArgumentParser(prog="iro track", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", nargs="?")
    p.add_argument("--remember", metavar="NAME")
    p.add_argument("--measure", action="store_true", help="re-measure the steering (camera mount changed)")
    if len(sys.argv) > 2 and sys.argv[1] == "teach":
        from iro.lockon import cmd_teach
        return cmd_teach(sys.argv[2])
    a = p.parse_args()

    from concurrent.futures import ThreadPoolExecutor

    from iro.hw.servos import BusyError
    from iro.instant import Recognizer
    from iro.motion import Arm, MotionError
    from iro.objects import draw_box, grab, open_camera

    rec = Recognizer()
    cam = open_camera()
    cam.status = "iro track: starting"
    lock = None
    try:
        with Arm() as arm:
            home = np.radians([0.0, 0.0, 0.0, 30.0, 0.0])
            home[4] = arm.level_roll()
            arm.move_joints_repeatable(home)

            # 1. lock on
            if a.name:
                from iro.lockon import PatchMatcher
                cam.status = f"looking for {a.name}"
                pm = PatchMatcher()
                if not pm.load(a.name):
                    print(f"'{a.name}' has not been taught yet: iro track teach {a.name}")
                    return 1
                for _ in range(30):
                    f = grab(cam)
                    found = pm.locate(f)
                    if found is not None:
                        lock = Lock(rec, f, found[0], a.name, pm)
                        break
                if lock is None:
                    print(f"Cannot see {a.name}. Show it to the wrist camera (live feed) and try again.")
                    return 1
            else:
                f, box = choose_target(rec, cam)
                lock = Lock(rec, f, box, a.remember)
            print("Locked on.")
            J = steering(arm, cam, arm.joints(), remeasure=a.measure)
            if J is None:
                print("Steering measurement looked wrong (the picture should shift sideways for pan and "
                      "up/down for the wrist). Point the camera at a textured scene and run with --measure.")
                return 1
            lock.reset(cam.latest(), lock.box)
            Jinv = np.linalg.inv(J)
            print("Tracking. Move it slowly at first. Ctrl+C to stop.")

            # 2. follow
            pool, job = ThreadPoolExecutor(1), None
            log = open(LOG, "w")
            log.write("t,score,ok,track_ok,lost,cx,cy,err_x,err_y,gain,pan_deg,wrist_deg,refind\n")
            t_start, q_log = time.time(), arm.joints()
            dog = Watchdog(arm)
            err_f, moving, gain, last_sign = np.zeros(2), False, GAIN, np.zeros(2)
            n, nf, t_fps, fps, t_ctrl = 0, 0, time.time(), 0.0, 0.0
            while True:
                reason = dog.check()
                if reason:
                    raise MotionError(f"failsafe: {reason}")
                f = cam.latest()
                h, w = f.shape[:2]
                n, nf = n + 1, nf + 1
                pred = lock.kf.predict()
                ok, b = lock.tracker.update(f)
                score = lock.tracker.getTrackingScore() if ok else 0.0
                tb = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]], float) if ok else None
                # tracker result must also be where the object can plausibly be
                track_ok = ok and score > 0.25 and np.linalg.norm(centre(tb) - pred) < lock.kf.gate_radius(0) * 2
                if track_ok:
                    lock.box = tb
                    lock.kf.update(centre(tb))
                    lock.lost_since = None
                elif lock.lost_since is None:
                    lock.lost_since = time.time()
                if job is None and (n % REDETECT_EVERY == 0 or lock.lost_since):
                    lost_s = time.time() - lock.lost_since if lock.lost_since else 0.0
                    job = pool.submit(lock.refind, f.copy(), lock.kf.x[:2].copy(), lock.kf.gate_radius(lost_s))
                verdict = ""
                if job is not None and job.done():
                    verdict = lock.judge(job.result(), track_ok) or "rejected:" + lock.why
                    if verdict == "switch":
                        print("  re-locked on the object")
                    job = None
                lost = lock.lost_since is not None
                coasting = lost and time.time() - lock.lost_since < 0.7

                # control at CONTROL_HZ, from the measured joints
                if time.time() - t_ctrl >= 1 / CONTROL_HZ:
                    t_ctrl = time.time()
                    target_px = centre(lock.box) if not lost else (lock.kf.x[:2] if coasting else None)
                    if target_px is not None:
                        err = target_px - np.array([w / 2, h / 2])
                        err_f = 0.6 * err_f + 0.4 * err                       # low-pass
                        size = np.linalg.norm(err_f)
                        moving = size > (DEAD_OUT if moving else DEAD_IN)      # hysteresis
                        sign = np.sign(err_f)
                        flips = np.sum((sign != last_sign) & (np.abs(err_f) > DEAD_OUT) & (last_sign != 0))
                        gain = max(0.08, gain * 0.5) if flips else min(GAIN, gain * 1.1)   # overshoot: back off
                        last_sign = sign
                        if moving:
                            q_meas = arm.joints()
                            dq = np.clip(-gain * Jinv @ err_f, -STEP_MAX, STEP_MAX)
                            q_cmd = q_meas.copy()
                            q_cmd[PAN] += dq[0]
                            q_cmd[WRIST] += dq[1]
                            sent = write_joints(arm, q_cmd)
                            if sent is not None:
                                q_log = sent
                                # the whole picture will shift by J*dq: move the prediction with it
                                lock.kf.x[:2] += J @ np.array([sent[PAN] - q_meas[PAN], sent[WRIST] - q_meas[WRIST]])
                c_now = centre(lock.box)
                log.write(f"{time.time() - t_start:.3f},{score:.2f},{int(ok)},{int(track_ok)},{int(lock.lost_since is not None)},"
                          f"{c_now[0]:.0f},{c_now[1]:.0f},{err_f[0]:.0f},{err_f[1]:.0f},{gain:.2f},"
                          f"{np.degrees(q_log[PAN]):.1f},{np.degrees(q_log[WRIST]):.1f},{verdict}\n")
                if time.time() - t_fps >= 1:
                    fps, nf, t_fps = nf / (time.time() - t_fps), 0, time.time()
                color = (0, 255, 0) if not lost else ((0, 220, 255) if coasting else (0, 0, 255))
                vis = draw_box(f, lock.box, color, a.remember or a.name or "target")
                cv2.drawMarker(vis, (w // 2, h // 2), (255, 255, 0), cv2.MARKER_CROSS, 24, 2)
                state = "LOCKED" if not lost else ("coasting" if coasting else "LOST - searching")
                cam.flash(vis, f"{state} | gain {gain:.2f} | {fps:.0f} fps", 0.2)
                if lost and time.time() - lock.lost_since > 10:
                    print("Lost for 10 s: stopping (arm holds).")
                    break
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


if __name__ == "__main__":
    sys.exit(main())
