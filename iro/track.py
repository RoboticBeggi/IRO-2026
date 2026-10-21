"""iro track: the arm follows ONE object with the wrist camera and stays locked on it.

    iro track                  hold anything in the middle of the wrist camera view, press Enter
    iro track Tapak_Sirih      start on a remembered object (iro object show / iro efad show)
    iro track --remember NAME  follow it AND save it to the instant memory as NAME (no training)

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

    def __init__(self, rec, frame, box, name=None):
        from iro.instant import crop
        self.rec, self.name, self.crop = rec, name, crop
        self.looks = rec.fp([crop(frame, box)])
        others = [b for b in rec.things(frame) if iou(b, box) < 0.1]
        self.distract = rec.fp([crop(frame, b) for b in others]) if others else np.zeros((0, 384), np.float32)
        if len(rec.memory.background):                      # the room, as learnt before
            self.distract = np.vstack([self.distract, rec.memory.background[-500:]])
        self.kf = Kalman(centre(box))
        self.pending = None                                  # candidate waiting for a 2nd confirmation
        self.reset(frame, box)

    def reset(self, frame, box):
        self.box = np.asarray(box, float)
        self.tracker = new_tracker(frame, box)
        self.lost_since = None

    def appearance(self, e):
        """(similarity to this object, margin over the best distractor) per fingerprint."""
        sim = (e @ self.looks.T).max(1)
        if self.name is not None and self.rec.memory.objects.get(self.name) is not None:
            sim = np.maximum(sim, (e @ self.rec.memory.objects[self.name].T).max(1))
        neg = (e @ self.distract.T).max(1) if len(self.distract) else np.zeros(len(e))
        return sim, sim - neg

    def refind(self, frame):
        """Candidates in view with their fingerprints (runs in a background thread)."""
        boxes = self.rec.things(frame)
        return frame, boxes, (self.rec.fp([self.crop(frame, b) for b in boxes]) if boxes else None)

    def judge(self, result, track_ok):
        """Apply the gates to a re-find. Returns 'confirm', 'switch', or None."""
        frame, boxes, e = result
        if not boxes:
            return None
        sim, margin = self.appearance(e)
        lost_s = time.time() - self.lost_since if self.lost_since else 0.0
        pred, radius = self.kf.x[:2], self.kf.gate_radius(lost_s)
        ok = [i for i, b in enumerate(boxes)
              if np.linalg.norm(centre(b) - pred) < radius
              and 0.5 < area(b) / area(self.box) < 2.0
              and sim[i] > 0.6 and margin[i] > (0.12 if lost_s else 0.06)]
        # everything that looked like it but failed the gates is a distractor: remember it
        near_miss = [i for i in range(len(boxes)) if i not in ok and sim[i] > 0.55 and iou(boxes[i], self.box) < 0.1]
        if near_miss:
            self.distract = np.vstack([self.distract, e[near_miss]])[-1500:]
        if not ok:
            self.pending = None
            return None
        i = max(ok, key=lambda k: margin[k])
        b = boxes[i]
        if track_ok and iou(b, self.box) > 0.4:
            self.pending = None
            if margin[i] > 0.1:
                self.looks = np.vstack([self.looks, e[i:i + 1]])[-80:]   # tracker and EFAD agree: learn
            return "confirm"
        # disagreement or lost: need the same candidate twice in a row before jumping to it
        if self.pending is not None and iou(self.pending, b) > 0.3:
            self.pending = None
            self.reset(frame, b)
            self.kf.update(centre(b))
            return "switch"
        self.pending = b
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


def measure_jacobian(arm, cam, lock, q0):
    """Pixels the object moves per radian of pan and of wrist bend (2x2), from small nudges."""
    from iro.objects import grab
    J = np.zeros((2, 2))
    for k, joint in enumerate((PAN, WRIST)):
        pts = []
        for sign in (+1, -1):
            q = q0.copy()
            q[joint] += sign * np.radians(5)
            arm.move_joints(q, duration=0.6)
            time.sleep(0.3)
            ok, b = lock.tracker.update(grab(cam))
            if not ok:
                return None
            pts.append(np.array([b[0] + b[2] / 2, b[1] + b[3] / 2]))
        J[:, k] = (pts[0] - pts[1]) / np.radians(10)
        arm.move_joints(q0, duration=0.6)
        time.sleep(0.3)
        ok, b = lock.tracker.update(grab(cam))
        if ok:
            lock.box = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]], float)
            lock.kf = Kalman(centre(lock.box))
    return J if abs(np.linalg.det(J)) > 1e3 else None


def main():
    p = argparse.ArgumentParser(prog="iro track", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", nargs="?")
    p.add_argument("--remember", metavar="NAME")
    a = p.parse_args()

    from concurrent.futures import ThreadPoolExecutor

    from iro.hw.servos import BusyError
    from iro.instant import Recognizer, centre_thing
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
                cam.status = f"looking for {a.name}"
                for _ in range(20):
                    f = grab(cam)
                    hit = [d for d in rec.recognize(f) if d[0] == a.name]
                    if hit:
                        lock = Lock(rec, f, hit[0][3], a.name)
                        break
                if lock is None:
                    print(f"Cannot see {a.name}. Show it to the wrist camera (live feed) and try again.")
                    return 1
            else:
                cam.status = "hold the object in the MIDDLE of the picture, then press Enter"
                input("Hold the object in the middle of the wrist camera view (live feed), then press Enter. ")
                f = grab(cam)
                box = centre_thing(rec.things(f), f.shape)
                if box is None:
                    h, w = f.shape[:2]
                    box = (w / 2 - 60, h / 2 - 60, w / 2 + 60, h / 2 + 60)
                lock = Lock(rec, f, box, a.remember)
            print(f"Locked ({len(lock.distract)} distractors noted). Hold it still ~3 s while the arm measures...")
            cam.status = "hold still: measuring"
            J = measure_jacobian(arm, cam, lock, arm.joints())
            if J is None:
                print("Lost it while measuring. Keep it still and in view, and try again.")
                return 1
            Jinv = np.linalg.inv(J)
            print("Tracking. Move it slowly at first. Ctrl+C to stop.")

            # 2. follow
            pool, job = ThreadPoolExecutor(1), None
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
                track_ok = ok and score > 0.35 and np.linalg.norm(centre(tb) - pred) < lock.kf.gate_radius(0) * 1.5
                if track_ok:
                    lock.box = tb
                    lock.kf.update(centre(tb))
                    lock.lost_since = None
                elif lock.lost_since is None:
                    lock.lost_since = time.time()
                if job is None and (n % REDETECT_EVERY == 0 or lock.lost_since):
                    job = pool.submit(lock.refind, f.copy())
                if job is not None and job.done():
                    if lock.judge(job.result(), track_ok) == "switch":
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
                            write_joints(arm, q_cmd)
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
            if len(lock.distract):
                rec.memory.add_background(lock.distract[-300:])
            print(f"Remembered {len(lock.looks)} views as '{a.remember}' (+ room views as 'not this').")
    return 0


if __name__ == "__main__":
    sys.exit(main())
