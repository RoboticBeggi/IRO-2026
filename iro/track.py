"""iro track: the arm follows an object with the wrist camera, like a head turning to watch.

    iro track                  hold anything (sponge, hand, toy) in the middle of the wrist camera's
                               view, press Enter: the arm follows it
    iro track Tapak_Sirih      follow a remembered object (iro object show / iro efad show)
    iro track --remember NAME  follow it AND save it to the instant memory as NAME (no training)

Watch it live at http://localhost:8080. Ctrl+C = stop and hold the pose.

How (from visual-servoing practice): VitTrack follows the object at camera speed; every few frames
EFAD re-finds it (YOLOE + DINOv2 fingerprints of THIS object) to correct drift and re-catch it after
it is lost. The arm moves pan (left/right) and wrist bend (up/down) to keep it centred; how each
joint moves the picture is measured with two small test nudges at the start, so the camera mount
does not matter.

Failsafes: steps of at most 2.5 deg, joint limits from the calibration, the gripper never points
lower than 3 cm above the floor, and a watchdog twice a second: a joint straining (load > 85%),
too hot (60 C) or reporting an error freezes the arm in place.
"""

import argparse
import sys
import time

import cv2
import numpy as np

from iro import ROOT

MODEL = ROOT / "models" / "vittrack.onnx"
STEP_MAX = np.radians(2.5)     # largest move per control step
GAIN = 0.35                    # fraction of the error corrected per step (smooth, no overshoot)
DEADBAND = 14                  # pixels: close enough, do not twitch
REDETECT_EVERY = 8             # frames
LOAD_STOP, TEMP_STOP = 85, 60
PAN, LIFT, ELBOW, WRIST = 0, 1, 2, 3


def new_tracker(frame, box):
    p = cv2.TrackerVit_Params()
    p.net = str(MODEL)
    t = cv2.TrackerVit_create(p)
    x1, y1, x2, y2 = map(int, box)
    t.init(frame, (x1, y1, max(8, x2 - x1), max(8, y2 - y1)))
    return t


class Target:
    """What we follow: a box, a fast tracker, and fingerprints of this very object."""

    def __init__(self, rec, frame, box, name=None):
        from iro.instant import crop
        self.rec, self.name, self.crop = rec, name, crop
        self.fps = rec.fp([crop(frame, box)])
        self.reset(frame, box)

    def reset(self, frame, box):
        self.box = np.asarray(box, float)
        self.tracker = new_tracker(frame, box)
        self.lost_since = None

    def learn(self, frame):
        """Add a fingerprint of the current view (the object seen from new angles)."""
        self.fps = np.vstack([self.fps, self.rec.fp([self.crop(frame, self.box)])])[-60:]

    def redetect(self, frame, thr=0.55):
        """Find it again among everything in view: best fingerprint match (or memory name)."""
        boxes = self.rec.things(frame)
        if not boxes:
            return None
        e = self.rec.fp([self.crop(frame, b) for b in boxes])
        sim = (e @ self.fps.T).max(1)
        if self.name and self.rec.memory.objects.get(self.name) is not None:
            sim = np.maximum(sim, (e @ self.rec.memory.objects[self.name].T).max(1))
        i = int(np.argmax(sim))
        return (boxes[i], float(sim[i])) if sim[i] >= thr else None


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
        r = arm.to_raw(j, a)
        c = arm.cal[j]
        raw[j] = int(np.clip(r, c["range_min"], c["range_max"]))
    if arm.kin.fk(q)[2, 3] < arm.pick["table_z"] + 0.03:
        return None                                    # would point the gripper into the floor
    arm.bus.sync_write("Goal_Position", raw, normalize=False)
    return q


def measure_jacobian(arm, cam, target, q0):
    """Pixels the object moves per radian of pan and of wrist bend (2x2), from two small nudges."""
    from iro.objects import grab
    J = np.zeros((2, 2))
    centre = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])   # noqa: E731
    for k, joint in enumerate((PAN, WRIST)):
        pts = []
        for sign in (+1, -1):
            q = q0.copy()
            q[joint] += sign * np.radians(5)
            arm.move_joints(q, duration=0.6)
            time.sleep(0.3)
            f = grab(cam)
            ok, box = target.tracker.update(f)
            if not ok:
                return None
            x, y, w, h = box
            pts.append(centre((x, y, x + w, y + h)))
        J[:, k] = (pts[0] - pts[1]) / np.radians(10)
        arm.move_joints(q0, duration=0.6)
        time.sleep(0.3)
        f = grab(cam)
        ok, box = target.tracker.update(f)
    return J if abs(np.linalg.det(J)) > 1e3 else None


def main():
    p = argparse.ArgumentParser(prog="iro track", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", nargs="?")
    p.add_argument("--remember", metavar="NAME")
    a = p.parse_args()

    from iro.hw.servos import BusyError
    from iro.instant import Recognizer, centre_thing
    from iro.motion import Arm, MotionError
    from iro.objects import draw_box, grab, open_camera

    rec = Recognizer()
    cam = open_camera()
    cam.status = "iro track: starting"
    try:
        with Arm() as arm:
            # watch pose: forearm level, wrist tilted 30 deg down: sees hands in front AND the floor
            home = np.radians([0.0, 0.0, 0.0, 30.0, 0.0])
            home[4] = arm.level_roll()
            arm.move_joints_repeatable(home)

            # 1. pick the target
            if a.name:
                cam.status = f"looking for {a.name}"
                target = None
                for _ in range(20):
                    f = grab(cam)
                    found = rec.recognize(f)
                    hit = [d for d in found if d[0] == a.name]
                    if hit:
                        target = Target(rec, f, hit[0][3], a.name)
                        break
                if target is None:
                    print(f"Cannot see {a.name}. Show it to the wrist camera (live feed) and try again.")
                    return 1
            else:
                cam.status = "hold the object in the MIDDLE of the picture, then press Enter"
                input("Hold the object (sponge, hand, anything) in the middle of the wrist camera view "
                      "(live feed), then press Enter. ")
                f = grab(cam)
                box = centre_thing(rec.things(f), f.shape)
                if box is None:                            # nothing found there: take the middle patch
                    h, w = f.shape[:2]
                    box = (w / 2 - 60, h / 2 - 60, w / 2 + 60, h / 2 + 60)
                target = Target(rec, f, box, a.remember)
            print("Got it. Hold it still for 3 s while the arm measures how it moves...")
            cam.status = "hold still: measuring"
            J = measure_jacobian(arm, cam, target, arm.joints())
            if J is None:
                print("Lost it while measuring. Keep it still and in view, and try again.")
                return 1
            Jinv = np.linalg.inv(J)
            print("Tracking. Move it around (slowly at first). Ctrl+C to stop.")

            # 2. follow
            from concurrent.futures import ThreadPoolExecutor
            pool, pending = ThreadPoolExecutor(1), None     # re-finding runs beside the control loop
            dog = Watchdog(arm)
            q = arm.joints()
            smooth, n, nf, t_fps, fps = None, 0, 0, time.time(), 0.0
            while True:
                reason = dog.check()
                if reason:
                    raise MotionError(f"failsafe: {reason}")
                f = cam.latest()
                h, w = f.shape[:2]
                ok, b = target.tracker.update(f)
                score = target.tracker.getTrackingScore() if ok else 0.0
                if ok and score > 0.3:
                    x, y, bw, bh = b
                    target.box = np.array([x, y, x + bw, y + bh], float)
                    target.lost_since = None
                elif target.lost_since is None:
                    target.lost_since = time.time()
                n += 1
                nf += 1
                if pending is None and (n % REDETECT_EVERY == 0 or target.lost_since):
                    pending = (pool.submit(target.redetect, f.copy()), f.copy())
                if pending is not None and pending[0].done():
                    found, f_found = pending[0].result(), pending[1]
                    pending = None
                    if found:
                        box, sim = found
                        iou_ok = not target.lost_since and _iou(box, target.box) > 0.3
                        if not iou_ok:
                            target.reset(f_found, box)     # drifted or lost: snap back to the real thing
                        elif score > 0.5 and n % (REDETECT_EVERY * 4) < REDETECT_EVERY:
                            target.learn(f_found)          # remember this angle too
                lost = target.lost_since is not None
                c = np.array([(target.box[0] + target.box[2]) / 2, (target.box[1] + target.box[3]) / 2])
                smooth = c if smooth is None else 0.5 * smooth + 0.5 * c
                err = smooth - np.array([w / 2, h / 2])
                if not lost and np.linalg.norm(err) > DEADBAND:
                    dq2 = np.clip(-GAIN * Jinv @ err, -STEP_MAX, STEP_MAX)
                    q_new = q.copy()
                    q_new[PAN] += dq2[0]
                    q_new[WRIST] += dq2[1]
                    sent = write_joints(arm, q_new)
                    if sent is not None:
                        q = sent
                if time.time() - t_fps >= 1:
                    fps, nf, t_fps = nf / (time.time() - t_fps), 0, time.time()
                vis = draw_box(f, target.box, (0, 0, 255) if lost else (0, 255, 0), a.remember or a.name or "target")
                cv2.drawMarker(vis, (w // 2, h // 2), (255, 255, 0), cv2.MARKER_CROSS, 24, 2)
                state = "LOST - searching" if lost else f"tracking  pan {np.degrees(q[PAN]):+.0f}  wrist {np.degrees(q[WRIST]):+.0f}"
                cam.flash(vis, f"{state} | {fps:.0f} fps", 0.2)
                if lost and time.time() - target.lost_since > 10:
                    print("Lost for 10 s: stopping (arm holds).")
                    break
    except KeyboardInterrupt:
        print("\nStopped: the arm holds its pose (`iro arm off` = limp).")
    except (MotionError, BusyError) as e:
        print(f"STOPPED: {e}. The arm holds its pose.")
    finally:
        cam.release()
        if a.remember and "target" in locals() and target is not None:
            rec.memory.add(a.remember.replace(" ", "_"), target.fps)
            print(f"Remembered {len(target.fps)} views as '{a.remember}': try `iro track {a.remember}`.")
    return 0


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


if __name__ == "__main__":
    sys.exit(main())
