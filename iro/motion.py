"""Safe motion for the SO-101: every move the robot makes goes through here.

    with Arm() as arm:
        arm.move_tip([0.20, 0.0, 0.15])     # gripper straight down above a point
        arm.gripper(open=True)

Safety rules, enforced on every move:
  - refuses to run on a calibration without joint limits (wrapped joints could
    take the long way round and sweep through the robot)
  - torque comes on only after the goal is set to where the arm already is
    (no jump at start)
  - smooth (minimum-jerk) moves, joint speed capped by `max_speed`
  - targets outside the calibrated range or the URDF limits are refused
  - every waypoint's gripper tip must stay above `table_z + clearance`
  - stops and holds if a servo reports an error, overheats or is overloaded
"""

import json
import os
import time

import numpy as np

from iro import ROOT, load_config
from iro.hw.servos import decode_status, open_bus
from iro.kinematics import JOINTS, Kinematics

TICKS = 4096
SEEDS = [np.radians(s) for s in ([0, 0, 0, 0, 0], [0, -30, 60, 60, 0], [0, 30, -30, 80, 0], [0, 0, 45, 90, 0])]
CAL = ROOT / "arm" / "calibration" / "robots" / "so_follower" / f"{os.environ.get('SO101_ID', 'iro_follower')}.json"


SAG_FILE = ROOT / "arm" / "calibration" / "sag_model.json"


class MotionError(RuntimeError):
    pass


class SagModel:
    """Where each joint really stops versus where it is sent. LeRobot runs the servos soft (P 16), so
    gravity and friction leave them 30-40 counts short (20 Oct). Learnt from every move that settled:
    the goal offset (counts) that made joint j arrive = a_j * g_j(q) + b_j * s_j, where g_j is gravity's
    lever on the joint (how much the tip height changes per radian of it) and s_j the direction the
    joint came from (+1/-1, 0 = did not move). Aimed for ahead of time, so most moves arrive without
    correcting; the settle step still checks. Old moves count less (the battery and the servos warm
    up), and it is only used after MIN_N moves. Saved in arm/calibration/ (backed up)."""
    MIN_N, DECAY, CLIP = 8, 0.97, 60

    def __init__(self, kin, path=SAG_FILE):
        self.kin, self.path = kin, path
        self.A = np.zeros((len(JOINTS), 2, 2))
        self.y = np.zeros((len(JOINTS), 2))
        self.n = np.zeros(len(JOINTS), int)
        try:
            d = json.loads(path.read_text())
            self.A, self.y, self.n = np.array(d["A"]), np.array(d["y"]), np.array(d["n"], int)
        except (OSError, ValueError, KeyError):
            pass

    def features(self, q, s):
        q = np.asarray(q, float)
        z0 = self.kin.fk(q)[2, 3]
        g = []
        for i in range(len(JOINTS)):
            d = np.zeros(len(JOINTS))
            d[i] = 1e-4
            g.append((self.kin.fk(q + d)[2, 3] - z0) / 1e-4)
        return np.c_[g, np.asarray(s, float)]

    def predict(self, q, s):
        X, out = self.features(q, s), np.zeros(len(JOINTS))
        for j in range(len(JOINTS)):
            if self.n[j] >= self.MIN_N:
                w = np.linalg.solve(self.A[j] + 1e-4 * np.eye(2), self.y[j])
                out[j] = np.clip(X[j] @ w, -self.CLIP, self.CLIP)
        return out

    def learn(self, q, s, offset):
        X = self.features(q, s)
        for j in range(len(JOINTS)):
            self.A[j] = self.DECAY * self.A[j] + np.outer(X[j], X[j])
            self.y[j] = self.DECAY * self.y[j] + X[j] * offset[j]
            self.n[j] += 1

    def save(self):
        try:
            self.path.write_text(json.dumps({"A": self.A.tolist(), "y": self.y.tolist(), "n": self.n.tolist(),
                                             "note": "iro.motion.SagModel: learnt goal offsets (counts)"}))
        except OSError:
            pass


class Arm:
    def __init__(self, max_speed=None, dry_run=False):
        cfg = load_config()
        self.dev = cfg["devices"]["arm"]
        self.pick = cfg["pick"]
        self.lim = cfg["limits"]["servos"]
        self.signs = self.dev.get("joint_signs", {})
        self.offsets = {j: np.radians(v) for j, v in self.dev.get("joint_offsets_deg", {}).items()}
        self.max_speed = max_speed or self.pick["max_joint_speed"]
        self.dry_run = dry_run
        self.kin = Kinematics()
        self.sag = SagModel(self.kin)
        self.settle_log = []                            # (seconds, nudges, arrived) per settle
        self.cal = json.loads(CAL.read_text())
        wrapped = [j for j in JOINTS[:-1] if self.cal[j]["range_min"] <= 0 and self.cal[j]["range_max"] >= TICKS - 1]
        if wrapped:
            raise MotionError(f"calibration has no limits for {wrapped}: run `iro arm calibrate` first "
                              "(moving every joint end to end). Refusing to move.")
        self.bus = None

    # --- units ---------------------------------------------------------------
    def to_raw(self, name, rad):
        rad = rad - self.offsets.get(name, 0.0)
        return int(round(TICKS / 2 + self.signs.get(name, 1) * rad * TICKS / (2 * np.pi)))

    def to_rad(self, name, raw):
        return self.signs.get(name, 1) * (raw - TICKS / 2) * 2 * np.pi / TICKS + self.offsets.get(name, 0.0)

    # --- connection ----------------------------------------------------------
    def __enter__(self):
        self._cm = open_bus(self.dev["port"], self.dev["servo_ids"])
        self.bus = self._cm.__enter__()
        if not self.dry_run:
            # Goal = present position BEFORE torque on, so nothing jumps
            now = self.bus.sync_read("Present_Position", normalize=False, num_retry=3)
            self.bus.sync_write("Goal_Position", now, normalize=False, num_retry=3)
            # one motor at a time, retried: all six at once is a current surge, and a single garbled
            # reply from shoulder_lift once aborted `iro track` (21 Oct 21:24)
            for name in self.bus.motors:
                try:
                    self.bus.enable_torque(name, num_retry=3)
                except ConnectionError as e:
                    self._cm.__exit__(None, None, None)
                    raise MotionError(f"{name} does not answer when switched on ({e}). Check its cable and "
                                      "the 12 V supply, then: iro doctor") from e
                time.sleep(0.02)
        return self

    def limp(self):
        """Torque off: the arm can be moved by hand (it will drop under gravity)."""
        self.bus.disable_torque()

    def hold(self):
        """Torque back on, holding wherever the arm is now."""
        now = self.bus.sync_read("Present_Position", normalize=False)
        self.bus.sync_write("Goal_Position", now, normalize=False)
        self.bus.enable_torque()

    def __exit__(self, *exc):
        if not self.dry_run:
            self.sag.save()
        self._cm.__exit__(*exc)  # torque stays on: the arm holds its pose (and anything it carries)

    def joints(self) -> np.ndarray:
        raw = self.bus.sync_read("Present_Position", normalize=False)
        return np.array([self.to_rad(j, raw[j]) for j in JOINTS])

    def tip(self) -> np.ndarray:
        return self.kin.fk(self.joints())[:3, 3]

    # --- checks --------------------------------------------------------------
    def _check_target(self, q):
        for j, a, (lo, hi) in zip(JOINTS, q, self.kin.limits):
            raw = self.to_raw(j, a)
            c = self.cal[j]
            if not (lo - 1e-6 <= a <= hi + 1e-6) or not (c["range_min"] <= raw <= c["range_max"]):
                raise MotionError(f"{j} target {np.degrees(a):.0f} deg is outside its safe range")

    def _check_path(self, q0, q1, steps=20):
        # Never below the floor, but a move that starts lower (e.g. after a hand-guided
        # pose during teach) may still go up: it just must not go lower than its start.
        floor = min(self.pick["table_z"] + self.pick["clearance"], self.kin.fk(q0)[2, 3] - 0.002)
        for s in np.linspace(0, 1, steps):
            z = self.kin.fk(q0 + s * (q1 - q0))[2, 3]
            if z < floor:
                raise MotionError(f"path would take the gripper to z={z * 100:.1f} cm, below the safe floor "
                                  f"{floor * 100:.1f} cm (table_z + clearance in config/robot.yaml)")

    def _check_servos(self):
        """Stop on a servo error or overheating, but only if a second read agrees: a garbled
        packet once reported 66 C on a servo that was really at 38 C (20 Oct)."""
        def bad():
            st = self.bus.sync_read("Status", normalize=False)
            temp = self.bus.sync_read("Present_Temperature", normalize=False)
            volts = self.bus.sync_read("Present_Voltage", normalize=False)
            out = {}
            low = min(volts.values()) / 10
            if low < self.lim.get("volt_fail", 10.0):        # 22 Oct: supply sagged 11.4 -> 7.2 -> 5.8 V
                out["power"] = (f"servo supply at {low:.1f} V (needs ~12 V): charge/replace the arm's battery or "
                                "power supply, then `iro doctor`")
            for j in st:
                if st[j]:
                    out[j] = f"{j} servo reports {', '.join(decode_status(st[j]))}"
                elif temp[j] >= self.lim["temp_fail_c"]:
                    out[j] = f"{j} at {temp[j]} C: too hot to keep moving"
            return out
        first = bad()
        if first:
            time.sleep(0.1)
            again = bad()
            for j, msg in first.items():
                if j in again:
                    raise MotionError(again[j])

    # --- moves ---------------------------------------------------------------
    def move_joints(self, q_target, duration=None, floor_check=True, settle=True):
        q_target = np.asarray(q_target, float)
        q0 = self.joints()
        self._check_target(q_target)
        if floor_check:
            self._check_path(q0, q_target)
        span = float(np.max(np.abs(q_target - q0)))
        duration = max(duration or 0.0, span / self.max_speed, 0.3)
        if self.dry_run:
            print(f"  [dry run] move {np.round(np.degrees(q0), 0)} -> {np.round(np.degrees(q_target), 0)} deg in {duration:.1f}s")
            return
        # aim past the target by the learnt sag, faded in along the move (no jump at the end)
        s_dir = self._approach(q0, q_target)
        pred = self.sag.predict(q_target, s_dir) if settle else np.zeros(len(JOINTS))
        rate, t0 = 50.0, time.time()
        while True:
            s = min((time.time() - t0) / duration, 1.0)
            k = 10 * s**3 - 15 * s**4 + 6 * s**5          # minimum-jerk profile
            q = q0 + k * (q_target - q0)
            self.bus.sync_write("Goal_Position", self._goal(q, k * pred), normalize=False)
            if s >= 1.0:
                break
            time.sleep(1 / rate)
        if settle:                                     # off = keep moving (passing points)
            self._settle(q_target, self._goal(q_target, pred), s_dir)
        self._check_servos()

    @staticmethod
    def _approach(q0, q1):
        """Which way each joint moves (+1/-1), 0 when it hardly moves (< 0.5 deg)."""
        d = np.asarray(q1, float) - np.asarray(q0, float)
        return np.where(np.abs(d) < np.radians(0.5), 0.0, np.sign(d))

    def _goal(self, q, extra=None):
        """Raw goal positions for joint angles q, plus `extra` counts per joint (sag aim)."""
        extra = np.zeros(len(JOINTS)) if extra is None else extra
        return {j: int(round(self.to_raw(j, a) + e)) for j, a, e in zip(JOINTS, q, extra)}

    def _settle(self, q_target, goal=None, s_dir=None, tol=3, max_comp=90, timeout=1.2):
        """Make the joints really arrive (soft servos stop 30-40 counts short, 20 Oct). Watches the
        positions every 10 ms (a read of all six takes 1 ms) instead of fixed 0.25 s waits: done the
        moment every joint is within `tol` counts; a joint that has STOPPED short (no change for 40 ms)
        gets its goal nudged by part of the error. What finally worked is learnt (SagModel)."""
        want = {j: self.to_raw(j, a) for j, a in zip(JOINTS, q_target)}
        goal = dict(goal or want)
        t0, prev, still, nudges = time.time(), None, 0, 0
        arrived = False
        while time.time() - t0 < timeout:
            time.sleep(0.01)
            pos = self.bus.sync_read("Present_Position", normalize=False)
            err = {j: want[j] - pos[j] for j in want}
            if max(abs(e) for e in err.values()) <= tol:
                arrived = True
                break
            moving = prev is None or max(abs(pos[j] - prev[j]) for j in want) > 1
            prev = pos
            still = 0 if moving else still + 1
            if still >= 4:
                for j, e in err.items():
                    goal[j] = int(np.clip(goal[j] + 0.6 * e, want[j] - max_comp, want[j] + max_comp))
                self.bus.sync_write("Goal_Position", goal, normalize=False)
                still, nudges = 0, nudges + 1
        self.settle_log.append((time.time() - t0, nudges, arrived))
        if arrived and s_dir is not None:
            self.sag.learn(q_target, s_dir, np.array([goal[j] - want[j] for j in JOINTS], float))
        return arrived

    def level_roll(self):
        """Wrist roll with the jaws level: raw 2048 = the calibration middle pose."""
        return self.to_rad("wrist_roll", TICKS // 2)

    # Every joint finishes its move in the same direction: the servos stop slightly differently
    # depending on which side they come from (up to ~19 px in the wrist camera, measured 20 Oct).
    APPROACH = np.radians([3, -3, -3, -3, 0])

    def move_joints_repeatable(self, q, duration=None):
        """Go to q always via the same nearby pre-pose, so the pose repeats to the pixel."""
        q = np.asarray(q, float)
        pre = np.clip(q + self.APPROACH, self.kin.limits[:, 0], self.kin.limits[:, 1])
        self.move_joints(pre, duration)
        self.move_joints(q, 0.5)

    def move_tip(self, xyz, duration=None, tilt=0.0, roll=None, settle=True):
        """Gripper tip to xyz (metres, base_link), pointing straight down. The wrist roll is
        held at `roll` (default: jaws level), so the gripper and camera never twist sideways."""
        roll = self.level_roll() if roll is None else roll
        start = self.joints()
        start[4] = roll
        q, err = self.kin.ik_down(xyz, start, tilt=tilt)
        if err > 0.003:  # stuck from the current pose: try a few typical poses as starting points
            for seed in SEEDS:
                seed = np.array(seed)
                seed[4] = roll
                q2, err2 = self.kin.ik_down(xyz, seed, tilt=tilt)
                if err2 < err:
                    q, err = q2, err2
        if err > 0.005:
            raise MotionError(f"cannot reach {np.round(xyz, 3)} pointing down (closest {err * 1000:.0f} mm away)")
        self.move_joints(q, duration, settle=settle)
        return q

    def move_line(self, xyz, speed=0.05, tilt=0.0, roll=None, watch=None, settle=True, step=0.005):
        """Tip in a STRAIGHT line to xyz, pointing down, as ONE smooth movement (no stops on the way;
        minimum-jerk, so it starts and lands softly). `speed` = top tip speed (m/s). IK is solved every
        `step` metres before moving (~10 ms each), then streamed at 50 Hz. While moving,
        watch(progress 0-1, q_commanded, q_measured) is called each tick (a position read takes 1 ms);
        returning True stops the arm where it is. Returns True when it arrived, False when stopped."""
        roll = self.level_roll() if roll is None else roll
        q0 = self.joints()
        p0, p1 = self.kin.fk(q0)[:3, 3], np.asarray(xyz, float)
        n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / step)))
        qs, seed = [q0], q0.copy()
        seed[4] = roll
        for i in range(1, n + 1):
            p = p0 + (p1 - p0) * i / n
            q, err = self.kin.ik_down(p, seed, tilt=tilt)
            if err > 0.003:
                raise MotionError(f"straight line: cannot reach {np.round(p, 3)} pointing down "
                                  f"({err * 1000:.0f} mm off)")
            self._check_target(q)
            qs.append(q)
            seed = q
        qs = np.array(qs)
        for a, b in zip(qs[:-1], qs[1:]):
            self._check_path(a, b, steps=3)
        span = float(np.max(np.abs(np.diff(qs, axis=0)).sum(0)))      # each joint's total travel
        dist = float(np.linalg.norm(p1 - p0))
        duration = max(1.875 * dist / speed, 1.875 * span / self.max_speed, 0.3)   # 1.875: min-jerk peak
        if self.dry_run:
            print(f"  [dry run] line {np.round(p0, 3)} -> {np.round(p1, 3)} in {duration:.1f}s ({n} IK points)")
            return True
        q_end = qs[-1]
        s_dir = self._approach(q0, q_end)
        pred = self.sag.predict(q_end, s_dir) if settle else np.zeros(len(JOINTS))
        t0 = time.time()
        while True:
            s = min((time.time() - t0) / duration, 1.0)
            k = 10 * s**3 - 15 * s**4 + 6 * s**5
            u = k * n
            i = min(int(u), n - 1)
            q = qs[i] + (u - i) * (qs[i + 1] - qs[i])
            self.bus.sync_write("Goal_Position", self._goal(q, k * pred), normalize=False)
            if watch is not None and watch(k, q, self.joints()):
                now = self.bus.sync_read("Present_Position", normalize=False)
                self.bus.sync_write("Goal_Position", {j: now[j] for j in JOINTS}, normalize=False)
                return False
            if s >= 1.0:
                break
            time.sleep(0.02)
        if settle:
            self._settle(q_end, self._goal(q_end, pred), s_dir)
        self._check_servos()
        return True

    def gripper(self, open: bool, wait=0.6):
        c = self.cal["gripper"]
        pct = self.pick["gripper_open_pct"] if open else self.pick["gripper_closed_pct"]
        raw = int(c["range_min"] + pct / 100 * (c["range_max"] - c["range_min"]))
        if self.dry_run:
            print(f"  [dry run] gripper {'open' if open else 'close'} ({pct}%)")
            return
        self.bus.write("Goal_Position", "gripper", raw, normalize=False)
        time.sleep(wait)

    def gripper_pct(self) -> float:
        c = self.cal["gripper"]
        raw = self.bus.read("Present_Position", "gripper", normalize=False)
        return 100 * (raw - c["range_min"]) / (c["range_max"] - c["range_min"])
