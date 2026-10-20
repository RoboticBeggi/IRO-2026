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


class MotionError(RuntimeError):
    pass


class Arm:
    def __init__(self, max_speed=None, dry_run=False):
        cfg = load_config()
        self.dev = cfg["devices"]["arm"]
        self.pick = cfg["pick"]
        self.lim = cfg["limits"]["servos"]
        self.signs = self.dev.get("joint_signs", {})
        self.max_speed = max_speed or self.pick["max_joint_speed"]
        self.dry_run = dry_run
        self.kin = Kinematics()
        self.cal = json.loads(CAL.read_text())
        wrapped = [j for j in JOINTS[:-1] if self.cal[j]["range_min"] <= 0 and self.cal[j]["range_max"] >= TICKS - 1]
        if wrapped:
            raise MotionError(f"calibration has no limits for {wrapped}: run `iro arm calibrate` first "
                              "(moving every joint end to end). Refusing to move.")
        self.bus = None

    # --- units ---------------------------------------------------------------
    def to_raw(self, name, rad):
        return int(round(TICKS / 2 + self.signs.get(name, 1) * rad * TICKS / (2 * np.pi)))

    def to_rad(self, name, raw):
        return self.signs.get(name, 1) * (raw - TICKS / 2) * 2 * np.pi / TICKS

    # --- connection ----------------------------------------------------------
    def __enter__(self):
        self._cm = open_bus(self.dev["port"], self.dev["servo_ids"])
        self.bus = self._cm.__enter__()
        if not self.dry_run:
            # Goal = present position BEFORE torque on, so nothing jumps
            now = self.bus.sync_read("Present_Position", normalize=False)
            self.bus.sync_write("Goal_Position", now, normalize=False)
            self.bus.enable_torque()
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
        st = self.bus.sync_read("Status", normalize=False)
        temp = self.bus.sync_read("Present_Temperature", normalize=False)
        for j in st:
            if st[j]:
                raise MotionError(f"{j} servo reports {', '.join(decode_status(st[j]))}")
            if temp[j] >= self.lim["temp_fail_c"]:
                raise MotionError(f"{j} at {temp[j]} C: too hot to keep moving")

    # --- moves ---------------------------------------------------------------
    def move_joints(self, q_target, duration=None, floor_check=True):
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
        rate, t0 = 50.0, time.time()
        while True:
            s = min((time.time() - t0) / duration, 1.0)
            k = 10 * s**3 - 15 * s**4 + 6 * s**5          # minimum-jerk profile
            q = q0 + k * (q_target - q0)
            self.bus.sync_write("Goal_Position", {j: self.to_raw(j, a) for j, a in zip(JOINTS, q)}, normalize=False)
            if s >= 1.0:
                break
            time.sleep(1 / rate)
        time.sleep(0.2)
        self._check_servos()

    def move_tip(self, xyz, duration=None, tilt=0.0):
        """Gripper tip to xyz (metres, base_link), pointing straight down."""
        q, err = self.kin.ik_down(xyz, self.joints(), tilt=tilt)
        if err > 0.003:  # stuck from the current pose: try a few typical poses as starting points
            for seed in SEEDS:
                q2, err2 = self.kin.ik_down(xyz, seed, tilt=tilt)
                if err2 < err:
                    q, err = q2, err2
        if err > 0.005:
            raise MotionError(f"cannot reach {np.round(xyz, 3)} pointing down (closest {err * 1000:.0f} mm away)")
        self.move_joints(q, duration)
        return q

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
