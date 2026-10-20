"""SO-101 forward and inverse kinematics, read straight from the URDF that RViz uses.

    kin = Kinematics()
    T = kin.fk(q)                        # 4x4 pose of the gripper tip in base_link
    q = kin.ik_down(target_xyz, q_seed)  # joints that put the tip at xyz, pointing straight down

Joint angles q are radians in the order of JOINTS (the gripper joint is not part
of the chain). Zero = LeRobot's calibration middle pose (the "new_calib" URDF).
"""

import xml.etree.ElementTree as ET

import numpy as np

from iro import ROOT

URDF = ROOT / "ros2_ws" / "src" / "iro_bringup" / "urdf" / "so101" / "so101_new_calib_camera.urdf"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TIP = "gripper_frame_link"     # between the jaw tips, z axis pointing out of the gripper
CAMERA = "wrist_camera_link"


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, a):
    x, y, z = axis
    c, s, t = np.cos(a), np.sin(a), 1 - np.cos(a)
    return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                     [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                     [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])


def _tf(R=np.eye(3), p=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


class Kinematics:
    def __init__(self, urdf=URDF):
        root = ET.parse(urdf).getroot()
        self.joints = {}
        for j in root.findall("joint"):
            o, a, lim = j.find("origin"), j.find("axis"), j.find("limit")
            self.joints[j.find("child").get("link")] = {
                "name": j.get("name"), "type": j.get("type"), "parent": j.find("parent").get("link"),
                "origin": _tf(_rpy(*map(float, o.get("rpy", "0 0 0").split())), list(map(float, o.get("xyz", "0 0 0").split()))),
                "axis": np.array(list(map(float, a.get("xyz").split()))) if a is not None else None,
                "limit": (float(lim.get("lower")), float(lim.get("upper"))) if lim is not None else None,
            }
        self.limits = np.array([self._by_name(n)["limit"] for n in JOINTS])
        self._chains = {}

    def _by_name(self, name):
        return next(j for j in self.joints.values() if j["name"] == name)

    def _chain(self, link):
        """Joints from base_link to `link`, as (origin, axis, index into q or None); cached."""
        if link not in self._chains:
            chain, cur = [], link
            while cur in self.joints:
                j = self.joints[cur]
                idx = JOINTS.index(j["name"]) if j["name"] in JOINTS else None
                chain.append((j["origin"], j["axis"] if j["type"] == "revolute" else None, idx))
                cur = j["parent"]
            self._chains[link] = chain[::-1]
        return self._chains[link]

    def fk(self, q, link=TIP) -> np.ndarray:
        """4x4 transform of `link` in base_link for joint angles q (len 5)."""
        T = np.eye(4)
        for origin, axis, idx in self._chain(link):
            T = T @ origin
            if axis is not None and idx is not None:
                R = _axis_angle(axis, q[idx])
                T[:3, :3] = T[:3, :3] @ R
        return T

    def ik_down(self, xyz, q_seed, tilt=0.0, iters=200, tol=1e-3):
        """Joints putting the tip at xyz with the gripper pointing straight down
        (tilt: radians away from vertical; positive leans toward the base, negative leans
        outward, which the SO-101 wrist needs above about 8 cm). Returns (q, error_m).
        Damped least squares on position + the tip's z axis; keeps joint limits.
        wrist_roll is NOT solved: it stays at q_seed's value. Roll does not change where the
        tip is or where it points, so a solver left free to turn it twists the gripper sideways."""
        q = np.clip(np.array(q_seed, float), self.limits[:, 0], self.limits[:, 1])
        active = [0, 1, 2, 3]   # pan, lift, elbow, wrist_flex
        target = np.asarray(xyz, float)
        down = np.array([0.0, 0.0, -1.0])
        if tilt:
            r = np.hypot(*target[:2]) or 1.0
            toward = -np.array([target[0] / r, target[1] / r, 0.0])
            down = np.cos(tilt) * down + np.sin(tilt) * toward

        def residual(q):
            T = self.fk(q)
            return np.concatenate([target - T[:3, 3], 0.1 * (down - T[:3, 2])])  # 0.1: 1 rad ~ 10 cm

        for _ in range(iters):
            e = residual(q)
            if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) < 0.01 * 0.1:
                break
            J = np.zeros((6, len(active)))
            for k, i in enumerate(active):
                dq = np.zeros(5)
                dq[i] = 1e-5
                J[:, k] = (residual(q) - residual(q + dq)) / 1e-5
            lam = 0.01
            step = np.zeros(5)
            step[active] = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            q = np.clip(q + np.clip(step, -0.2, 0.2), self.limits[:, 0], self.limits[:, 1])
        return q, float(np.linalg.norm(target - self.fk(q)[:3, 3]))
