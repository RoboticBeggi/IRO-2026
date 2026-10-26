"""Visitor detection: the lidar finds and follows people, the mic array says who is speaking.

    iro visitors                  watch live (radar picture on the live page, events in the terminal)
    iro visitors arm              pose the limp arm by hand while it shows how much of the lidar's
                                  view is blocked; Enter saves the pose that blocks least ("lidar-safe")
    iro visitors learn            learn the empty room (everyone out of the way for 10 s)
    iro visitors mic-cal          align the mic's angles with the lidar's (talk while walking around)
    iro visitors zone add NAME [--nogo]   teach a floor zone by walking its edge (iro/surveil.py)
    iro visitors --from RUN       replay a recording (no hardware)

How it works (each step is a standard method; see docs in the code):
  1. background subtraction: every one of the ~3240 lidar beams has a learned "empty room" range;
     points clearly CLOSER than that are new things. Beams the robot itself blocks are masked.
  2. segmentation: neighbouring new points form blobs; blobs of leg / legs / torso size (5-100 cm)
     are person candidates; blobs closer than 45 cm to each other merge (two legs = one person)
  3. tracking: a constant-velocity Kalman filter per person, gated nearest-neighbour association,
     confirmed after 3 sightings (no ghosts from one bad scan), dropped after 1.5 s unseen
  4. proxemics (E. T. Hall): personal space < 1.2 m, social 1.2-3.6 m. Someone walking in = "approaching";
     staying within 2.5 m and slowing down = "engaged" (ARC greets them)
  5. speaker: the mic array's processed voice direction (only while someone speaks) is matched to the
     tracked person in that direction (+/- 25 deg) after `mic-cal` aligned the two sensors
Everything is recorded (radar video, tracks, events, raw scans for --from).
"""

import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from iro import ROOT

CFG = ROOT / "config" / "visitors.yaml"
BG = ROOT / "config" / "visitors_room.npz"
SELF_R = 0.60            # m: anything this close in the empty-room scan is the robot itself
MAX_R = 6.0              # m: ignore farther (the lidar sees 30 m, the exhibition is smaller)
PERSONAL, SOCIAL = 1.2, 3.6
ENGAGE_R = 2.5


def cfg():
    if CFG.exists():
        return yaml.safe_load(CFG.read_text()) or {}
    return {}


def save_cfg(**kw):
    c = cfg()
    c.update(kw)
    CFG.write_text(yaml.safe_dump(c, sort_keys=False))


# ------------------------------------------------------------------ the lidar (ROS 2 rplidar_ros)

class Lidar:
    """Starts Slamtec's driver (DenseBoost mode, 10 Hz, ~3240 beams) and keeps the newest scan."""

    def __init__(self, port="/dev/rplidar"):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import LaserScan
        self.proc = subprocess.Popen(
            ["bash", "-c", f"source /opt/ros/humble/setup.bash && exec ros2 launch rplidar_ros "
                           f"rplidar_s2_launch.py serial_port:={port}"],
            stdout=open("/tmp/iro-lidar.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
        if not rclpy.ok():
            rclpy.init()
        self.rclpy = rclpy
        self.node = Node("iro_visitors")
        self.scan, self.n, self.lock = None, 0, threading.Lock()
        self.node.create_subscription(LaserScan, "/scan", self._cb, qos_profile_sensor_data)
        self.run = True
        self.th = threading.Thread(target=self._spin, daemon=True)
        self.th.start()
        t0 = time.time()
        while self.scan is None:
            if time.time() - t0 > 20:
                self.close()
                raise RuntimeError("no lidar scans in 20 s: see /tmp/iro-lidar.log (iro doctor devices)")
            time.sleep(0.1)

    def _cb(self, m):
        r = np.array(m.ranges, np.float32)
        a = m.angle_min + np.arange(len(r), dtype=np.float32) * m.angle_increment
        with self.lock:
            self.scan = (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, a, r)
            self.n += 1

    def _spin(self):
        while self.run:
            self.rclpy.spin_once(self.node, timeout_sec=0.2)

    def read(self, last_n=None):
        """(n, (stamp, angles, ranges)), waiting for a scan newer than last_n."""
        while True:
            with self.lock:
                if self.n != last_n:
                    return self.n, self.scan
            time.sleep(0.01)

    def close(self):
        self.run = False
        try:
            self.node.destroy_node()
        except Exception:                          # noqa: BLE001
            pass
        import os
        import signal
        try:
            os.killpg(self.proc.pid, signal.SIGINT)
            self.proc.wait(timeout=5)
        except Exception:                          # noqa: BLE001
            os.killpg(self.proc.pid, signal.SIGKILL)


class Replay:
    """Scans (and mic samples) from a recording: same interface as Lidar + MicThread."""

    def __init__(self, run_dir):
        d = np.load(Path(run_dir) / "scans.npz")
        self.t, self.a, self.R = d["t"], d["a"], d["r"].astype(np.float32)
        self.mic = json.loads((Path(run_dir) / "mic.json").read_text()) if (Path(run_dir) / "mic.json").exists() else []
        self.i = 0

    def read(self, last_n=None):
        if self.i >= len(self.t):
            raise EOFError
        self.i += 1
        time.sleep(0.02)
        return self.i, (float(self.t[self.i - 1]), self.a, self.R[self.i - 1])

    def mic_at(self, t):
        best = [m for m in self.mic if abs(m["t"] - t) < 0.3]
        return best[-1] if best else None

    def close(self):
        pass


class MicThread:
    """Polls the mic array ~10x a second in the background; newest sample in .last."""

    def __init__(self):
        from iro.mic import MicArray
        self.m = MicArray()
        self.m.led_doa()
        self.last, self.run = None, True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.run:
            try:
                s = self.m.sample()
                if s:
                    self.last = s
            except Exception:                      # noqa: BLE001 - a USB hiccup: try again
                time.sleep(0.2)
            time.sleep(0.08)

    def mic_at(self, t):
        s = self.last
        return s if s and abs(s["t"] - t) < 0.5 else None

    def close(self):
        self.run = False


# ------------------------------------------------------------------ the room (background) and the robot itself

def learn_room(lidar, seconds=10):
    """Per-beam empty-room range (median) and spread (MAD) over `seconds`; beams closer than SELF_R
    are the robot itself. Saved to config/visitors_room.npz."""
    rs, n = [], None
    t0 = time.time()
    while time.time() - t0 < seconds:
        n, (_, a, r) = lidar.read(n)
        rs.append(r)
    R = np.array(rs)
    R[~np.isfinite(R) | (R <= 0)] = np.nan
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(R, 0)
            mad = np.nanmedian(np.abs(R - med), 0)
            seen = np.mean(np.isfinite(R), 0)
    med[seen < 0.3] = np.inf                       # mostly no echo: open space
    mad = np.nan_to_num(mad, nan=0.0)
    self_mask = np.isfinite(med) & (med < SELF_R)
    np.savez(BG, a=a, med=med, mad=mad, self_mask=self_mask, date=time.strftime("%Y-%m-%d %H:%M"))
    return a, med, mad, self_mask


def blocked_deg(a, r, self_r=SELF_R):
    """How many degrees of the lidar's view something closer than self_r blocks (the arm, the body)."""
    near = np.isfinite(r) & (r > 0) & (r < self_r)
    return float(near.sum() * abs(float(a[1] - a[0])) * 180 / math.pi)


# ------------------------------------------------------------------ detection and tracking

def shape(px, py):
    """Blob shape features (as in leg detectors, e.g. leg_tracker): width, straightness (0 = a
    straight line, larger = curved), and the radius of the best-fitting circle (a leg at knee
    height is a ~5-8 cm arc facing the lidar)."""
    w = float(np.hypot(px[-1] - px[0], py[-1] - py[0]))
    pts = np.stack([px - px.mean(), py - py.mean()], 1)
    ev = np.linalg.eigvalsh(pts.T @ pts / max(len(pts), 1))
    straight = float(np.sqrt(max(ev[0], 0)) / max(np.sqrt(max(ev[1], 1e-9)), 1e-6))
    # algebraic circle fit (Kasa): x^2 + y^2 + D x + E y + F = 0
    A = np.stack([px, py, np.ones_like(px)], 1)
    try:
        D, E, F = np.linalg.lstsq(A, -(px ** 2 + py ** 2), rcond=None)[0]
        rad = float(np.sqrt(max(D * D / 4 + E * E / 4 - F, 0)))
    except np.linalg.LinAlgError:
        rad = float("inf")
    return w, straight, rad


def leg_score(w, straight, rad):
    """0..1: how much a blob looks like a leg (knee height: 6-20 cm wide, curved, radius 3-15 cm)."""
    s = 1.0
    s *= 1.0 if 0.05 <= w <= 0.22 else 0.4 if w <= 0.30 else 0.1
    s *= 1.0 if straight > 0.08 else 0.5         # a flat piece of wall or box side is straight
    s *= 1.0 if 0.025 <= rad <= 0.18 else 0.5
    return s


def detect(a, r, med, mad, self_mask, max_r=MAX_R, others=None):
    """Person candidates [(x, y, width_m, n_points, person_score)] in the lidar frame (x forward,
    y left). Legs are paired into people (two legs 10-60 cm apart); a single leg (the other one
    hidden) or a body-wide blob still counts, with a lower score. Foreground blobs that do not look
    like a person go to `others` (a list) for change detection."""
    ok = np.isfinite(r) & (r > 0.15) & (r < max_r) & ~self_mask
    with np.errstate(invalid="ignore"):            # inf - inf where both are "no echo": not foreground
        margin = np.maximum(0.12, np.maximum(3 * mad, 0.03 * r))
        fg = ok & (~np.isfinite(med) | (r < med - margin))
    idx = np.flatnonzero(fg)
    if len(idx) < 3:
        return []
    x, y = r[idx] * np.cos(a[idx]), r[idx] * np.sin(a[idx])
    # break into blobs where neighbouring points jump apart (or a gap of beams)
    jump = np.hypot(np.diff(x), np.diff(y)) > 0.12 + 0.01 * r[idx][1:]
    gap = np.diff(idx) > 8
    cuts = np.flatnonzero(jump | gap) + 1
    blobs = []
    for s in np.split(np.arange(len(idx)), cuts):
        if len(s) < 4:
            continue
        px, py = x[s], y[s]
        w, straight, rad = shape(px, py)
        if 0.04 <= w <= 1.0:
            blobs.append([float(px.mean()), float(py.mean()), w, len(s), leg_score(w, straight, rad), straight])
    # pair legs: the two best-matching leg-like blobs 10-60 cm apart are one person
    legs = sorted((b for b in blobs if b[4] >= 0.5 and b[2] <= 0.30), key=lambda b: -b[4])
    people, used = [], set()
    for i, b in enumerate(legs):
        if id(b) in used:
            continue
        mate = min((c for c in legs[i + 1:] if id(c) not in used and 0.10 <= math.hypot(b[0] - c[0], b[1] - c[1]) <= 0.60),
                   key=lambda c: math.hypot(b[0] - c[0], b[1] - c[1]), default=None)
        used.add(id(b))
        if mate is not None:
            used.add(id(mate))
            n = b[3] + mate[3]
            people.append(((b[0] * b[3] + mate[0] * mate[3]) / n, (b[1] * b[3] + mate[1] * mate[3]) / n,
                           math.hypot(b[0] - mate[0], b[1] - mate[1]) + (b[2] + mate[2]) / 2, n,
                           min(1.0, (b[4] + mate[4]) / 2 + 0.2)))
        else:
            people.append((b[0], b[1], b[2], b[3], 0.5 * b[4]))   # one leg: the other may be hidden
    for b in blobs:
        if id(b) in used or b in legs:
            continue
        if 0.25 < b[2] <= 0.7 and b[5] > 0.08:     # legs together / long coat: one curved body-wide blob
            people.append((b[0], b[1], b[2], b[3], 0.4))
        elif others is not None:
            others.append((b[0], b[1], b[2], b[3]))
    return people


class Track:
    """Constant-velocity Kalman filter: state (x, y, vx, vy)."""
    _next = 1

    def __init__(self, x, y, t):
        self.id = None
        self.s = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([0.05, 0.05, 1.0, 1.0])
        self.t = self.t0 = self.seen = t
        self.hits, self.misses = 1, 0
        self.score, self.start, self.moved = 0.5, (x, y), 0.0   # person-likeness, where first seen
        self.recent = []                            # last ~3 s of measured positions (sway)
        self.state, self.speaking_t, self.engaged_t = "new", 0.0, None

    def predict(self, t):
        dt = max(1e-3, t - self.t)
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = 0.5                                     # m/s^2: people change speed gently
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        self.s = F @ self.s
        self.P = F @ self.P @ F.T + G @ G.T * q ** 2
        self.t = t

    def update(self, z, t):
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1
        Rm = np.eye(2) * 0.06 ** 2                  # a blob centre is good to ~6 cm
        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)
        self.s = self.s + K @ (np.asarray(z) - H @ self.s)
        self.P = (np.eye(4) - K @ H) @ self.P
        self.seen, self.hits, self.misses = t, self.hits + 1, 0
        self.moved = max(self.moved, math.hypot(self.s[0] - self.start[0], self.s[1] - self.start[1]))
        self.recent = (self.recent + [tuple(z)])[-30:]
        # A visitor must have WALKED (people come in walking) or, standing, SWAY like a person (a few cm;
        # chair and table legs, which look exactly like human legs at knee height, do not move at all:
        # 26 Oct, a stale room turned furniture into 12 "visitors" in 12 s)
        sway = float(np.std(np.array(self.recent), 0).max()) if len(self.recent) >= 20 else 0.0
        if self.id is None and self.hits >= 3 and (self.moved > 0.3 or (self.score >= 0.6 and sway > 0.025)):
            self.id, Track._next = Track._next, Track._next + 1

    @property
    def dist(self):
        return float(math.hypot(self.s[0], self.s[1]))

    @property
    def bearing(self):
        return float(math.degrees(math.atan2(self.s[1], self.s[0])))

    @property
    def speed(self):
        return float(math.hypot(self.s[2], self.s[3]))

    @property
    def radial(self):
        """m/s towards (-) or away from (+) the robot."""
        return float((self.s[0] * self.s[2] + self.s[1] * self.s[3]) / max(self.dist, 1e-3))

    def zone(self):
        d = self.dist
        return "personal" if d < PERSONAL else "social" if d < SOCIAL else "public"


class Visitors:
    """Lidar + mic -> confirmed visitors with position, speed, zone, approaching/engaged, speaking."""

    def __init__(self):
        if not BG.exists():
            raise SystemExit("The room is not learned yet: iro visitors learn")
        d = np.load(BG)
        self.a, self.med, self.mad, self.self_mask = d["a"], d["med"], d["mad"], d["self_mask"]
        c = cfg()
        self.mic_sign = c.get("mic_sign", 1)
        self.mic_offset = c.get("mic_offset_deg", 0.0)
        self.mic_source = c.get("mic_source", "speaker")
        self.tracks = []
        self.events = []
        self.voice = None                          # last speech bearing in the lidar frame

    def mic_to_lidar(self, deg):
        return (self.mic_sign * deg + self.mic_offset + 180) % 360 - 180

    def step(self, t, a, r, mic=None):
        if len(r) != len(self.med):                # a different scan mode: resample the room
            self.med = np.interp(a, self.a, self.med)
            self.mad = np.interp(a, self.a, self.mad)
            self.self_mask = np.interp(a, self.a, self.self_mask.astype(float)) > 0.5
            self.a = a
        self.others = []
        dets = detect(a, r, self.med, self.mad, self.self_mask, others=self.others)
        for tr in self.tracks:
            tr.predict(t)
        # gated greedy nearest-neighbour association (few people: as good as the Hungarian method)
        pairs = sorted(((math.hypot(tr.s[0] - d[0], tr.s[1] - d[1]), i, j)
                        for i, tr in enumerate(self.tracks) for j, d in enumerate(dets)))
        used_t, used_d = set(), set()
        for dist, i, j in pairs:
            gate = 0.5 + 0.5 * self.tracks[i].speed * max(0.1, t - self.tracks[i].seen)
            if dist > gate or i in used_t or j in used_d:
                continue
            self.tracks[i].score = 0.8 * self.tracks[i].score + 0.2 * dets[j][4]
            self.tracks[i].update(dets[j][:2], t)
            used_t.add(i)
            used_d.add(j)
        for j, d in enumerate(dets):
            if j not in used_d:
                self.tracks.append(Track(d[0], d[1], t))
                self.tracks[-1].score = d[4]
        keep = []
        for tr in self.tracks:
            if t - tr.seen > (1.5 if tr.id else 0.4):
                if tr.id:
                    self.event(t, f"visitor {tr.id} left")
                continue
            keep.append(tr)
        self.tracks = keep
        # voice -> person
        self.voice = None
        deg = (mic.get("dirs") or {}).get(self.mic_source) if mic else None
        if deg is not None:
            self.voice = self.mic_to_lidar(deg)
            best = min((tr for tr in self.tracks if tr.id),
                       key=lambda tr: abs((tr.bearing - self.voice + 180) % 360 - 180), default=None)
            if best and abs((best.bearing - self.voice + 180) % 360 - 180) < 25:
                if t - best.speaking_t > 3:
                    self.event(t, f"visitor {best.id} is speaking ({best.dist:.1f} m, {best.bearing:+.0f} deg)")
                best.speaking_t = t
        # approaching / engaged
        for tr in self.tracks:
            if not tr.id:
                continue
            if tr.state == "new":
                tr.state = "seen"
                self.event(t, f"visitor {tr.id} appeared {tr.dist:.1f} m away at {tr.bearing:+.0f} deg")
            if tr.state == "seen" and tr.radial < -0.2 and tr.dist < SOCIAL:
                tr.state = "approaching"
                self.event(t, f"visitor {tr.id} is approaching ({tr.dist:.1f} m)")
            if tr.dist < ENGAGE_R and tr.speed < 0.3:
                tr.engaged_t = tr.engaged_t or t
                if tr.state != "engaged" and t - tr.engaged_t > 1.0:
                    tr.state = "engaged"
                    self.event(t, f"visitor {tr.id} stopped to engage ({tr.dist:.1f} m, {tr.zone()} zone)")
            else:
                tr.engaged_t = None
                if tr.state == "engaged" and tr.dist > ENGAGE_R + 0.5:
                    tr.state = "seen"
        return dets

    def event(self, t, text):
        self.events.append((t, text))

    def visitors(self):
        return [tr for tr in self.tracks if tr.id]

    def summary(self):
        """One line for ARC's status: who is there, how far, who speaks."""
        v = self.visitors()
        if not v:
            return "no visitor in front of you"
        parts = []
        for tr in sorted(v, key=lambda tr: tr.dist):
            side = "ahead" if abs(tr.bearing) < 20 else ("to your left" if tr.bearing > 0 else "to your right")
            spk = ", speaking" if tr.t - tr.speaking_t < 2 else ""
            parts.append(f"visitor {tr.id} {tr.dist:.1f} m {side}, {tr.state}{spk}")
        return "; ".join(parts)


# ------------------------------------------------------------------ the radar picture

def radar(a, r, dets, vis, voice=None, size=640, span=4.0, title=""):
    img = np.full((size, size, 3), 250, np.uint8)
    c, sc = size // 2, size / 2 / span
    def px(x, y):                                  # x forward = up, y left = left
        return int(c - y * sc), int(c - x * sc)
    for m, col in ((PERSONAL, (200, 220, 255)), (ENGAGE_R, (210, 240, 210)), (SOCIAL, (230, 230, 230))):
        cv2.circle(img, (c, c), int(m * sc), col, 2)
    for m in range(1, int(span) + 1):
        cv2.putText(img, f"{m}m", (c + 3, c - int(m * sc) + 12), 0, 0.35, (150, 150, 150), 1)
    ok = np.isfinite(vis.med) & np.isfinite(r) & (r > 0) & (r < span * 1.5)
    for x, y in zip(r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])):
        img[min(size - 1, max(0, px(x, y)[1])), min(size - 1, max(0, px(x, y)[0]))] = (90, 90, 90)
    sm = vis.self_mask
    for x, y in zip(np.cos(a[sm]) * 0.25, np.sin(a[sm]) * 0.25):
        cv2.circle(img, px(x, y), 1, (0, 140, 255), -1)          # orange: blocked by the robot itself
    for d in dets:
        cv2.circle(img, px(d[0], d[1]), max(3, int(d[2] / 2 * sc)), (180, 180, 180), 1)
    if voice is not None:
        t = math.radians(voice)
        cv2.line(img, (c, c), px(math.cos(t) * span, math.sin(t) * span), (255, 120, 0), 2)
    for tr in vis.visitors():
        col = {"engaged": (0, 160, 0), "approaching": (0, 120, 255)}.get(tr.state, (200, 0, 0))
        p = px(tr.s[0], tr.s[1])
        cv2.circle(img, p, 9, col, -1)
        cv2.arrowedLine(img, p, px(tr.s[0] + tr.s[2], tr.s[1] + tr.s[3]), col, 2, tipLength=0.3)
        spk = " (speaking)" if tr.t - tr.speaking_t < 2 else ""
        cv2.putText(img, f"{tr.id}: {tr.dist:.1f}m {tr.state}{spk}", (p[0] + 12, p[1] + 4), 0, 0.45, col, 1)
    cv2.circle(img, (c, c), 5, (0, 0, 0), -1)
    cv2.putText(img, title, (8, 18), 0, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "blue ring 1.2 m personal | green 2.5 m engage | grey 3.6 m social", (8, size - 8), 0, 0.4,
                (80, 80, 80), 1)
    return img


# ------------------------------------------------------------------ commands

def cmd_watch(src_run=None, seconds=None):
    from iro.record import run
    from iro.stream import feed
    from iro.surveil import Surveil
    r = run("visitors" + ("-replay" if src_run else ""))
    vis = Visitors()
    sv = Surveil()
    if src_run:
        lidar = mic = Replay(src_run)
    else:
        lidar = Lidar()
        try:
            mic = MicThread()
        except Exception as e:                     # noqa: BLE001 - lidar alone still works
            print(f"(no mic array: {e})")
            mic = None
    f = feed()
    raw_t, raw_r, mic_log = [], [], []
    n, t_start, n_ev, n_scans = None, time.time(), 0, 0
    print("Watching for visitors (Ctrl+C to stop). Live picture: the live page.")
    try:
        while seconds is None or time.time() - t_start < seconds:
            try:
                n, (t, a, rr) = lidar.read(n)
            except EOFError:
                break
            m = mic.mic_at(t) if mic else None
            if m and not src_run:
                mic_log.append(m)
            dets = vis.step(t, a, rr, m)
            for text in sv.step(t, a, rr, vis):
                vis.event(t, text)
            if n_scans == 20:                      # 2 s in: does the room still look as learned?
                fg = float(np.mean(sv.changes.close > 0)) if sv.changes is not None else 0.0
                if fg > 0.10:
                    print(f"  (the room looks different from when it was learned: {fg:.0%} of the view changed. "
                          "If ARC or the furniture moved, re-learn it: iro visitors learn)")
            n_scans += 1
            if not src_run:
                raw_t.append(t)
                raw_r.append(rr.astype(np.float16))
            for tt, text in vis.events[n_ev:]:
                print(f"  {time.strftime('%H:%M:%S', time.localtime(tt))}  {text}")
                r.note(text)
            n_ev = len(vis.events)
            img = sv.draw(radar(a, rr, dets, vis, vis.voice,
                                title=f"visitors: {len(vis.visitors())}   {vis.summary()[:60]}"))
            f.show(img, "")
            r.video("radar", img)
            for tr in vis.visitors():
                r.log("tracks", t=round(t, 3), id=tr.id, x=round(tr.s[0], 3), y=round(tr.s[1], 3),
                      vx=round(tr.s[2], 3), vy=round(tr.s[3], 3), dist=round(tr.dist, 3), state=tr.state,
                      speaking=int(tr.t - tr.speaking_t < 2))
    except KeyboardInterrupt:
        pass
    finally:
        lidar.close()
        if mic and not src_run:
            mic.close()
        if raw_t:
            np.savez_compressed(r.dir / "scans.npz", t=np.array(raw_t), a=a, r=np.array(raw_r))
            (r.dir / "mic.json").write_text(json.dumps(mic_log))
        sv.stats.save(r.dir, len(vis.visitors()))
        s = sv.stats.summary(len(vis.visitors()))
        print(f"\n{s['total']} visitors seen, {s['engaged']} engaged with ARC, mean stay {s['mean_stay_s']} s. "
              f"Statistics and heat map: {r.dir}")
    return 0


def cmd_learn(seconds=10):
    from iro.record import run
    r = run("visitors-learn")
    input(f"Learning the empty room: move everyone (and yourself) out of the lidar's view, put the arm in its "
          f"lidar-safe pose (iro visitors arm), then Enter. It takes {seconds} s. ")
    lidar = Lidar()
    try:
        a, med, mad, sm = learn_room(lidar, seconds)
    finally:
        lidar.close()
    blocked = float(sm.sum() * abs(float(a[1] - a[0])) * 180 / math.pi)
    open_ = float(np.mean(~np.isfinite(med)) * 360)
    r.note(f"room learned: {len(a)} beams, robot blocks {blocked:.0f} deg, open (no echo) {open_:.0f} deg")
    img = radar(a, np.where(np.isfinite(med), med, np.nan), [], type("V", (), {"med": med, "self_mask": sm,
                                                                             "visitors": lambda self: []})(),
                title="learned room")
    r.snap("room", img)
    print(f"Learned the room: {len(a)} beams; the robot itself blocks {blocked:.0f} deg of the view "
          f"(orange in the picture). Saved {BG.relative_to(ROOT)}. Next: iro visitors")
    return 0


def cmd_arm():
    """Find the arm pose that blocks the lidar least, by hand, with a live readout."""
    from iro.motion import Arm
    from iro.record import run
    from iro.skills import inside
    if input("The arm goes LIMP so you can pose it. Hold it. Ready? [y/N] ").strip().lower() != "y":
        return 1
    r = run("visitors-arm")
    lidar = Lidar()
    lines = []
    threading.Thread(target=lambda: [lines.append(x.strip()) for x in sys.stdin], daemon=True).start()
    best = None
    try:
        with Arm(dry_run=True) as arm:
            arm.bus.disable_torque()
            print("Move the arm; the number is how much of the lidar's view something within 60 cm blocks.\n"
                  "Enter = save this pose as lidar-safe, q + Enter = finish.")
            n = None
            while True:
                n, (_, a, rr) = lidar.read(n)
                b = blocked_deg(a, rr)
                q = arm.joints()[:5]
                print(f"\r  blocked {b:5.1f} deg   joints {np.round(np.degrees(q)).astype(int).tolist()}      ",
                      end="", flush=True)
                r.log("arm", blocked_deg=round(b, 1), joints_deg=np.round(np.degrees(q), 1).tolist())
                if lines:
                    w = lines.pop(0)
                    if w == "q":
                        break
                    q, notes = inside(arm, q)
                    best = (b, np.round(np.degrees(q), 1).tolist())
                    print(f"\n  saved: {best[0]:.1f} deg blocked at {best[1]}" + (f" ({', '.join(notes)})" if notes else ""))
    finally:
        lidar.close()
    if best:
        save_cfg(lidar_safe_joints_deg=best[1], lidar_safe_blocked_deg=round(best[0], 1))
        r.note(f"lidar-safe pose {best}")
        print(f"\nSaved the lidar-safe pose in {CFG.relative_to(ROOT)}. The arm stays limp: rest it in that pose.")
    return 0


def to_lidar_safe(arm):
    """Move the arm into the saved lidar-safe pose (before watching for visitors)."""
    q = cfg().get("lidar_safe_joints_deg")
    if q is None:
        raise RuntimeError("no lidar-safe pose yet: iro visitors arm")
    arm.move_joints(np.radians(q), duration=2.5)


def fit_mic(pairs):
    """(sign, offset_deg, median_err, p90_err) mapping mic angles to lidar bearings: circular mean
    of the difference for both turning directions, then again without the worst 20 %."""
    L, M = np.radians(np.array(pairs)).T
    best = None
    for sign in (1, -1):
        d = L - sign * M
        off = math.atan2(np.sin(d).mean(), np.cos(d).mean())
        for _ in range(2):                         # refit on the best 80 % (stale or reflected readings)
            res = np.abs((d - off + np.pi) % (2 * np.pi) - np.pi)
            keep = res <= np.percentile(res, 80)
            off = math.atan2(np.sin(d[keep]).mean(), np.cos(d[keep]).mean())
        res = np.degrees(np.abs((d - off + np.pi) % (2 * np.pi) - np.pi))
        cand = (sign, math.degrees(off), float(np.median(res)), float(np.percentile(res, 90)))
        if best is None or cand[2] < best[2]:
            best = cand
    return best


def cmd_mic_cal(seconds=60):
    """Talk continuously while walking slowly ALL the way round the robot (alone, 1-3 m away). Pairs of
    (lidar bearing of the only visitor, each mic direction reading) give the mic's rotation and
    direction, and which reading follows a person best."""
    from iro.record import run
    r = run("visitors-mic-cal")
    input(f"Mic calibration: be the ONLY person near the robot. After Enter, talk (count, read aloud) while "
          f"walking slowly ALL the way round it (a full circle), 1-3 m away, for {seconds} s. ")
    vis, lidar, mic = Visitors(), Lidar(), MicThread()
    pairs = {"speaker": [], "doa": [], "free": []}
    n, t0 = None, time.time()
    try:
        while time.time() - t0 < seconds:
            n, (t, a, rr) = lidar.read(n)
            vis.step(t, a, rr)
            m = mic.mic_at(t)
            v = vis.visitors()
            if m and len(v) == 1:
                for k, deg in (m.get("dirs") or {}).items():
                    if deg is not None:
                        pairs[k].append((v[0].bearing, deg))
                r.log("pairs", lidar_deg=round(v[0].bearing, 1),
                      **{k: (round(x, 1) if x is not None else "") for k, x in (m.get("dirs") or {}).items()})
            cover = np.ptp([p[0] for p in pairs["doa"] or pairs["speaker"] or [(0, 0)]])
            print(f"\r  {max(len(x) for x in pairs.values())} moments, circle covered ~{cover:.0f} deg, "
                  f"{len(v)} visitor(s) seen   ", end="", flush=True)
    finally:
        lidar.close()
        mic.close()
    fits = {k: fit_mic(p) for k, p in pairs.items() if len(p) >= 20}
    if not fits:
        print("\nToo few usable moments (need 20): talk louder / stay alone in view, then try again.")
        return 1
    for k, f in fits.items():
        print(f"\n  {k:8s} {len(pairs[k]):3d} moments: typical error {f[2]:4.0f} deg, 90% within {f[3]:4.0f}")
        r.note(f"mic cal {k}: sign {f[0]}, offset {f[1]:.1f}, median {f[2]:.1f}, p90 {f[3]:.1f}, n {len(pairs[k])}")
    src = min(fits, key=lambda k: fits[k][2])
    sign, off, med_err, p90 = fits[src]
    bear = [p[0] for p in pairs[src]]
    cover = float(np.ptp(bear))
    save_cfg(mic_source=src, mic_sign=sign, mic_offset_deg=round(off, 1), mic_cal_median_err_deg=round(med_err, 1))
    print(f"\nUsing the '{src}' reading: angle = {sign:+d} x mic {off:+.1f} deg; typical error {med_err:.0f} deg "
          f"(90% within {p90:.0f}). Saved {CFG.relative_to(ROOT)}.")
    if cover < 270:
        print(f"  Only ~{cover:.0f} deg of the circle was covered: run it again walking all the way round.")
    return 0


def main():
    a = sys.argv[1:]
    if a[:1] == ["learn"]:
        return cmd_learn()
    if a[:1] == ["arm"]:
        return cmd_arm()
    if a[:1] == ["mic-cal"]:
        return cmd_mic_cal()
    if a[:1] == ["zone"]:
        from iro.surveil import cmd_zone
        return cmd_zone(a[1:])
    if "--from" in a:
        from iro.record import RUNS
        p = a[a.index("--from") + 1]
        return cmd_watch(src_run=RUNS / p if not p.startswith("/") else Path(p))
    if not a:
        return cmd_watch()
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
