"""The mecanum base (TETRIX PRIZM Pro running the PUSAKA motor firmware), in the ARM's frame:
x forward (where the arm and the OAK-D look), y to the left, turning counter-clockwise positive;
metres per second and radians per second.

    iro base ping                is the motor program answering (press the green START first)
    iro base calibrate           measure speed per power on each axis with the lidar (scan matching);
                                 needs ~30 cm of free floor all round
    iro base nudge DX DY [DEG]   move by that much (cm, cm, degrees), slowly, then stop

Firmware (backup/esp32s3/README.md): `MOVE a b c` sets power -100..100 and the board mixes the four
wheels itself. Measured 21 Nov 2026 with the OAK-D watching the tape:
    b > 0 = forward, a > 0 = to the RIGHT, c > 0 = turn CLOCKWISE  (so a = -y, b = x, c = -yaw)
    `WDT ms` stops the motors when no MOVE arrives for that long (MOTOR does not feed it).
Power is not speed (open loop): `iro base calibrate` fits speed = gain * (power - deadband) per axis
into config/base.yaml. The approach to an object steers by what the cameras see, so it corrects for
the rest.

Safety: the watchdog (300 ms) stops the wheels if this program hangs or dies; every exit path sends
STOP; speeds are capped (MAX_V, MAX_W). A hardware E-stop that cuts motor power is still needed
before driving near people (PLAN.md).
"""

import atexit
import math
import sys
import threading
import time

import numpy as np
import yaml

from iro import ROOT, load_config

CFG = ROOT / "config" / "base.yaml"
MAX_V = 0.15            # m/s, any direction
MAX_W = 0.6             # rad/s
WDT_MS = 300
DEFAULTS = {            # measured 21 Nov with 0.6 s pulses at power 20 (incl. the start); calibrate replaces
    "gain_x": 0.0016, "gain_y": 0.0016, "gain_w": 0.0090,      # m/s (rad/s) per power above the deadband
    "dead_x": 10.0, "dead_y": 10.0, "dead_w": 10.0,            # power that only just starts it moving
    "lidar_yaw_deg": 180.0,                                    # lidar angle of the robot's forward (mounted backwards)
    "cnt_x": 4800.0, "cnt_y": 5200.0, "cnt_w": 1700.0,         # encoder counts per metre / radian
}


def settings():
    d = dict(DEFAULTS)
    if CFG.exists():
        d.update(yaml.safe_load(CFG.read_text()) or {})
    return d


class Base:
    def __init__(self, port=None):
        import serial
        cfg = load_config()["devices"]["base_controller"]
        port = port or cfg.get("port", "/dev/base_esp32")
        # pyserial's defaults (DTR, RTS on): setting them off resets the ESP32-S3 (backup README)
        self.s = serial.Serial(port, 115200, timeout=0.05)
        self.k = settings()
        self.lock = threading.Lock()
        self.cmd = (0, 0, 0)
        self.last_reply = ""
        self.run = True
        if "PONG" not in self._ask("PING", 0.3):
            self.s.close()
            raise RuntimeError("the base does not answer PING: press the green START on the PRIZM Pro")
        self._ask(f"WDT {WDT_MS}")
        self.th = threading.Thread(target=self._feed, daemon=True)
        self.th.start()
        atexit.register(self.close)

    def _ask(self, line, wait=0.06):
        with self.lock:
            self.s.reset_input_buffer()
            self.s.write((line + "\n").encode())
            time.sleep(wait)
            r = self.s.read(256).decode(errors="replace").strip()
        self.last_reply = r
        return r

    def _feed(self):
        """Resend the current command 20x a second: keeps the firmware watchdog from stopping it."""
        while self.run:
            a, b, c = self.cmd
            try:
                with self.lock:
                    self.s.write(f"MOVE {a} {b} {c}\n".encode())
                    self.s.read(64)
            except Exception:                       # noqa: BLE001 - the watchdog stops the wheels
                return
            time.sleep(0.05)

    # --- driving ----------------------------------------------------------------
    @staticmethod
    def _power(v, gain, dead):
        if abs(v) < 1e-4:
            return 0.0
        return math.copysign(dead + abs(v) / gain, v)

    def drive(self, vx, vy, wz):
        """Velocities in the arm's frame (m/s, m/s, rad/s), capped at MAX_V / MAX_W."""
        sp = math.hypot(vx, vy)
        if sp > MAX_V:
            vx, vy = vx * MAX_V / sp, vy * MAX_V / sp
        wz = max(-MAX_W, min(MAX_W, wz))
        k = self.k
        px = self._power(vx, k["gain_x"], k["dead_x"])
        py = self._power(vy, k["gain_y"], k["dead_y"])
        pw = self._power(wz, k["gain_w"], k["dead_w"])
        # firmware: a = right, b = forward, c = clockwise
        self.cmd = tuple(int(round(max(-60, min(60, p)))) for p in (-py, px, -pw))

    def move_by(self, dx, dy, dth=0.0, v=0.06, w=0.35, tol=0.004, tol_th=0.02, timeout=None, check=None):
        """Move by (dx forward, dy left) metres and dth radians (counter-clockwise), closed loop on the
        wheel encoders (odometry, ~1-2 % of the distance). Slows down near the end. check(): called
        ~20x a second, return True to stop at once (e.g. something in the way). Returns (moved x, y, th)."""
        e0 = self.enc()
        if e0 is None:
            raise RuntimeError("no encoder reading from the base")
        dist = math.hypot(dx, dy)
        timeout = timeout or 3.0 + 2.5 * dist / v + 2.5 * abs(dth) / w
        t0 = time.time()
        x = y = th = 0.0
        try:
            while time.time() - t0 < timeout:
                e = self.enc()
                if e is None:
                    continue
                x, y, th = odometry(e - e0, self.k)
                ex, ey, eth = dx - x, dy - y, dth - th
                if math.hypot(ex, ey) < tol and abs(eth) < tol_th:
                    break
                if check is not None and check():
                    break
                # P control with a speed cap, and at least enough to keep moving
                vx, vy, wz = 2.0 * ex, 2.0 * ey, 2.5 * eth
                sp = math.hypot(vx, vy)
                if sp > v:
                    vx, vy = vx * v / sp, vy * v / sp
                elif 0 < sp < 0.02 and math.hypot(ex, ey) >= tol:
                    vx, vy = vx * 0.02 / sp, vy * 0.02 / sp
                wz = max(-w, min(w, wz))
                if abs(eth) >= tol_th and abs(wz) < 0.08:
                    wz = math.copysign(0.08, wz)
                self.drive(vx, vy, wz)
                time.sleep(0.03)
        finally:
            self.stop()
        return x, y, th

    def raw(self, a, b, c):
        """Firmware MOVE powers directly (calibration only)."""
        self.cmd = (int(a), int(b), int(c))

    def stop(self):
        self.cmd = (0, 0, 0)
        self._ask("STOP")

    def enc(self):
        r = self._ask("ENC", 0.08)
        try:
            i = r.index("ENC ")
            return np.array([int(v) for v in r[i + 4:].split()[:4]])
        except (ValueError, IndexError):
            return None

    def close(self):
        if not getattr(self, "run", False):
            return
        self.run = False
        try:
            self.th.join(timeout=0.5)
            self._ask("MOVE 0 0 0")
            self._ask("STOP")
            self.s.close()
        except Exception:                           # noqa: BLE001
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ------------------------------------------------------------------ odometry and calibration
#
# Encoder patterns measured 21 Nov (power 30, 0.8 s): forward = [-,+,-,+], left = [-,-,+,+],
# counter-clockwise = [+,+,+,+]. So with d = the change of the four encoder counts:
#     forward = (-d1 + d2 - d3 + d4) / 4 / cnt_x,  left = (-d1 - d2 + d3 + d4) / 4 / cnt_y,
#     ccw     = ( d1 + d2 + d3 + d4) / 4 / cnt_w     (counts per metre / per radian, calibrated)


def odometry(d, k=None):
    """(forward m, left m, ccw rad) from a change of the four encoder counts."""
    k = k or settings()
    d = np.asarray(d, float)
    return ((-d[0] + d[1] - d[2] + d[3]) / 4 / k["cnt_x"], (-d[0] - d[1] + d[2] + d[3]) / 4 / k["cnt_y"],
            (d[0] + d[1] + d[2] + d[3]) / 4 / k["cnt_w"])


def icp(A, B, iters=40, th0=0.0, t0=None, na=1500, nb=600):
    """Rigid 2D fit: (theta, t) with R(theta) B + t ~ A (point sets, metres), starting from (th0, t0).
    Returns (theta, t, median residual). Brute-force nearest neighbours on samples: ~0.3 s at the
    default sizes, ~0.05 s for tracking (na=700, nb=250, iters=12, warm start)."""
    rng = np.random.default_rng(0)
    A = A[rng.choice(len(A), min(na, len(A)), replace=False)]
    B = B[rng.choice(len(B), min(nb, len(B)), replace=False)]
    th, t = th0, (np.zeros(2) if t0 is None else np.asarray(t0, float))
    res = 1.0
    for _ in range(iters):
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        Q = B @ R.T + t
        d2 = ((Q[:, None, :] - A[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(1)
        dd = np.sqrt(d2[np.arange(len(Q)), j])
        keep = dd < max(0.05, np.percentile(dd, 70))
        P, M = B[keep], A[j[keep]]
        mp, mm = P.mean(0), M.mean(0)
        H = (P - mp).T @ (M - mm)
        th = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        t = mm - R @ mp
        res = float(np.median(dd[keep]))
    return th, t, res


class LidarOdom:
    """Where the robot is now relative to where it was at reset(), from lidar scan matching against
    that first scan (walls and furniture do not slip; wheels do: 6-9 deg of unnoticed turn on a 10 cm
    straight move, 21 Nov). Good for moves of up to ~1 m while the room stays mostly the same."""

    def __init__(self, lp):
        self.lp = lp
        self.reset()

    def reset(self):
        self.ref = self.lp.points(scans=3)
        self.th, self.t = 0.0, np.zeros(2)

    def pose(self):
        """(x, y, th) of the robot in the frame it had at reset()."""
        S = self.lp.points(scans=1)
        # scan S (robot frame now) -> reference frame: ref ~ R(th) S + t  (= the robot's pose)
        self.th, self.t, self.fit = icp(self.ref, S, iters=12, th0=self.th, t0=self.t, na=900, nb=300)
        return float(self.t[0]), float(self.t[1]), float(self.th)


def move_by_lidar(b, lo, dx, dy, dth=0.0, v=0.07, w=0.35, tol=0.008, tol_th=math.radians(2.0), timeout=None,
                  check=None):
    """Move by (dx forward, dy left) metres and dth radians, steered by lidar odometry (LidarOdom).
    Returns the final (x, y, th) error-free estimate. check() -> True stops at once."""
    lo.reset()
    timeout = timeout or 4.0 + 3 * math.hypot(dx, dy) / v + 3 * abs(dth) / w
    t0 = time.time()
    x = y = th = 0.0
    LAG = 0.30                       # s: one scan match (~0.15 s) + the wheels coasting after STOP
    sp = wz = 0.0
    best, best_t = None, time.time()
    try:
        while time.time() - t0 < timeout:
            x, y, th = lo.pose()
            ex_w, ey_w, eth = dx - x, dy - y, dth - th
            c, s_ = math.cos(th), math.sin(th)
            ex, ey = c * ex_w + s_ * ey_w, -s_ * ex_w + c * ey_w          # error in the robot's frame now
            dist = math.hypot(ex, ey)
            # where it will be by the time this command acts: subtract what it still travels meanwhile
            ahead, ahead_th = dist - sp * LAG, eth - wz * LAG
            if (ahead < tol or dist < tol) and (abs(ahead_th) < tol_th or abs(eth) < tol_th):
                break
            score = dist / tol + abs(eth) / tol_th
            if best is None or score < best - 0.2:
                best, best_t = score, time.time()
            elif time.time() - best_t > 1.2 and dist < 2 * tol and abs(eth) < 2 * tol_th:
                break                                     # creeping without getting closer: good enough
            if check is not None and check():
                break
            sp = min(v, max(0.03, 1.2 * ahead)) if ahead >= tol else 0.0
            vx, vy = (ex / dist * sp, ey / dist * sp) if sp else (0.0, 0.0)
            wz = 0.0 if abs(ahead_th) < tol_th else math.copysign(min(w, max(0.12, 2.0 * abs(ahead_th))), ahead_th)
            b.drive(vx, vy, wz)
    finally:
        b.stop()
    return x, y, th


class LidarPoints:
    """Lidar scans as 2D points (metres) in the ROBOT's frame (x forward), using lidar_yaw_deg."""

    def __init__(self, lidar=None):
        if lidar is None:
            from iro.visitors import Lidar
            lidar = Lidar()
        self.li, self.n = lidar, None
        y = settings().get("lidar_yaw_deg")
        self.yaw = math.radians(y if y is not None else 0.0)

    def points(self, scans=4, rmin=0.2, rmax=5.0):
        P = []
        for _ in range(scans):
            self.n, (t, a, r) = self.li.read(self.n)
            ok = np.isfinite(r) & (r > rmin) & (r < rmax)
            P.append(np.c_[r[ok] * np.cos(a[ok] - self.yaw), r[ok] * np.sin(a[ok] - self.yaw)])
        return np.vstack(P)

    def raw(self):
        """(angles in the robot frame, ranges) of the newest scan."""
        self.n, (t, a, r) = self.li.read(self.n)
        return (a - self.yaw + math.pi) % (2 * math.pi) - math.pi, r

    def close(self):
        self.li.close()


def cmd_calibrate():
    """Pulses on each axis at two powers; the lidar (scan matching, ~7 mm) measures each one. Fits
    speed per power, the power where it starts moving, the encoder scale and the lidar's mounting
    angle. Needs ~30 cm of free floor all round and walls or furniture within 5 m."""
    from iro.record import run
    r = run("base-calibrate")
    lp = LidarPoints()
    lp.yaw = 0.0                                    # measure in the lidar's own frame first
    rows = []
    T = 0.8
    plan = [(0, 20, 0), (0, -20, 0), (0, 40, 0), (0, -40, 0), (-20, 0, 0), (20, 0, 0), (-40, 0, 0), (40, 0, 0),
            (0, 0, -20), (0, 0, 20), (0, 0, -40), (0, 0, 40)]
    try:
        with Base() as b:
            for cmd in plan:
                S0, e0 = lp.points(), b.enc()
                b.raw(*cmd)
                time.sleep(T)
                b.stop()
                time.sleep(0.7)
                S1, e1 = lp.points(), b.enc()
                th, t, fit = icp(S1, S0)
                move = -np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]]) @ t   # robot motion, lidar frame
                d = None if e0 is None or e1 is None else (e1 - e0)
                rows.append((cmd, move, -th, d, fit))
                r.log("cal", cmd=str(cmd), dx=move[0], dy=move[1], dth=-th, enc=str(None if d is None else d.tolist()), fit=fit)
                print(f"  MOVE {cmd}: {np.round(move * 100, 1).tolist()} cm, {math.degrees(-th):+.1f} deg (lidar frame, "
                      f"fit {fit * 100:.1f} cm)  enc {None if d is None else d.tolist()}", flush=True)
    finally:
        lp.close()
    k = settings()
    fwd = [m for c, m, *_ in rows if c[1] > 0]
    lidar_yaw = math.degrees(math.atan2(np.mean(fwd, 0)[1], np.mean(fwd, 0)[0]))   # lidar angle of "forward"
    k["lidar_yaw_deg"] = round(lidar_yaw, 1)
    R = np.array([[math.cos(math.radians(-lidar_yaw)), -math.sin(math.radians(-lidar_yaw))],
                  [math.sin(math.radians(-lidar_yaw)), math.cos(math.radians(-lidar_yaw))]])
    fits = {}
    for axis, sel, val in (("x", lambda c: c[1] != 0, lambda m, th: (R @ m)[0]),
                           ("y", lambda c: c[0] != 0, lambda m, th: (R @ m)[1]),
                           ("w", lambda c: c[2] != 0, lambda m, th: th)):
        pts = [(c, m, th, d) for c, m, th, d, f in rows if sel(c)]
        # power in the robot's sense: x = b, y = -a (a is "right"), w = -c (c is clockwise)
        P = np.array([c[1] if axis == "x" else (-c[0] if axis == "y" else -c[2]) for c, *_ in pts], float)
        V = np.array([val(m, th) / T for c, m, th, d in pts])
        A = np.c_[np.abs(P), np.ones(len(P))]
        g, c0 = np.linalg.lstsq(A, V * np.sign(P), rcond=None)[0]
        if g > 0:
            k[f"gain_{axis}"], k[f"dead_{axis}"] = round(float(g), 5), round(float(max(0.0, -c0 / g)), 1)
        # encoder scale: counts per metre (radian) of the measured motion
        sig = {"x": np.array([-1, 1, -1, 1]), "y": np.array([-1, -1, 1, 1]), "w": np.array([1, 1, 1, 1])}[axis]
        cnt = [float(sig @ d) / 4 / (val(m, th)) for c, m, th, d in pts if d is not None and abs(val(m, th)) > 0.01]
        if cnt:
            k[f"cnt_{axis}"] = round(float(np.median(cnt)), 1)
        fits[axis] = (g, c0, np.round(V, 3).tolist())
    CFG.write_text("# Written by `iro base calibrate` (lidar-measured; see iro/base.py). speed = gain * (power - dead)\n"
                   + yaml.safe_dump({kk: v for kk, v in k.items()}))
    print(f"Saved {CFG.name}: lidar forward at {lidar_yaw:.0f} deg; "
          + "; ".join(f"{a}: {k['gain_' + a]:.4f}/power from {k['dead_' + a]:.0f}, {k.get('cnt_' + a, 0):.0f} counts/{'rad' if a == 'w' else 'm'}"
                      for a in "xyw"))
    return 0


def cmd_nudge(dx, dy, ddeg):
    """Move by (dx, dy) metres and ddeg degrees, open loop at a gentle speed."""
    v, w = 0.06, 0.4
    t = max(math.hypot(dx, dy) / v, abs(math.radians(ddeg)) / w, 0.01)
    with Base() as b:
        b.drive(dx / t, dy / t, math.radians(ddeg) / t)
        time.sleep(t)
        b.stop()
    return 0


def main():
    a = sys.argv[1:] or ["ping"]
    if a[0] == "ping":
        try:
            with Base():
                print("PONG: the base answers; watchdog set to", WDT_MS, "ms")
            return 0
        except Exception as e:                      # noqa: BLE001
            print(e)
            return 1
    if a[0] == "calibrate":
        return cmd_calibrate()
    if a[0] == "nudge" and len(a) >= 3:
        return cmd_nudge(float(a[1]) / 100, float(a[2]) / 100, float(a[3]) if len(a) > 3 else 0.0)
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
