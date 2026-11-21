"""The mecanum base (TETRIX PRIZM Pro running the PUSAKA motor firmware), in the ARM's frame:
x forward (where the arm and the OAK-D look), y to the left, turning counter-clockwise positive;
metres per second and radians per second.

    iro base ping                is the motor program answering (press the green START first)
    iro base calibrate           measure speed per power on each axis, with the OAK-D watching an
                                 object on the floor (needs ~40 cm of free floor all round)
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
    "lidar_yaw_deg": None,                                     # lidar angle of the robot's forward (calibrate)
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


# ------------------------------------------------------------------ calibration

def landmark_motion(p0, p1, theta=None):
    """Robot motion (dx, dy in its frame at the start, dyaw) from one floor point seen before (p0)
    and after (p1), both in the arm frame. A point gives 2 numbers, so either the turn is known
    (theta) and the shift is solved, or the shift is taken as zero and the turn is solved."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    if theta is None:                               # pure turn: p1 = R(-th) p0
        th = math.atan2(p0[1], p0[0]) - math.atan2(p1[1], p1[0])
        return 0.0, 0.0, (th + math.pi) % (2 * math.pi) - math.pi
    c, s = math.cos(theta), math.sin(theta)
    d = p0 - np.array([[c, -s], [s, c]]) @ p1          # p0 = R(th) p1 + t
    return float(d[0]), float(d[1]), theta


def cmd_calibrate():
    from iro.oakguide import Scout
    from iro.record import run
    r = run("base-calibrate")
    sc = Scout()

    def where():
        ps = []
        for _ in range(2):
            o = sc.objects(3)
            if o:
                ps.append(o[0]["centre_raw"])
        return np.median(ps, 0) if ps else None
    rows = {"x": [], "y": [], "w": []}
    try:
        with Base() as b:
            if where() is None:
                print("The OAK-D needs one object on the floor 20-45 cm in front of the robot to watch.")
                return 1
            plan = [("x", (0, 25, 0)), ("x", (0, -25, 0)), ("x", (0, 40, 0)), ("x", (0, -40, 0)),
                    ("y", (-25, 0, 0)), ("y", (25, 0, 0)), ("y", (-40, 0, 0)), ("y", (40, 0, 0)),
                    ("w", (0, 0, -25)), ("w", (0, 0, 25)), ("w", (0, 0, -40)), ("w", (0, 0, 40))]
            T = 0.8
            for axis, cmd in plan:
                p0, e0 = where(), b.enc()
                b.raw(*cmd)
                time.sleep(T)
                b.stop()
                time.sleep(0.6)
                p1, e1 = where(), b.enc()
                if p0 is None or p1 is None:
                    print(f"  {cmd}: lost the landmark, skipped")
                    continue
                if axis == "w":
                    dx, dy, th = landmark_motion(p0, p1)
                    v = th / T
                    power = -cmd[2]
                else:
                    dx, dy, th = landmark_motion(p0, p1, 0.0)
                    v = (dx if axis == "x" else dy) / T
                    power = cmd[1] if axis == "x" else -cmd[0]
                rows[axis].append((power, v))
                de = None if e0 is None or e1 is None else (e1 - e0).tolist()
                r.log("cal", axis=axis, power=power, v=v, dx=dx, dy=dy, dth=th, enc=str(de))
                print(f"  {axis} power {power:+4d}: moved {dx * 100:+5.1f} cm fwd, {dy * 100:+5.1f} cm left, "
                      f"{math.degrees(th):+5.1f} deg  -> {v:+.3f} {'rad' if axis == 'w' else 'm'}/s  enc {de}")
    finally:
        sc.close()
    k = settings()
    for axis, pts in rows.items():
        if len(pts) < 3:
            continue
        P, V = np.abs([p for p, _ in pts]), np.array([v * np.sign(p) for p, v in pts])
        A = np.c_[P, np.ones(len(P))]
        g, c0 = np.linalg.lstsq(A, V, rcond=None)[0]      # v = g * P + c0  ->  dead = -c0 / g
        if g <= 0:
            print(f"  {axis}: the fit makes no sense (gain {g:.4f}); kept the old values")
            continue
        k[f"gain_{axis}"], k[f"dead_{axis}"] = round(float(g), 5), round(float(max(0.0, -c0 / g)), 1)
    CFG.write_text("# Written by `iro base calibrate` (speed = gain * (power - dead), per axis; see iro/base.py)\n"
                   + yaml.safe_dump({kk: v for kk, v in k.items()}))
    print(f"Saved {CFG.name}: " + ", ".join(f"{a}: {k['gain_' + a]:.4f}/power, starts at {k['dead_' + a]:.0f}"
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
