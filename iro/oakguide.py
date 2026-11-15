"""The OAK-D as the arm's scout: where things lie on the floor, in the arm's own coordinates, before
the arm moves. The wrist camera then only has to confirm and fine-centre (iro pick).

    iro pick scout           what the OAK-D sees on the floor right now (positions, sizes)

Why (15 Nov 2026): searching and centring with the wrist camera alone took 10-25 s per pick, and its
slanted view misjudged which way a standing tape was narrow (a jaw came down on it). The OAK-D sees
the true footprint from depth, and is calibrated to the arm without markers (iro efad calibrate,
9 mm). The arm at look height (15 cm) is ABOVE the OAK-D's field of view (its top picture row is
level with the lens, 11.5 cm up), so the scout can look at any time without parking the arm.

Method: depth points -> arm frame (T_base_cam) -> height above the floor (the floor's own small
tilt/offset is re-fitted every look) -> everything standing higher than MIN_H on a top-down grid ->
each blob: smallest rotated rectangle (length x width, direction), height = 98th percentile.
Limits: the OAK-D sees the floor from 18 cm in front of the arm base; depth works from ~11 cm from
the lens at 640x400. Things taller than the lens (11.5 cm) show only their front face, so their
centre is biased toward the robot (flagged `front_only`).
"""

import threading

import cv2
import numpy as np
import yaml

from iro import ROOT

CELL = 0.004               # m, top-down grid
MIN_H = 0.010              # m above the floor to count as part of an object
REGION = (0.14, 0.50, -0.30, 0.30)     # x0, x1, y0, y1 (m, arm frame): where the arm could ever pick
LENS_H = 0.115             # m above the floor


PAIRS = ROOT / "config" / "scout_pairs.yaml"


class Correction:
    """The scout's positions corrected by what the picks themselves found. Every successful
    scout-guided pick gives a pair: where the scout said the object was, and where the wrist camera
    and the grip then really found it. A rigid shift + turn fitted to the pairs (turn only once the
    pairs are spread over >= 8 cm; at most 12 deg) is applied to all later scout positions.
    15 Nov: the OAK-D had been 9 mm exact when calibrated (21 Oct), but after the move to the Orin NX
    it was 3-4.5 cm off, as a ~7 deg turn: the mount had shifted. Three picks' pairs remove most of
    it without a new calibration. A new `iro efad calibrate` empties the list (its date is stored)."""

    def __init__(self):
        self.theta, self.t, self.c, self.n = 0.0, np.zeros(2), np.zeros(2), 0
        self.load()

    @staticmethod
    def _cal_date():
        from iro.efad import CALIB
        try:
            return str(yaml.safe_load(CALIB.read_text()).get("date"))
        except (OSError, AttributeError):
            return None

    def load(self):
        self.pairs = []
        if PAIRS.exists():
            d = yaml.safe_load(PAIRS.read_text()) or {}
            if d.get("calibration_date") == self._cal_date():
                self.pairs = [p for p in d.get("pairs", [])][-40:]
        self.fit()

    def fit(self):
        self.theta, self.t, self.c, self.n = 0.0, np.zeros(2), np.zeros(2), len(self.pairs)
        if not self.pairs:
            return
        A = np.array([p["scout"] for p in self.pairs], float)
        B = np.array([p["arm"] for p in self.pairs], float)
        ca, cb = A.mean(0), B.mean(0)
        if len(A) >= 3 and np.linalg.norm(A - ca, axis=1).max() > 0.04:
            H = (A - ca).T @ (B - cb)
            self.theta = float(np.clip(np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]), -0.21, 0.21))
        self.c, self.t = ca, cb - ca

    def rot(self):
        c, s_ = np.cos(self.theta), np.sin(self.theta)
        return np.array([[c, -s_], [s_, c]])

    def point(self, p):
        return self.rot() @ (np.asarray(p, float) - self.c) + self.c + self.t

    def direction(self, v):
        return self.rot() @ np.asarray(v, float)

    def add(self, scout_xy, arm_xy, name=""):
        """Remember one pick's pair (raw scout position, where it really was) and refit."""
        import time
        d = np.linalg.norm(np.asarray(scout_xy) - np.asarray(arm_xy))
        if d > 0.08:                                   # not a calibration error any more: do not learn it
            return False
        self.pairs.append({"scout": [round(float(v), 4) for v in scout_xy], "arm": [round(float(v), 4) for v in arm_xy],
                           "object": name, "date": time.strftime("%Y-%m-%d %H:%M")})
        self.pairs = self.pairs[-40:]
        self.fit()
        PAIRS.write_text("# Written by iro pick: where the OAK-D scout said an object was vs where the pick found it\n"
                         "# (iro.oakguide.Correction). Emptied by a new `iro efad calibrate`.\n"
                         + yaml.safe_dump({"calibration_date": self._cal_date(), "pairs": self.pairs}))
        return True

    def residual_mm(self):
        if not self.pairs:
            return None
        return float(np.median([np.linalg.norm(self.point(p["scout"]) - p["arm"]) for p in self.pairs]) * 1000)


class Scout:
    def __init__(self, oak=None, ir=0.5):
        from iro.efad import load_calib
        from iro.pick import pick_config
        if oak is None:
            from iro.oak import Oak
            oak = Oak(ir=ir)
        self.oak, self.T, self.table = oak, load_calib(), pick_config()["table_z"]
        if self.T is None:
            raise RuntimeError("the OAK-D is not calibrated to the arm yet: iro efad calibrate")
        self.corr = Correction()
        self.lock = threading.Lock()               # the side-view recorder reads the same camera
        self.last = None                           # (colour picture, objects) of the last look

    def read(self, n=1):
        with self.lock:
            return self.oak.read(n)

    def objects(self, n=3):
        """Everything standing on the floor inside REGION: dicts with centre (x, y in m, arm frame),
        yaw (rad, direction of the long side), length, width, height (m), front_only, box (pixels)."""
        img, depth = self.read(n)
        K = self.oak.K
        v, u = np.nonzero((depth > 100) & (depth < 800))
        z = depth[v, u] / 1000.0
        B = np.c_[(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z] @ self.T[:3, :3].T + self.T[:3, 3]
        h = B[:, 2] - self.table
        # the floor as measured NOW (removes the calibration's few mm of offset and tilt)
        A = np.c_[np.ones(len(B)), B[:, :2]]
        sel = np.abs(h) < 0.02
        for _ in range(3):
            if sel.sum() < 500:
                break
            coef = np.linalg.lstsq(A[sel], h[sel], rcond=None)[0]
            res = h - A @ coef
            sel = np.abs(res) < max(0.004, 2.5 * np.std(res[sel]))
        else:
            h = res
        x0, x1, y0, y1 = REGION
        inr = (B[:, 0] > x0) & (B[:, 0] < x1) & (B[:, 1] > y0) & (B[:, 1] < y1) & (h > MIN_H) & (h < 0.30)
        out = []
        if inr.sum() >= 40:
            gx = ((B[inr, 0] - x0) / CELL).astype(int)
            gy = ((B[inr, 1] - y0) / CELL).astype(int)
            grid = np.zeros((int((x1 - x0) / CELL) + 1, int((y1 - y0) / CELL) + 1), np.uint8)
            np.add.at(grid, (gx, gy), 1)
            occ = cv2.morphologyEx((grid >= 2).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            k, lab = cv2.connectedComponents(occ)
            cell = lab[gx, gy]
            Bi, hi, ui, vi = B[inr], h[inr], u[inr], v[inr]
            for c in range(1, k):
                m = cell == c
                if m.sum() < 60 or (lab == c).sum() < 12:
                    continue
                height = float(np.percentile(hi[m], 98))
                if height < 0.012:
                    continue
                top = m & (hi > max(MIN_H, 0.3 * height))          # low "flying pixels" smear the outline
                pts = Bi[top, :2] if top.sum() > 30 else Bi[m, :2]
                mu = pts.mean(0)
                ev = np.linalg.eigh(np.cov((pts - mu).T))[1]
                pr = (pts - mu) @ ev
                lo, hi_ = np.percentile(pr, 1.5, 0), np.percentile(pr, 98.5, 0)
                pts = pts[np.all((pr >= lo) & (pr <= hi_), 1)]
                (cx, cy), (a, b), ang = cv2.minAreaRect(pts.astype(np.float32))
                yaw = np.radians(ang if a >= b else ang + 90)
                # the same blob in the camera's own directions: ACROSS the line of sight is seen in
                # full; ALONG it only the near face is seen (the far side hides behind the object)
                cam = self.T[:2, 3]
                ev_ = np.array([cx, cy]) - cam
                ev_ /= np.linalg.norm(ev_)
                ep_ = np.array([-ev_[1], ev_[0]])
                al, ac = (pts - cam) @ ev_, (pts - cam) @ ep_
                near, far = np.percentile(al, 2), np.percentile(al, 98)
                a0, a1 = np.percentile(ac, 2), np.percentile(ac, 98)
                yaw += self.corr.theta
                raw_c = np.array([cx, cy])
                front_raw = cam + near * ev_ + (a0 + a1) / 2 * ep_
                ev_ = self.corr.direction(ev_)
                out.append({"centre": self.corr.point(raw_c), "centre_raw": raw_c, "front_raw": front_raw,
                            "yaw": float((yaw + np.pi / 2) % np.pi - np.pi / 2),
                            "length": float(max(a, b)), "width": float(min(a, b)), "height": height,
                            "front_only": bool(height > LENS_H - 0.01), "points": int(m.sum()),
                            "bottom": float(np.percentile(hi[m], 3)),       # > ~4 cm: held in the air
                            "view": ev_, "front": self.corr.point(front_raw),
                            "across": float(a1 - a0), "along_seen": float(far - near),
                            "box": [int(ui[m].min()), int(vi[m].min()), int(ui[m].max()), int(vi[m].max())]})
        out.sort(key=lambda o: -o["points"])
        self.last = (img, out)
        return out

    def find(self, name=None, near=None, n=3):
        """The object called `name`: the blob whose three sizes fit objects/NAME/size.yaml best (a
        single blob is taken as it is; the wrist camera confirms what it is anyway). `near`: prefer
        the blob closest to this (x, y). None when nothing fits."""
        objs = self.objects(n)
        if not objs:
            return None
        if near is not None:
            o = min(objs, key=lambda o: np.linalg.norm(o["centre"] - near))
            return o if np.linalg.norm(o["centre"] - near) < 0.05 else None
        sizes = taught_sizes(name) if name else None
        if sizes is None:
            return objs[0]

        best = min((complete(o, sizes) for o in objs), key=lambda o: o["misfit_m"])
        return best if best["misfit_m"] < 0.02 else None

    def draw(self, img=None, objs=None, pick=None):
        img, objs = (self.last if img is None else (img, objs))
        vis = img.copy()
        for o in objs:
            col = (0, 255, 0) if o is pick else (255, 0, 255)
            x1, y1, x2, y2 = o["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.putText(vis, f"{o['centre'][0] * 100:.0f},{o['centre'][1] * 100:+.0f}cm "
                             f"{o['length'] * 100:.1f}x{o['width'] * 100:.1f}x{o['height'] * 100:.1f}",
                        (max(0, x1 - 40), max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        return vis

    def close(self):
        self.oak.close()


def complete(o, sizes):
    """Which way the object lies, and its TRUE centre, from its three known sizes. The OAK-D sees the
    width across its line of sight and the height properly, but along its line of sight only the
    near face (15 Nov: a tape standing edge-on, 7.6 cm deep, measured 3.0 cm). So: try all six ways
    of assigning the three sizes to (across, along, height); the best fit must match `across`, the
    height (seen up to 2 cm short: depth is sparse on a top edge), and be at least as deep as what was
    seen. The centre is then the front face + half the true depth along the line of sight.
    Adds: centre, length, width, yaw (of the long footprint side), misfit_m, lying."""
    import itertools
    best = None
    for across, along, height in itertools.permutations(sizes):
        e_h = o["height"] - height
        e = max(abs(o["across"] - across), max(0.0, e_h), max(0.0, -e_h - 0.02), max(0.0, o["along_seen"] - along - 0.01))
        if best is None or e < best[0] - 1e-6:
            best = (e, across, along, height)
    e, across, along, height = best
    o = dict(o)
    ev = o["view"]
    o["centre"] = o["front"] + ev * along / 2
    o["shift"] = ev * along / 2                 # from the seen front face to the true centre
    o["length"], o["width"] = max(across, along), min(across, along)
    long_dir = ev if along >= across else np.array([-ev[1], ev[0]])
    o["yaw"] = float((np.arctan2(long_dir[1], long_dir[0]) + np.pi / 2) % np.pi - np.pi / 2)
    o["height"], o["misfit_m"], o["completed"] = float(height), float(e), True
    return o


def taught_sizes(name):
    """The object's three sizes (m) from objects/NAME/size.yaml (the ruler ones when given), or None."""
    from iro.objects import OBJ_DIR
    f = OBJ_DIR / name / "size.yaml"
    if not f.exists():
        return None
    d = yaml.safe_load(f.read_text()) or {}
    r = (d.get("ruler") or {}).get("sizes_m")
    if r and len(r) == 3:                       # largest, middle, smallest; the thinnest side counts too
        t = (d.get("ruler") or {}).get("thickness_m")
        return sorted([r[0], r[1], t or r[2]], reverse=True)
    s = [d.get(k) for k in ("length_m", "width_m", "height_m")]
    return sorted(s, reverse=True) if all(isinstance(v, (int, float)) for v in s) else None


def cmd_scout():
    s = Scout()
    try:
        objs = s.objects(5)
        if not objs:
            print("The OAK-D sees nothing standing on the floor in the arm's area (18-50 cm ahead).")
        for o in objs:
            print(f"  at x {o['centre'][0] * 100:.1f}, y {o['centre'][1] * 100:+.1f} cm: "
                  f"{o['length'] * 100:.1f} x {o['width'] * 100:.1f} cm, {o['height'] * 100:.1f} cm tall, "
                  f"long side at {np.degrees(o['yaw']):+.0f} deg" + ("  (front face only)" if o["front_only"] else ""))
        from iro.stream import feed
        feed().show(s.draw(), f"OAK-D scout: {len(objs)} object(s)")
    finally:
        s.close()
    return 0
