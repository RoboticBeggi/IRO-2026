"""Continuous surveillance with the lidar while ARC stands still (PLAN.md, LiDAR layer L6).

    iro visitors                          includes all of this (live picture + events)
    iro visitors zone add NAME [--nogo]   teach a zone by walking around its edge (no markers)
    iro visitors zone list | remove NAME

Three jobs, every scan (10 Hz), on top of iro.visitors (people) and its learned empty room:
  1. visitor statistics: how many now / in total, arrivals and departures, how long each one
     stayed, who engaged with ARC, time spent in each zone, and a heat map of where people stood
  2. zones: named floor areas taught by walking their edge; a "no-go" zone (around an artefact)
     raises an alert as soon as anything enters it (also a bag or a foot, not only tracked people)
  3. changes: something new standing still where the room was empty ("object left behind"), or the
     room now reaching FARTHER than it used to ("something that was here is gone": a moved or
     missing artefact). A change must persist for a few seconds and form a solid patch; if a big
     part of the view changes at once, the robot or the room moved: one "view changed" warning and
     no flood of alarms (the classic map-differencing pitfall: pose error looks like change).
The lidar is at knee height: it sees legs, chairs, stands and boxes on the floor, not a hand
reaching onto a table top.
"""

import json
import math
import sys
import time

import cv2
import numpy as np
import yaml

from iro import ROOT

ZONES = ROOT / "config" / "zones.yaml"
CELL = 0.10                                         # heat map cell (m)
SPAN = 6.0                                          # heat map covers +/- 6 m
PERSIST_S = 4.0                                     # a change must last this long
MIN_BEAMS = 4                                       # ... and cover this many neighbouring beams
MIN_WIDTH_M = 0.10                                  # ... and be at least this wide
VIEW_CHANGED = 0.25                                 # this share of beams changed = the view moved
ABSORB_S = 60.0                                     # a change that stays this long becomes part of the room


# ------------------------------------------------------------------ zones

def load_zones():
    return (yaml.safe_load(ZONES.read_text()) or {}) if ZONES.exists() else {}


def save_zones(z):
    ZONES.write_text(yaml.safe_dump(z, sort_keys=False))


def inside(poly, x, y):
    return cv2.pointPolygonTest(np.asarray(poly, np.float32), (float(x), float(y)), False) >= 0


def hull(points, grow=0.10):
    """Convex hull of the walked points, grown outward by `grow` metres."""
    p = np.asarray(points, np.float32)
    h = cv2.convexHull(p)[:, 0, :]
    c = h.mean(0)
    d = h - c
    n = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
    return (h + d / n * grow).round(3).tolist()


# ------------------------------------------------------------------ statistics

class Stats:
    def __init__(self):
        self.t0 = None
        self.seen = {}                              # id -> {first, last, engaged, zones{name: s}}
        self.heat = np.zeros((int(2 * SPAN / CELL),) * 2, np.float32)
        self.per_min = {}

    def update(self, t, visitors, zones, dt):
        self.t0 = self.t0 or t
        for tr in visitors:
            v = self.seen.setdefault(tr.id, {"first": t, "last": t, "engaged": False, "zones": {}})
            v["last"] = t
            v["engaged"] |= tr.state == "engaged"
            for name, z in zones.items():
                if inside(z["poly"], tr.s[0], tr.s[1]):
                    v["zones"][name] = v["zones"].get(name, 0.0) + dt
            i, j = int((SPAN - tr.s[0]) / CELL), int((SPAN - tr.s[1]) / CELL)
            if 0 <= i < self.heat.shape[0] and 0 <= j < self.heat.shape[1]:
                self.heat[i, j] += dt
        m = time.strftime("%H:%M", time.localtime(t))
        self.per_min[m] = max(self.per_min.get(m, 0), len(visitors))

    def summary(self, now_count):
        stays = [v["last"] - v["first"] for v in self.seen.values() if v["last"] - v["first"] > 1]
        return {"now": now_count, "total": len(self.seen), "engaged": sum(v["engaged"] for v in self.seen.values()),
                "mean_stay_s": round(float(np.mean(stays)), 1) if stays else 0.0,
                "busiest_minute": max(self.per_min.items(), key=lambda kv: kv[1], default=("-", 0))}

    def heat_image(self, size=480):
        h = self.heat / max(float(self.heat.max()), 1e-6)
        img = cv2.applyColorMap((np.sqrt(h) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
        c = size // 2
        cv2.circle(img, (c, c), 5, (255, 255, 255), -1)      # ARC
        cv2.putText(img, "where visitors stood (ARC = white dot, forward = up)", (6, 16), 0, 0.42,
                    (255, 255, 255), 1)
        return img

    def save(self, d, now_count):
        (d / "stats.json").write_text(json.dumps({"summary": self.summary(now_count), "visitors": self.seen,
                                                  "per_minute": self.per_min}, indent=1, default=float))
        cv2.imwrite(str(d / "heatmap.png"), self.heat_image())


# ------------------------------------------------------------------ changes

class Changes:
    """Per-beam persistence of 'closer than the empty room' (not a person) and 'farther than it'."""

    CLEAR_S = 2.0                                   # a change must be gone this long to be "back to normal"
    SAME_M = 0.6                                    # patches this close are one object

    def __init__(self, n):
        self.close = np.zeros(n, np.float32)        # seconds each beam has stayed changed
        self.far = np.zeros(n, np.float32)
        self.active = []                            # [kind, x, y, last_seen]
        self.view_warned = False
        self.t = 0.0
        self.absorb = np.zeros(n, bool)

    @staticmethod
    def _bridge(mask, gap=6):
        """Close holes of up to `gap` beams (noise, a dropout) so one object is one patch."""
        k = np.ones(gap + 1)
        grown = np.convolve(mask.astype(float), k, "same") > 0
        return np.convolve((~grown).astype(float), k, "same") == 0

    def step(self, a, r, med, mad, self_mask, people_xy, dt, near=None, far=None):
        """near / far: the room's edge-tolerant envelope (iro.visitors.envelope); default med."""
        near = med if near is None else near
        far = med if far is None else far
        n = len(r)
        if len(self.close) != n:
            self.__init__(n)
        self.t += dt
        valid = np.isfinite(r) & (r > 0.15) & ~self_mask
        with np.errstate(invalid="ignore"):
            margin = np.maximum(0.15, np.maximum(3 * mad, 0.04 * np.where(np.isfinite(med), med, r)))
            closer = valid & (~np.isfinite(near) | (r < near - margin))
            farther = valid & np.isfinite(far) & (r > far + margin)
        # a beam that hits a person (or near one) is not a change of the room
        if people_xy:
            x, y = r * np.cos(a), r * np.sin(a)
            for px, py in people_xy:
                closer &= np.hypot(x - px, y - py) > 0.5
        # a dropout (no echo for one scan) keeps the beam's timer; only a clear "normal" resets it
        self.close = np.where(closer, self.close + dt, np.where(valid, 0, self.close))
        self.far = np.where(farther, self.far + dt, np.where(valid, 0, self.far))
        events = []
        share = float(np.mean((self.close > 1) | (self.far > 1)))
        if share > VIEW_CHANGED:
            if not self.view_warned:
                events.append(f"the view changed ({share:.0%} of the room): was ARC moved? Re-learn the room "
                              "(iro visitors learn) before trusting change alerts")
                self.view_warned = True
            return events
        self.view_warned = False
        for kind, cnt, rng in (("appeared", self.close, r), ("gone", self.far, med)):
            idx = np.flatnonzero(self._bridge(cnt >= PERSIST_S))
            if not len(idx):
                continue
            for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                good = seg[np.isfinite(rng[seg]) & (rng[seg] > 0)]
                if len(good) < MIN_BEAMS:
                    continue
                d, ang = float(np.median(rng[good])), float(np.median(a[good]))
                # real size: an object left / taken is >= 10 cm wide; a robot nudged a few cm (or a beam
                # grazing an edge) only makes slivers (26 Oct: ~25 sliver alerts at once)
                if d * abs(float(a[good[-1]] - a[good[0]])) < MIN_WIDTH_M:
                    continue
                x, y = d * math.cos(ang), d * math.sin(ang)
                hit = next((c for c in self.active if c[0] == kind and math.hypot(c[1] - x, c[2] - y) < self.SAME_M), None)
                if hit:
                    hit[3] = self.t
                    continue
                self.active.append([kind, x, y, self.t])
                where = f"{d:.1f} m away at {math.degrees(ang):+.0f} deg"
                events.append(f"something new is standing {where} (object left behind?)" if kind == "appeared"
                              else f"something that stood {where} is gone (moved or missing?)")
        # background upkeep: what has stayed changed for a minute is the room now (reported once, above);
        # without this a moved chair would stay "foreground" forever
        self.absorb = (self.close >= ABSORB_S) | (self.far >= ABSORB_S)
        if self.absorb.any():
            self.close[self.absorb] = 0
            self.far[self.absorb] = 0
            self.active = [c for c in self.active if c[0] != "gone" or self.t - c[3] < 1]
        for c in [c for c in self.active if self.t - c[3] > self.CLEAR_S]:
            self.active.remove(c)
            events.append(f"the {'new object' if c[0] == 'appeared' else 'missing thing'} at "
                          f"{math.hypot(c[1], c[2]):.1f} m is back to normal")
        return events


# ------------------------------------------------------------------ all together

class Surveil:
    def __init__(self):
        self.zones = load_zones()
        self.stats = Stats()
        self.changes = None
        self.in_nogo = {}                           # zone -> since when something is inside
        self.t = None

    def step(self, t, a, r, vis):
        """Events (strings) for this scan. `vis` is the iro.visitors.Visitors that just stepped."""
        dt = 0.1 if self.t is None else min(0.5, max(0.0, t - self.t))
        self.t = t
        ev = []
        people = vis.visitors()
        self.stats.update(t, people, self.zones, dt)
        if self.changes is None:
            self.changes = Changes(len(r))
        # only real people mask changes: a box put down is an unconfirmed, motionless track
        movers = [(tr.s[0], tr.s[1]) for tr in vis.tracks if tr.id or tr.moved > 0.2]
        ev += self.changes.step(a, r, vis.med, vis.mad, vis.self_mask, movers, dt,
                                near=getattr(vis, "near", None), far=getattr(vis, "far", None))
        ab = self.changes.absorb & np.isfinite(r) & (r > 0.15)
        if ab.any():
            vis.med = vis.med.copy()
            vis.med[ab] = r[ab]                    # the room as it is now
            if hasattr(vis, "refresh"):
                vis.refresh()
        # no-go zones: any tracked person, or any foreground blob, inside
        things = [(tr.s[0], tr.s[1], f"visitor {tr.id}") for tr in people] + \
                 [(o[0], o[1], "something") for o in getattr(vis, "others", [])]
        for name, z in self.zones.items():
            if z.get("kind") != "nogo":
                continue
            who = next((w for x, y, w in things if inside(z["poly"], x, y)), None)
            if who and name not in self.in_nogo:
                self.in_nogo[name] = t
            elif who and self.in_nogo.get(name) is not None and t - self.in_nogo[name] >= 0.3:
                ev.append(f"{who} entered the no-go zone '{name}'")
                self.in_nogo[name] = None           # reported; reset when it is empty again
            elif not who:
                self.in_nogo.pop(name, None)
        return ev

    def draw(self, img, span=4.0):
        """Zones and changes on iro.visitors.radar's picture (same scale)."""
        size = img.shape[0]
        c, sc = size // 2, size / 2 / span

        def px(x, y):
            return int(c - y * sc), int(c - x * sc)
        for name, z in self.zones.items():
            col = (0, 0, 220) if z.get("kind") == "nogo" else (160, 120, 0)
            pts = np.array([px(x, y) for x, y in z["poly"]], np.int32)
            cv2.polylines(img, [pts], True, col, 2)
            cv2.putText(img, name, tuple(pts.min(0) + [2, 12]), 0, 0.4, col, 1)
        for kind, x, y, _ in (self.changes.active if self.changes else []):
            p = px(x, y)
            col = (0, 0, 255) if kind == "gone" else (255, 0, 255)
            cv2.drawMarker(img, p, col, cv2.MARKER_TILTED_CROSS if kind == "gone" else cv2.MARKER_SQUARE, 18, 2)
        s = self.stats.summary(0)
        cv2.putText(img, f"total {s['total']}  engaged {s['engaged']}  mean stay {s['mean_stay_s']} s",
                    (8, 36), 0, 0.45, (60, 60, 60), 1)
        return img


# ------------------------------------------------------------------ teaching zones

def cmd_zone(args):
    z = load_zones()
    if not args or args[0] == "list":
        if not z:
            print("No zones yet. Teach one: iro visitors zone add NAME [--nogo]")
        for name, v in z.items():
            print(f"  {name:15} {v.get('kind', 'area'):5} {len(v['poly'])} corners, taught {v.get('date', '?')}")
        return 0
    if args[0] == "remove" and len(args) > 1:
        print("removed" if z.pop(args[1], None) else "no such zone")
        save_zones(z)
        return 0
    if args[0] != "add" or len(args) < 2:
        print(__doc__)
        return 1
    name, nogo = args[1], "--nogo" in args
    from iro.record import run
    from iro.visitors import BG, Lidar, Visitors
    if not BG.exists():
        raise SystemExit("Learn the empty room first: iro visitors learn")
    rec = run(f"visitors-zone-{name}")
    secs = 20
    input(f"Zone '{name}': when you press Enter, walk slowly along the zone's edge for {secs} s "
          "(nobody else in view). Enter: ")
    vis = Visitors()
    lidar = Lidar()
    paths, n, t0 = {}, None, time.time()
    try:
        while time.time() - t0 < secs:
            n, (t, a, r) = lidar.read(n)
            vis.step(t, a, r)
            for tr in vis.visitors():               # only the tracked walker, not stray blobs
                paths.setdefault(tr.id, []).append((float(tr.s[0]), float(tr.s[1])))
            print(f"\r  {secs - (time.time() - t0):4.0f} s left, {sum(map(len, paths.values()))} points",
                  end="", flush=True)
    finally:
        lidar.close()
    print()
    pts = max(paths.values(), key=len, default=[])  # the person who walked the longest
    if len(pts) < 20:
        print("Too few points: walk inside the lidar's view (within 6 m), slowly, and try again.")
        return 1
    area = cv2.contourArea(np.asarray(hull(pts, 0.0), np.float32))
    if area > 9:
        print(f"That zone would be {area:.0f} m2: too big for one artefact. Was someone else moving? Try again.")
        return 1
    rec.note(f"zone {name}: {len(pts)} points from visitor path, {area:.1f} m2")
    z[name] = {"kind": "nogo" if nogo else "area", "poly": hull(pts), "date": time.strftime("%Y-%m-%d %H:%M")}
    save_zones(z)
    print(f"Saved zone '{name}' ({'no-go' if nogo else 'area'}, {len(z[name]['poly'])} corners) in "
          f"{ZONES.relative_to(ROOT)}. It shows on the live picture in iro visitors.")
    return 0


if __name__ == "__main__":
    sys.exit(cmd_zone(sys.argv[1:]))
