"""A simple 2D simulator of ARC on the floor: test driving, mapping and exploring with no hardware.

The world is a 2 cm raster of walls and furniture (an exhibition booth by default) plus visitors
walking between random spots. ARC is a mecanum base (drives in any direction and turns) with
acceleration limits and a little wheel slip; its lidar is like the RPLidar S2 (360 deg, 0.15-8 m
used, ~1 cm noise, a few dropouts). Collisions are counted, never allowed: the robot just stops.
Same interface as the real robot (iro.explore.Robot): scan(), pose(), drive(), stop().
"""

import math

import cv2
import numpy as np

RES = 0.02


class World:
    def __init__(self, w=7.0, h=6.0, layout="booth", seed=0):
        self.w, self.h = w, h
        self.origin = (-1.0, -h / 2)                     # world (x, y) of raster cell (0, 0): start at x=0
        self.g = np.zeros((int(h / RES), int(w / RES)), np.uint8)   # 1 = solid
        self.rng = np.random.default_rng(seed)
        self.people = []
        getattr(self, f"_{layout}")()

    def _rect(self, x0, y0, x1, y1):
        i0, j0 = self._ij(x0, y0)
        i1, j1 = self._ij(x1, y1)
        self.g[max(0, min(i0, i1)):max(i0, i1) + 1, max(0, min(j0, j1)):max(j0, j1) + 1] = 1

    def _disc(self, x, y, r):
        i, j = self._ij(x, y)
        cv2.circle(self.g, (j, i), int(r / RES), 1, -1)

    def _ij(self, x, y):
        return int((y - self.origin[1]) / RES), int((x - self.origin[0]) / RES)

    def _booth(self):
        x0, y0 = self.origin
        x1, y1 = x0 + self.w, y0 + self.h
        t = 0.1
        self._rect(x0, y0, x1, y0 + t)                   # walls
        self._rect(x0, y1 - t, x1, y1)
        self._rect(x0, y0, x0 + t, y1)
        self._rect(x1 - t, y0, x1, y0 + 1.6)             # far wall with a 1 m doorway
        self._rect(x1 - t, y0 + 2.6, x1, y1)
        self._rect(x1 + 0.5, y0 - 0.5, x1 + 0.6, y1 + 0.5)   # corridor wall behind the door
        self._rect(1.5, 1.6, 3.0, 2.2)                   # display table along the top wall
        self._rect(4.0, -2.4, 5.2, -1.6)                 # second table, bottom right
        self._disc(2.6, -0.6, 0.25)                      # artefact pedestal in the room
        self._disc(1.0, -2.0, 0.15)                      # pillar
        for cx, cy in ((4.4, 1.0), (4.9, 1.2)):          # two chairs (4 legs each, 3 cm)
            for dx in (-0.2, 0.2):
                for dy in (-0.2, 0.2):
                    self._disc(cx + dx, cy + dy, 0.02)
        self._rect(-0.3, -2.9, 0.4, -2.3)                # a cabinet by the bottom wall
        for _ in range(2):                               # visitors
            self.people.append({"p": np.array(self._free_spot()), "goal": np.array(self._free_spot()),
                                "v": 0.6 + 0.3 * self.rng.random()})

    def solid(self, x, y):
        i, j = self._ij(x, y)
        if not (0 <= i < self.g.shape[0] and 0 <= j < self.g.shape[1]):
            return True
        return bool(self.g[i, j])

    def _free_spot(self, margin=0.4):
        while True:
            x = self.origin[0] + margin + self.rng.random() * (self.w - 2 * margin)
            y = self.origin[1] + margin + self.rng.random() * (self.h - 2 * margin)
            if not self.blocked(x, y, margin):
                return x, y

    def blocked(self, x, y, radius):
        i, j = self._ij(x, y)
        k = int(radius / RES)
        sub = self.g[max(0, i - k):i + k + 1, max(0, j - k):j + k + 1]
        if sub.size == 0:
            return True
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1][:, :sub.shape[0], :sub.shape[1]]
        return bool((sub & ((yy ** 2 + xx ** 2) <= k * k)).any())

    def step_people(self, dt, robot_xy=None):
        """Visitors walk between random spots; like real people they give way to ARC (stop and turn
        aside within ~0.55 m) instead of walking through it."""
        for p in self.people:
            d = p["goal"] - p["p"]
            n = float(np.hypot(*d))
            if n < 0.2:
                p["goal"] = np.array(self._free_spot())
                continue
            nxt = p["p"] + d / n * p["v"] * dt
            if robot_xy is not None and math.hypot(nxt[0] - robot_xy[0], nxt[1] - robot_xy[1]) < 0.55:
                p["goal"] = np.array(self._free_spot())     # give way
                continue
            if not self.blocked(nxt[0], nxt[1], 0.2):
                p["p"] = nxt
            else:
                p["goal"] = np.array(self._free_spot())


class SimRobot:
    """ARC in the simulator."""

    RADIUS = 0.25                                         # footprint (circle), metres
    N_BEAMS = 1080                                        # 1/3 of DenseBoost: plenty for mapping

    def __init__(self, world=None, pose=(0.0, 0.0, 0.0), seed=1):
        self.world = world or World()
        self.p = np.array(pose, float)
        self.v = np.zeros(3)                              # current body velocity (vx, vy, wz)
        self.cmd = np.zeros(3)
        self.t = 0.0
        self.rng = np.random.default_rng(seed)
        self.collisions = 0
        self.a = np.linspace(-math.pi, math.pi, self.N_BEAMS, endpoint=False)

    # -- the interface
    def drive(self, vx, vy, wz):
        self.cmd = np.array([vx, vy, wz], float)

    def stop(self):
        self.cmd = np.zeros(3)

    def pose(self):
        return tuple(self.p)

    def scan(self):
        """(angles, ranges) in the robot frame, like the RPLidar (0 = forward, + = left)."""
        x, y, th = self.p
        steps = np.arange(0.15, 8.0, RES)
        wa = self.a + th
        px = x + np.cos(wa)[:, None] * steps
        py = y + np.sin(wa)[:, None] * steps
        i = ((py - self.world.origin[1]) / RES).astype(np.int32)
        j = ((px - self.world.origin[0]) / RES).astype(np.int32)
        ok = (i >= 0) & (i < self.world.g.shape[0]) & (j >= 0) & (j < self.world.g.shape[1])
        hit = np.zeros_like(ok)
        hit[ok] = self.world.g[i[ok], j[ok]] > 0
        hit |= ~ok
        first = np.where(hit.any(1), hit.argmax(1), len(steps) - 1)
        r = np.where(hit.any(1), steps[first], np.inf)
        for p in self.world.people:                      # people: two legs ~ a 30 cm circle at knee height
            cx, cy = p["p"][0] - x, p["p"][1] - y
            b = np.cos(wa) * cx + np.sin(wa) * cy
            c = cx * cx + cy * cy - 0.15 ** 2
            disc = b * b - c
            t = b - np.sqrt(np.maximum(disc, 0))
            r = np.where((disc >= 0) & (t > 0.15) & (t < r), t, r)
        r = r + self.rng.normal(0, 0.01, len(r))
        r[self.rng.random(len(r)) < 0.01] = 0             # dropouts
        return self.a.copy(), r.astype(np.float32)

    # -- physics
    def step(self, dt):
        lim = np.array([0.6, 0.6, 1.5]) * dt              # acceleration limits per step
        self.v += np.clip(self.cmd - self.v, -lim, lim)
        slip = 1 + self.rng.normal(0, 0.03, 3)            # mecanum rollers slip a little
        vx, vy, wz = self.v * slip
        th = self.p[2]
        nxt = self.p + np.array([vx * math.cos(th) - vy * math.sin(th),
                                 vx * math.sin(th) + vy * math.cos(th), wz]) * dt
        moving = math.hypot(vx, vy) > 0.01
        if moving and (self.world.blocked(nxt[0], nxt[1], self.RADIUS) or self._hits_person(nxt)):
            self.collisions += 1                      # ARC drove into something (not: someone walked into ARC)
            self.v[:] = 0
        elif not moving:
            self.p[2] = nxt[2]                        # turning on the spot is always possible
        else:
            self.p = nxt
        self.world.step_people(dt, self.p[:2])
        self.t += dt

    def _hits_person(self, p):
        return any(math.hypot(q["p"][0] - p[0], q["p"][1] - p[1]) < self.RADIUS + 0.15 for q in self.world.people)

    def truth_image(self, scale=40):
        img = cv2.cvtColor(np.flipud((1 - self.world.g) * 230 + 20).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return cv2.resize(img, None, fx=scale * RES, fy=scale * RES, interpolation=cv2.INTER_AREA)
