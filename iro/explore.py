"""ARC explores and learns its surroundings: it drives around on its own until it has mapped the place.

    iro explore --sim [--seed N] [--minutes M]   in the simulator (no hardware: safe to run any time)
    iro explore                                   on the real robot (needs the base + hardware E-stop)

Frontier-based exploration (Yamauchi 1997, the classic "map the room by yourself" method):
  1. map: every scan goes into an occupancy grid (iro.gridmap): free / obstacle / unknown
  2. frontiers: free cells next to unknown ones = the edges of what ARC knows; clustered, tiny ones
     ignored
  3. choose: the reachable frontier with the best "how much is unknown there / how far to drive"
  4. plan: shortest path on the map, kept a robot radius + margin from obstacles (wavefront search)
  5. drive: follow the path looking ahead (pure pursuit), facing the way it drives so the camera
     and the front of the lidar look where it goes
  6. safety, every cycle, from the RAW scan (not the map): slow down when anything is within 0.6 m
     in the direction of travel, stop within 0.3 m; people count like everything else
  7. stuck (no progress for 6 s)? look around, forget that goal, take another
  8. no frontiers left: drive back to the start and save the map (PGM + YAML, like ROS map_server)
Every run is recorded: a video of the map growing, the final map, and the numbers.
"""

import math
import sys
import time

import cv2
import numpy as np

from iro import ROOT
from iro.gridmap import GridMap

MAPS = ROOT / "config" / "maps"
MAX_V, MAX_W = 0.25, 0.8                  # m/s, rad/s: walking pace in a booth
LOOKAHEAD = 0.45                          # m along the path
MARGIN = 0.08                             # m kept beyond the robot's radius
SLOW_AT, STOP_AT = 0.60, 0.20             # m of clearance ahead (beyond the footprint); at 0.25 m/s and
#                                           0.6 m/s^2 braking ARC stops within ~8 cm (incl. one 0.1 s cycle)
COMFORT = 0.20                            # m: plan this far from obstacles when there is room
STUCK_S, STUCK_M = 6.0, 0.10
MIN_FRONTIER = 6                          # cells (5 cm) = 30 cm of frontier


# ------------------------------------------------------------------ planning on the grid

def wavefront(passable, start):
    """Steps (8-connected) from `start` to every passable cell (inf = unreachable)."""
    dist = np.full(passable.shape, np.inf, np.float32)
    i, j = start
    if not (0 <= i < passable.shape[0] and 0 <= j < passable.shape[1]):
        return dist
    reached = np.zeros(passable.shape, np.uint8)
    reached[i, j] = 1
    dist[i, j] = 0
    kernel = np.ones((3, 3), np.uint8)
    pas = passable.astype(np.uint8)
    for k in range(1, 4000):
        grown = cv2.dilate(reached, kernel) & pas
        new = (grown > 0) & (reached == 0)
        if not new.any():
            break
        dist[new] = k
        reached |= grown
    return dist


def path_from(dist, goal):
    """Walk down the wavefront from the goal to the start: the cells of the shortest path."""
    path = [goal]
    i, j = goal
    while dist[i, j] > 0:
        best = None
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if (di or dj) and 0 <= ni < dist.shape[0] and 0 <= nj < dist.shape[1] and dist[ni, nj] < dist[i, j]:
                    if best is None or dist[ni, nj] < dist[best]:
                        best = (ni, nj)
        if best is None:
            break
        i, j = best
        path.append(best)
    return path[::-1]


def frontiers(m, passable, dist):
    """Clusters of reachable frontier cells: [(size, (i, j) nearest cell, steps away)]."""
    unknown = m.unknown().astype(np.uint8)
    near_unknown = cv2.dilate(unknown, np.ones((3, 3), np.uint8)) > 0
    fr = (m.free() & near_unknown & passable).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fr, connectivity=8)
    out = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < MIN_FRONTIER:
            continue
        cells = np.argwhere(lab == k)
        d = dist[cells[:, 0], cells[:, 1]]
        if not np.isfinite(d).any():
            continue
        best = cells[int(np.argmin(d))]
        out.append((int(stats[k, cv2.CC_STAT_AREA]), (int(best[0]), int(best[1])), float(d.min())))
    return out


# ------------------------------------------------------------------ safety from the raw scan

def safe_velocity(vx, vy, wz, angles, ranges, radius):
    """Scale (vx, vy) down by the clearance in the direction of travel (Nav2 collision-monitor style:
    slow zone, stop zone); turning on the spot is always allowed. Returns (vx, vy, wz, clearance)."""
    speed = math.hypot(vx, vy)
    if speed < 1e-3:
        return vx, vy, wz, float("inf")
    heading = math.atan2(vy, vx)
    ok = np.isfinite(ranges) & (ranges > 0.05)
    d = (angles[ok] - heading + math.pi) % (2 * math.pi) - math.pi
    r = ranges[ok]
    # everything that would be swept by the footprint moving that way, within 1.5 m
    lateral = np.abs(r * np.sin(d))
    ahead = r * np.cos(d)
    in_path = (ahead > 0) & (lateral < radius + 0.05) & (ahead < 1.5)
    clear = float((ahead[in_path] - radius).min()) if in_path.any() else float("inf")
    if clear < STOP_AT:
        return 0.0, 0.0, wz, clear
    if clear < SLOW_AT:
        k = 0.3 + 0.7 * (clear - STOP_AT) / (SLOW_AT - STOP_AT)
        return vx * k, vy * k, wz, clear
    return vx, vy, wz, clear


# ------------------------------------------------------------------ the explorer

class Explorer:
    def __init__(self, robot, radius=0.25, log=print):
        self.robot, self.radius, self.log = robot, radius, log
        self.map = GridMap()
        self.start = robot.pose()
        self.goal, self.path, self.blacklist = None, [], []
        self.last_progress = (0.0, robot.pose())
        self.state = "look around"
        self.turned = 0.0
        self.driven = 0.0
        self.near_misses = 0
        self.prev = robot.pose()
        self.t = 0.0
        self.replan_t = -1.0
        self.frontier_list = []
        self.resume = None

    def passable(self, extra=0.0):
        p = self.map.free() & ~self.map.inflated(self.radius + MARGIN + extra)
        # the robot's own spot is passable even if it got close to something
        i, j = self.map.cell(*self.robot.pose()[:2])
        cv2.circle(p.view(np.uint8), (j, i), int((self.radius + MARGIN) / self.map.res), 1, -1)
        return p

    def plan_space(self, goal=None):
        """(passable, wavefront): with comfortable clearance if that still reaches the goal (or any
        frontier), else the tight one (26 Oct: planning 8 cm past a pedestal while the safety layer
        wanted 30 cm froze ARC next to it)."""
        here = self.map.cell(*self.robot.pose()[:2])
        for extra in (COMFORT, 0.0):
            pas = self.passable(extra)
            dist = wavefront(pas, here)
            if goal is not None and np.isfinite(dist[goal]):
                return pas, dist
            if goal is None and frontiers(self.map, pas, dist):
                return pas, dist
        return pas, dist

    def choose_goal(self):
        pas, dist = self.plan_space()
        fl = [f for f in frontiers(self.map, pas, dist)
              if all(math.hypot(f[1][0] - b[0], f[1][1] - b[1]) * self.map.res > 0.5 for b in self.blacklist)]
        self.frontier_list = fl
        if not fl:
            return None, dist
        # utility: bigger unknown edge is worth more, driving far costs (steps are 5 cm)
        best = max(fl, key=lambda f: f[0] * self.map.res * 2.0 - f[2] * self.map.res)
        return best[1], dist

    def step(self, dt):
        """One control cycle: map, decide, drive. Returns False when finished."""
        self.t += dt
        x, y, th = self.robot.pose()
        a, r = self.robot.scan()
        self.map.integrate((x, y, th), a, r)
        self.driven += math.hypot(x - self.prev[0], y - self.prev[1])
        self.prev = (x, y, th)
        if self.state == "look around":                 # a full turn on the spot: see all around
            self.robot.drive(0, 0, MAX_W)
            self.turned += MAX_W * dt
            if self.turned >= 2 * math.pi:
                self.state, self.turned = getattr(self, "resume", "explore") or "explore", 0.0
                self.resume = None
                self.robot.stop()
            return True
        if self.state == "back off":                    # 20 cm straight away from the nearest obstacle
            ok = np.isfinite(r) & (r > 0.05)
            k = int(np.argmin(np.where(ok, r, np.inf)))
            away = a[k] + math.pi
            vx, vy, _, clear = safe_velocity(0.12 * math.cos(away), 0.12 * math.sin(away), 0, a, r, self.radius)
            self.robot.drive(vx, vy, 0)
            self.turned += 0.12 * dt
            if self.turned >= 0.2 or clear < STOP_AT:
                self.robot.stop()
                self.state, self.turned = "look around", 0.0
            return True
        if self.state in ("explore", "home") and (self.goal is None or self.t - self.replan_t > 2.0):
            self.replan_t = self.t
            if self.state == "explore":
                goal, dist = self.choose_goal()
                if goal is None:
                    self.log(f"{self.t:6.1f}s  no more frontiers: everything reachable is mapped; going home")
                    self.state = "home"
                    goal = self.map.cell(*self.start[:2])
                    _, dist = self.plan_space(goal)
            else:
                goal = self.map.cell(*self.start[:2])
                _, dist = self.plan_space(goal)
            if goal != self.goal:
                self.last_progress = (self.t, (x, y))
            self.goal = goal
            self.path = path_from(dist, goal) if np.isfinite(dist[goal]) else []
            if not self.path and self.state == "home":
                self.log(f"{self.t:6.1f}s  cannot plan a way home from here; stopping where I am")
                self.robot.stop()
                return False
        # reached?
        gx, gy = self.map.world(*self.goal)
        if math.hypot(gx - x, gy - y) < (0.25 if self.state == "home" else 0.35):
            if self.state == "home":
                self.robot.stop()
                self.log(f"{self.t:6.1f}s  back at the start")
                return False
            self.goal = None
            return True
        # stuck?
        t0, p0 = self.last_progress
        if math.hypot(x - p0[0], y - p0[1]) > STUCK_M:
            self.last_progress = (self.t, (x, y))
        elif self.t - t0 > STUCK_S:
            self.log(f"{self.t:6.1f}s  stuck on the way to ({gx:.1f}, {gy:.1f}): backing off, then trying again")
            if self.state != "home":
                self.blacklist.append(self.goal)
            self.resume = self.state
            self.goal, self.state, self.turned = None, "back off", 0.0
            self.last_progress = (self.t, (x, y))
            return True
        # follow the path: the point LOOKAHEAD metres ahead along it
        pts = [self.map.world(i, j) for i, j in self.path]
        k = min(range(len(pts)), key=lambda n: math.hypot(pts[n][0] - x, pts[n][1] - y)) if pts else 0
        target = pts[-1] if pts else (gx, gy)
        for q in pts[k:]:
            if math.hypot(q[0] - x, q[1] - y) >= LOOKAHEAD:
                target = q
                break
        dx, dy = target[0] - x, target[1] - y
        dist_t = math.hypot(dx, dy)
        want = math.atan2(dy, dx)
        err = (want - th + math.pi) % (2 * math.pi) - math.pi
        wz = max(-MAX_W, min(MAX_W, 1.5 * err))
        speed = min(MAX_V, 0.6 * dist_t + 0.05) * max(0.0, math.cos(err)) ** 2   # turn first, then go
        bx, by = math.cos(-th) * dx - math.sin(-th) * dy, math.sin(-th) * dx + math.cos(-th) * dy
        n = math.hypot(bx, by) or 1.0
        vx, vy, wz, clear = safe_velocity(bx / n * speed, by / n * speed, wz, a, r, self.radius)
        if clear < STOP_AT and speed > 0.02:
            self.near_misses += 1
        self.robot.drive(vx, vy, wz)
        return True

    def draw(self, scale=2):
        img, box = self.map.image(scale)
        for _, (i, j), _ in self.frontier_list:
            cv2.circle(img, self.map.to_pixel(*self.map.world(i, j), scale, box), 3, (255, 160, 0), -1)
        if self.path:
            pts = np.array([self.map.to_pixel(*self.map.world(i, j), scale, box) for i, j in self.path], np.int32)
            cv2.polylines(img, [pts], False, (0, 170, 0), 1)
        if self.goal:
            cv2.drawMarker(img, self.map.to_pixel(*self.map.world(*self.goal), scale, box), (0, 0, 255),
                           cv2.MARKER_CROSS, 12, 2)
        x, y, th = self.robot.pose()
        p = self.map.to_pixel(x, y, scale, box)
        cv2.circle(img, p, int(self.radius / self.map.res * scale), (200, 60, 0), 2)
        cv2.line(img, p, (int(p[0] + 14 * math.cos(th)), int(p[1] - 14 * math.sin(th))), (200, 60, 0), 2)
        return img


# ------------------------------------------------------------------ commands

def _panel(left, right, text):
    h = max(left.shape[0], right.shape[0])
    pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    img = np.hstack([pad(left), pad(right)])
    img = cv2.copyMakeBorder(img, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(img, text, (6, 18), 0, 0.5, (0, 0, 0), 1)
    return img


def cmd_sim(seed=0, minutes=10.0, show=True):
    from iro.navsim import SimRobot, World
    from iro.record import run
    rec = run("explore-sim")
    world = World(seed=seed)
    bot = SimRobot(world, seed=seed + 1)
    ex = Explorer(bot, bot.RADIUS, log=lambda s: (print(s), rec.note(s)))
    dt, t_wall = 0.1, time.time()
    feed = None
    if show:
        try:
            from iro.stream import feed as _feed
            feed = _feed()
        except Exception:                             # noqa: BLE001 - the live page is a nicety
            feed = None
    truth_free = float((world.g == 0).sum() * 0.02 ** 2)
    n = 0
    while ex.t < minutes * 60:
        alive = ex.step(dt)
        bot.step(dt)
        if n % 5 == 0:                                  # 2 frames per simulated second
            truth = bot.truth_image()
            cv2.circle(truth, (int((bot.p[0] - world.origin[0]) * 40), int(truth.shape[0] - (bot.p[1] - world.origin[1]) * 40)),
                       int(bot.RADIUS * 40), (200, 60, 0), 2)
            for q in world.people:
                cv2.circle(truth, (int((q["p"][0] - world.origin[0]) * 40),
                                   int(truth.shape[0] - (q["p"][1] - world.origin[1]) * 40)), 6, (0, 0, 220), -1)
            frame = _panel(ex.draw(), truth, f"t {ex.t:5.0f}s  {ex.state:12}  mapped {ex.map.known_area_m2():5.1f} m2  "
                                              f"driven {ex.driven:4.1f} m  collisions {bot.collisions}")
            rec.video("explore", frame)
            if feed:
                feed.show(frame, "")
        n += 1
        if not alive:
            break
    MAPS.mkdir(parents=True, exist_ok=True)
    ex.map.save(str(rec.dir / "map"))
    cv2.imwrite(str(rec.dir / "map.png"), ex.draw(3))
    summary = (f"explored for {ex.t:.0f} s (simulated; {time.time() - t_wall:.0f} s real): mapped "
               f"{ex.map.known_area_m2():.1f} m2 of {truth_free:.1f} m2 floor, drove {ex.driven:.1f} m, "
               f"collisions {bot.collisions}, safety stops {ex.near_misses}")
    print(summary)
    rec.note(summary)
    return 0 if bot.collisions == 0 else 1


def main():
    a = sys.argv[1:]
    if "--sim" in a:
        seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 0
        minutes = float(a[a.index("--minutes") + 1]) if "--minutes" in a else 10.0
        return cmd_sim(seed, minutes)
    print("The real robot cannot explore yet: it needs the drive base (base_node) and a wired hardware E-stop.\n"
          "Try it in the simulator: iro explore --sim")
    return 1


if __name__ == "__main__":
    sys.exit(main())
