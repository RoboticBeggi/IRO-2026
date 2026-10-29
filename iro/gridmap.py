"""Occupancy grid: what ARC has learned about the floor around it (free / obstacle / unknown).

Standard log-odds mapping: every lidar beam makes the cells it passed through a bit more "free" and
the cell where it ended a bit more "occupied". Repeated looks make the map certain; a person who
walked through only leaves a faint trace, which the next looks wipe out.

With a known pose (the simulator, or slam_toolbox's on the real robot) this is all the mapping
needed; slam_toolbox is what corrects the pose itself on the real robot (wheel slip).
"""

import math

import cv2
import numpy as np

L_FREE, L_OCC = -0.4, 0.9                            # log-odds added per look
L_MIN, L_MAX = -4.0, 4.0
FREE_T, OCC_T = -1.0, 1.0                            # below = free, above = obstacle, between = unknown


class GridMap:
    def __init__(self, size_m=24.0, res=0.05):
        self.res = res
        self.n = int(size_m / res)
        self.origin = -size_m / 2                        # world coordinate of cell (0, 0)
        self.L = np.zeros((self.n, self.n), np.float32)  # rows = y, columns = x

    # -- coordinates
    def cell(self, x, y):
        return int((y - self.origin) / self.res), int((x - self.origin) / self.res)

    def world(self, i, j):
        return self.origin + (j + 0.5) * self.res, self.origin + (i + 0.5) * self.res

    # -- learning
    def integrate(self, pose, angles, ranges, max_range=8.0, step=None):
        """Add one scan taken at pose (x, y, heading)."""
        x0, y0, th = pose
        step = step or self.res
        ok = np.isfinite(ranges) & (ranges > 0.05)
        a, r = angles[ok] + th, np.minimum(ranges[ok], max_range)
        hit = ranges[ok] < max_range
        ks = np.arange(0, max_range, step)[None, :]                  # distances along each beam
        free = ks < (r[:, None] - self.res)
        px = x0 + np.cos(a)[:, None] * ks
        py = y0 + np.sin(a)[:, None] * ks
        i = ((py - self.origin) / self.res).astype(np.int32)
        j = ((px - self.origin) / self.res).astype(np.int32)
        inside = free & (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        cells = np.unique(i[inside] * self.n + j[inside])            # each cell once per scan
        np.add.at(self.L.reshape(-1), cells, L_FREE)
        ex, ey = x0 + np.cos(a[hit]) * r[hit], y0 + np.sin(a[hit]) * r[hit]
        ei = ((ey - self.origin) / self.res).astype(np.int32)
        ej = ((ex - self.origin) / self.res).astype(np.int32)
        ins = (ei >= 0) & (ei < self.n) & (ej >= 0) & (ej < self.n)
        cells = np.unique(ei[ins] * self.n + ej[ins])
        np.add.at(self.L.reshape(-1), cells, L_OCC - L_FREE)          # undo this scan's "free" there too
        np.clip(self.L, L_MIN, L_MAX, out=self.L)

    # -- reading
    def free(self):
        return self.L < FREE_T

    def occupied(self):
        return self.L > OCC_T

    def unknown(self):
        return ~self.free() & ~self.occupied()

    def inflated(self, radius):
        """Cells too close to an obstacle for the robot's centre (robot radius + margin)."""
        k = max(1, int(math.ceil(radius / self.res)))
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        return cv2.dilate(self.occupied().astype(np.uint8), disk) > 0

    def known_area_m2(self):
        return float((self.free() | self.occupied()).sum() * self.res ** 2)

    def image(self, scale=2, crop=True):
        """Grey = unknown, white = free, black = obstacle (y up)."""
        img = np.full(self.L.shape, 160, np.uint8)
        img[self.free()] = 250
        img[self.occupied()] = 20
        img = np.flipud(img)
        box = None
        if crop:
            known = np.argwhere(img != 160)
            if len(known):
                (i0, j0), (i1, j1) = known.min(0) - 20, known.max(0) + 20
                i0, j0 = max(0, i0), max(0, j0)
                box = (i0, j0)
                img = img[i0:i1, j0:j1]
        out = cv2.cvtColor(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST),
                           cv2.COLOR_GRAY2BGR)
        return out, box

    def to_pixel(self, x, y, scale=2, box=None):
        i, j = self.cell(x, y)
        i = self.n - 1 - i
        if box:
            i, j = i - box[0], j - box[1]
        return int(j * scale), int(i * scale)

    # -- saving (a map ARC can load again, same format as ROS map_server: PGM + YAML)
    def save(self, path_base):
        img = np.full(self.L.shape, 205, np.uint8)
        img[self.free()] = 254
        img[self.occupied()] = 0
        cv2.imwrite(f"{path_base}.pgm", np.flipud(img))
        np.save(f"{path_base}.npy", self.L)
        with open(f"{path_base}.yaml", "w") as f:
            f.write(f"image: {path_base.split('/')[-1]}.pgm\nresolution: {self.res}\n"
                    f"origin: [{self.origin}, {self.origin}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\n"
                    "free_thresh: 0.25\n")
