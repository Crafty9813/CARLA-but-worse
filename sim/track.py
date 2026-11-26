"""Catmull-Rom spline race track and scenery."""

import math
import random

ROAD_HALF_WIDTH = 7.0
SAMPLE_SPACING = 2.0

# (x, z) in metres, in driving order
CONTROL_POINTS = [
    (0, 0), (0, 120), (20, 190), (80, 220), (150, 200), (180, 140),
    (240, 110), (310, 130), (350, 190), (420, 200), (460, 140), (440, 60),
    (380, 20), (360, -50), (400, -120), (370, -190), (290, -210), (220, -170),
    (170, -100), (110, -130), (50, -170), (10, -120),
]


def _catmull_rom(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    return tuple(
        0.5 * (2 * p1[i] + (-p0[i] + p2[i]) * t
               + (2 * p0[i] - 5 * p1[i] + 4 * p2[i] - p3[i]) * t2
               + (-p0[i] + 3 * p1[i] - 3 * p2[i] + p3[i]) * t3)
        for i in range(2))


class Track:
    def __init__(self, seed=7):
        self.points = self._sample()
        n = len(self.points)
        self.tangents = []
        self.normals = []  # driver's right
        for i in range(n):
            ax, az = self.points[i - 1]
            bx, bz = self.points[(i + 1) % n]
            dx, dz = bx - ax, bz - az
            length = math.hypot(dx, dz)
            tx, tz = dx / length, dz / length
            self.tangents.append((tx, tz))
            self.normals.append((-tz, tx))
        self.length = sum(math.dist(self.points[i], self.points[(i + 1) % n]) for i in range(n))

        rng = random.Random(seed)
        self.trees = self._scatter(rng, 420, min_clear=ROAD_HALF_WIDTH + 5, spread=140)
        self.buildings = self._scatter(rng, 40, min_clear=ROAD_HALF_WIDTH + 30, spread=260,
                                       min_gap=22)
        self.cones = self._cones(rng)

    def _sample(self):
        cp = CONTROL_POINTS
        n = len(cp)
        raw = []
        for i in range(n):
            p0, p1, p2, p3 = cp[i - 1], cp[i], cp[(i + 1) % n], cp[(i + 2) % n]
            for k in range(40):
                raw.append(_catmull_rom(p0, p1, p2, p3, k / 40))
        # resample to even spacing
        out = [raw[0]]
        acc = 0.0
        for i in range(1, len(raw) + 1):
            a, b = raw[i - 1], raw[i % len(raw)]
            seg = math.dist(a, b)
            while acc + seg >= SAMPLE_SPACING:
                t = (SAMPLE_SPACING - acc) / seg
                a = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                seg = math.dist(a, b)
                out.append(a)
                acc = 0.0
            acc += seg
        if math.dist(out[-1], out[0]) < SAMPLE_SPACING * 0.5:
            out.pop()
        return out

    def distance_to_centerline(self, x, z, step=1):
        best = float("inf")
        for i in range(0, len(self.points), step):
            px, pz = self.points[i]
            d = (px - x) ** 2 + (pz - z) ** 2
            if d < best:
                best = d
        return math.sqrt(best)

    def nearest(self, x, z, hint=None, window=25):
        """Nearest centerline sample, searching near `hint` if given."""
        n = len(self.points)
        rng = range(n) if hint is None else (i % n for i in range(hint - window, hint + window + 1))
        best_i, best_d = 0, float("inf")
        for i in rng:
            px, pz = self.points[i]
            d = (px - x) ** 2 + (pz - z) ** 2
            if d < best_d:
                best_i, best_d = i, d
        return best_i

    def lateral_offset(self, i, x, z):
        """Signed offset from centerline (+ = right)."""
        px, pz = self.points[i]
        nx, nz = self.normals[i]
        return (x - px) * nx + (z - pz) * nz

    def heading_at(self, i):
        tx, tz = self.tangents[i]
        return math.atan2(tx, tz)

    def _bounds(self):
        xs = [p[0] for p in self.points]
        zs = [p[1] for p in self.points]
        return min(xs), max(xs), min(zs), max(zs)

    def _scatter(self, rng, count, min_clear, spread, min_gap=0.0):
        x0, x1, z0, z1 = self._bounds()
        items = []
        tries = 0
        while len(items) < count and tries < count * 40:
            tries += 1
            x = rng.uniform(x0 - spread, x1 + spread)
            z = rng.uniform(z0 - spread, z1 + spread)
            if self.distance_to_centerline(x, z, step=2) < min_clear:
                continue
            if min_gap and any(math.dist((x, z), (o[0], o[1])) < min_gap for o in items):
                continue
            scale = rng.uniform(0.7, 1.4)
            items.append((x, z, scale, rng.random()))
        return items

    def _cones(self, rng):
        """Cones on the outside of sharp corners."""
        n = len(self.points)
        cones = []
        for i in range(0, n, 3):
            (ax, az), (bx, bz) = self.tangents[i - 4], self.tangents[(i + 4) % n]
            turn = math.atan2(ax * bz - az * bx, ax * bx + az * bz)
            if abs(turn) > 0.35:
                side = -1 if turn > 0 else 1
                px, pz = self.points[i]
                nx, nz = self.normals[i]
                off = ROAD_HALF_WIDTH + 1.6
                cones.append((px + nx * off * side, pz + nz * off * side))
        return cones
