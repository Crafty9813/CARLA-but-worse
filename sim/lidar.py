"""Simulated spinning lidar: raycasts against scenery and the ground."""

import numpy as np
from OpenGL.GL import (GL_COLOR_ARRAY, GL_FLOAT, GL_FOG, GL_LIGHTING, GL_POINTS, GL_VERTEX_ARRAY,
                       glColorPointer, glDisable, glDisableClientState, glDrawArrays, glEnable,
                       glEnableClientState, glPointSize, glVertexPointer)

from .track import ROAD_HALF_WIDTH

HEIGHT = 1.7        # roof mount, m
RANGE = 80.0
SCAN_HZ = 15
AZIMUTHS = 240
ELEVATIONS = np.radians(np.linspace(-16, 4, 12))

TREE, BUILDING, CONE = 0, 1, 2


def obstacles(track):
    """Rows of (x, z, hit radius, height, kind, box half-width)."""
    rows = [(x, z, 1.6 * s, 6.2 * s, TREE, 2.2 * s) for x, z, s, _ in track.trees]
    rows += [(x, z, 6 * s, 8 + v * 30, BUILDING, 6 * s) for x, z, s, v in track.buildings]
    rows += [(x, z, 0.25, 0.6, CONE, 0.23) for x, z in track.cones]
    return np.array(rows, dtype=float)


def heat(t):
    """0..1 -> blue..red colormap, shape (N, 3)."""
    t = np.clip(np.asarray(t, float), 0, 1)[:, None]
    return np.clip(1.5 - np.abs(4 * t - np.array([3, 2, 1])), 0, 1)


class Lidar:
    def __init__(self, track):
        self.obs = obstacles(track)
        self.track_pts = np.array(track.points)
        self.track_nrm = np.array(track.normals)
        self.az = np.linspace(0, 2 * np.pi, AZIMUTHS, endpoint=False) + 1e-3  # avoid dx == 0
        self.tan_el = np.tan(ELEVATIONS)
        self.rng = np.random.default_rng(3)
        self.timer = 0.0
        self.points = np.zeros((0, 3), np.float32)
        self.colors = np.zeros((0, 3), np.float32)
        self.hits = np.zeros((0, 2))       # obstacle returns, world x/z
        self.markings = np.zeros((0, 2))   # bright ground returns (lines, kerbs)
        self.ranges = np.full(AZIMUTHS, np.inf)  # nearest obstacle per car-relative azimuth

    def update(self, dt, car):
        self.timer -= dt
        if self.timer <= 0:
            self.timer = 1 / SCAN_HZ
            self.scan(car)

    def _ray_hits(self, dx, dz, ox, oz):
        """Horizontal distance along each ray to each nearby obstacle, shape (rays, obstacles)."""
        rx, rz = self.obs[:, 0] - ox, self.obs[:, 1] - oz
        near = np.hypot(rx, rz) < RANGE + self.obs[:, 5]
        obs, rx, rz = self.obs[near], rx[near], rz[near]
        t = np.full((len(dx), len(obs)), np.inf)

        rnd = obs[:, 4] != BUILDING  # trees and cones are circles
        tc = dx[:, None] * rx[rnd] + dz[:, None] * rz[rnd]
        disc = obs[rnd, 2] ** 2 - (rx[rnd] ** 2 + rz[rnd] ** 2 - tc ** 2)
        th = tc - np.sqrt(np.maximum(disc, 0))
        t[:, rnd] = np.where((disc > 0) & (th > 0), th, np.inf)

        sq = ~rnd  # buildings are boxes: slab test
        hw = obs[sq, 5]
        tx1, tx2 = (rx[sq] - hw) / dx[:, None], (rx[sq] + hw) / dx[:, None]
        tz1, tz2 = (rz[sq] - hw) / dz[:, None], (rz[sq] + hw) / dz[:, None]
        tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(tz1, tz2))
        tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(tz1, tz2))
        t[:, sq] = np.where((tmax >= tmin) & (tmin > 0), tmin, np.inf)
        return t, obs

    def scan(self, car):
        ox, oz = car.x, car.z
        ang = car.heading + self.az
        dx, dz = np.sin(ang), np.cos(ang)
        n_ch, n_az = len(self.tan_el), len(dx)

        t, obs = self._ray_hits(dx, dz, ox, oz)
        if len(obs):
            y = HEIGHT + t[None] * self.tan_el[:, None, None]
            d_obj = np.where((y >= 0) & (y <= obs[:, 3]), t[None], np.inf).min(axis=2)
        else:
            d_obj = np.full((n_ch, n_az), np.inf)
        d_gnd = np.where(self.tan_el < 0, HEIGHT / np.abs(self.tan_el), np.inf)
        d_gnd = np.broadcast_to(d_gnd[:, None], (n_ch, n_az))

        is_obj = d_obj < d_gnd
        self.ranges = np.where(is_obj, d_obj, np.inf).min(axis=0)
        d = np.where(is_obj, d_obj, d_gnd) + self.rng.normal(0, 0.03, (n_ch, n_az))
        ok = d < RANGE

        dd = d - 0.12  # nudge towards the sensor so points don't z-fight the surface
        px = (ox + dd * dx)[ok]
        pz = (oz + dd * dz)[ok]
        py = np.maximum(HEIGHT + dd * self.tan_el[:, None], 0.08)[ok]
        is_obj, dist = is_obj[ok], d[ok]

        colors = np.empty((len(px), 3))
        colors[is_obj] = heat(1 - dist[is_obj] / 50)
        g = ~is_obj
        inten = self._intensity(px[g], pz[g], car.track_index)
        colors[g] = inten[:, None] * np.array([0.55, 0.9, 1.0])

        self.points = np.ascontiguousarray(np.column_stack([px, py, pz]), np.float32)
        self.colors = np.ascontiguousarray(colors, np.float32)
        self.hits = np.column_stack([px[is_obj], pz[is_obj]])
        bright = inten > 0.7
        self.markings = np.column_stack([px[g][bright], pz[g][bright]])

    def _intensity(self, px, pz, i):
        """Return strength by surface: paint and kerbs bright, asphalt dull, grass dark."""
        n = len(self.track_pts)
        idx = (i + np.arange(-45, 46)) % n
        P, N = self.track_pts[idx], self.track_nrm[idx]
        k = ((px[:, None] - P[:, 0]) ** 2 + (pz[:, None] - P[:, 1]) ** 2).argmin(axis=1)
        off = np.abs((px - P[k, 0]) * N[k, 0] + (pz - P[k, 1]) * N[k, 1])
        hw = ROAD_HALF_WIDTH
        out = np.full(len(px), 0.12)
        out[off < hw + 1.2] = 0.8
        out[off < hw] = 0.3
        out[(off > hw - 0.65) & (off < hw - 0.4)] = 1.0
        out[off < 0.15] = 0.9
        return out

    def draw(self):
        if not len(self.points):
            return
        glDisable(GL_LIGHTING)
        glDisable(GL_FOG)
        glPointSize(3)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.points)
        glColorPointer(3, GL_FLOAT, 0, self.colors)
        glDrawArrays(GL_POINTS, 0, len(self.points))
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glEnable(GL_FOG)
        glEnable(GL_LIGHTING)
