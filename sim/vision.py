"""Simulated front camera: object detection boxes and lane detection."""

import math

import numpy as np

from .lidar import AZIMUTHS, BUILDING, CONE, TREE, obstacles
from .track import ROAD_HALF_WIDTH

RANGE = 110.0
MAX_BOXES = 24
LABELS = {TREE: "tree", BUILDING: "building", CONE: "cone"}
COLORS = {TREE: (0.35, 1.0, 0.35), BUILDING: (1.0, 0.6, 0.15), CONE: (1.0, 0.3, 0.85)}

_SX = np.array([-1, 1, 1, -1, -1, 1, 1, -1])
_SY = np.array([0, 0, 0, 0, 1, 1, 1, 1])
_SZ = np.array([-1, -1, 1, 1, -1, -1, 1, 1])


def project(pts, mv, proj, w, h):
    """World points (N, 3) -> pixels from the top-left (N, 2), plus an in-front-of-camera mask."""
    clip = np.hstack([pts, np.ones((len(pts), 1))]) @ mv @ proj
    cw = clip[:, 3]
    front = cw > 0.1
    cw = np.where(front, cw, 1.0)
    px = (clip[:, 0] / cw + 1) * 0.5 * w
    py = (1 - clip[:, 1] / cw) * 0.5 * h
    return np.column_stack([px, py]), front


def _polylines(pts, mv, proj, w, h):
    """Project a 3D polyline and split it wherever it goes behind the camera."""
    px, front = project(pts, mv, proj, w, h)
    out, cur = [], []
    for p, f in zip(px, front):
        if f:
            cur.append(p)
        elif cur:
            out.append(np.array(cur))
            cur = []
    if len(cur) > 1:
        out.append(np.array(cur))
    return [line for line in out if len(line) > 1]


class Detector:
    def __init__(self, track, lidar):
        self.lidar = lidar
        self.obs = obstacles(track)
        self.jitter = np.random.default_rng(5).random(len(self.obs))
        self.pts = np.array(track.points)
        self.nrm = np.array(track.normals)
        self.detections = []   # (kind, confidence, (x0, y0, x1, y1), distance)
        self.lanes = []        # (polyline, rgba)

    def run(self, car, mv, proj, w, h, path=None):
        self._detect(car, mv, proj, w, h)
        self._lanes(car, mv, proj, w, h, path)

    def _detect(self, car, mv, proj, w, h):
        o = self.obs
        rx, rz = o[:, 0] - car.x, o[:, 1] - car.z
        dist = np.hypot(rx, rz)
        fwd = rx * math.sin(car.heading) + rz * math.cos(car.heading)
        cand = np.nonzero((dist < RANGE) & (fwd > 0))[0]

        # hide objects the lidar says are behind something closer
        local = (np.arctan2(rx[cand], rz[cand]) - car.heading) % (2 * np.pi)
        bins = np.round(local / (2 * np.pi) * AZIMUTHS).astype(int) % AZIMUTHS
        cand = cand[dist[cand] - o[cand, 2] * 1.42 <= self.lidar.ranges[bins] + 1.5]

        dets = []
        if len(cand):
            hw, ht = o[cand, 5][:, None], o[cand, 3][:, None]
            corners = np.stack([o[cand, 0][:, None] + _SX * hw, _SY * ht,
                                o[cand, 1][:, None] + _SZ * hw], axis=-1).reshape(-1, 3)
            px, front = project(corners, mv, proj, w, h)
            px, front = px.reshape(-1, 8, 2), front.reshape(-1, 8).all(axis=1)
            for k, i in enumerate(cand):
                if not front[k]:
                    continue
                x0, y0 = np.maximum(px[k].min(axis=0), 0)
                x1, y1 = px[k].max(axis=0)
                x1, y1 = min(x1, w), min(y1, h)
                if x1 - x0 < 3 or y1 - y0 < 3:
                    continue
                kind = int(o[i, 4])
                falloff = 0.45 if kind == CONE else 0.3
                conf = 0.98 - falloff * dist[i] / RANGE - 0.08 * self.jitter[i]
                dets.append((kind, conf, (x0, y0, x1, y1), dist[i]))
        dets.sort(key=lambda d: -d[3])  # far first so near boxes draw on top
        self.detections = dets[-MAX_BOXES:]

    def _lanes(self, car, mv, proj, w, h, path):
        n = len(self.pts)
        idx = (car.track_index + np.arange(1, 40)) % n
        P, N = self.pts[idx], self.nrm[idx]
        lanes = []
        edge = ROAD_HALF_WIDTH - 0.52
        for off, color in ((-edge, (0.2, 1, 0.4, 0.9)), (edge, (0.2, 1, 0.4, 0.9)),
                           (0.0, (1, 0.9, 0.2, 0.8))):
            pts = np.column_stack([P[:, 0] + N[:, 0] * off, np.full(len(P), 0.05),
                                   P[:, 1] + N[:, 1] * off])
            lanes += [(line, color) for line in _polylines(pts, mv, proj, w, h)]
        if path is not None:
            pts = np.column_stack([path[:, 0], np.full(len(path), 0.05), path[:, 1]])
            lanes += [(line, (0.2, 0.8, 1, 0.9)) for line in _polylines(pts, mv, proj, w, h)]
        self.lanes = lanes
