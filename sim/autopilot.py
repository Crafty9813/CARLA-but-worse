"""FSD: pure-pursuit lane following with curvature and lidar-based speed control."""

import math

import numpy as np

from .car import MAX_STEER, WHEELBASE
from .track import SAMPLE_SPACING

LANE_OFFSET = 2.5     # right-hand lane
MAX_SPEED = 38.0      # m/s (~137 km/h)
LAT_ACCEL = 11.0      # cornering limit we plan for
DECEL = 7.0           # comfortable braking
CORRIDOR = 1.4        # half-width of the path checked for obstacles
STOP_GAP = 7.0        # metres to keep from an obstacle


def _clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


class Autopilot:
    def __init__(self, track):
        pts, nrm = np.array(track.points), np.array(track.normals)
        self.lane = pts + nrm * LANE_OFFSET
        tan = np.array(track.tangents)
        h = np.arctan2(tan[:, 0], tan[:, 1])
        dh = np.angle(np.exp(1j * (np.roll(h, -3) - np.roll(h, 3))))
        curv = np.abs(dh) / (6 * SAMPLE_SPACING)
        self.vlimit = np.minimum(np.sqrt(LAT_ACCEL / np.maximum(curv, 1e-4)), MAX_SPEED)
        self.status = "standby"
        self.target_speed = 0.0
        self.obstacle = None

    def path(self, car, count=30):
        idx = (car.track_index + np.arange(1, count)) % len(self.lane)
        return np.vstack([[car.x, car.z], self.lane[idx]])

    def update(self, car, lidar):
        """Returns (steer, throttle, brake)."""
        n = len(self.lane)
        i = car.track_index
        sh, ch = math.sin(car.heading), math.cos(car.heading)

        # steering: pure pursuit towards a point further ahead at speed
        look = 5 + 0.45 * max(car.speed, 0)
        tx, tz = self.lane[(i + max(1, round(look / SAMPLE_SPACING))) % n]
        dx, dz = tx - car.x, tz - car.z
        alpha = math.atan2(-dx * ch + dz * sh, dx * sh + dz * ch)
        ld = max(math.hypot(dx, dz), 1.0)
        kappa = 2 * math.sin(alpha) / ld
        speed_factor = 1.0 / (1.0 + abs(car.speed) * 0.035)
        steer = _clamp(math.atan(WHEELBASE * kappa) / (MAX_STEER * speed_factor))

        # speed: slow early enough for every corner in the braking horizon
        k = np.arange(120)
        vt = float(np.sqrt(self.vlimit[(i + k) % n] ** 2 + 2 * DECEL * k * SAMPLE_SPACING).min())
        self.status = "cruise" if vt > MAX_SPEED - 1 else "corner"

        lx, lz = self.lane[i]
        if math.hypot(car.x - lx, car.z - lz) > 4:
            vt = min(vt, 12.0)
            self.status = "rejoin lane"

        # obstacles in the arc we're about to drive
        self.obstacle = None
        if len(lidar.hits):
            hx, hz = lidar.hits[:, 0] - car.x, lidar.hits[:, 1] - car.z
            f = hx * sh + hz * ch
            r = -hx * ch + hz * sh
            in_path = (f > 0.5) & (np.abs(r - kappa * f * f / 2) < CORRIDOR)
            if in_path.any():
                self.obstacle = float(f[in_path].min())
                v_obs = math.sqrt(max(0.0, 2 * DECEL * (self.obstacle - STOP_GAP)))
                if v_obs < vt:
                    vt = v_obs
                    self.status = "OBSTACLE" if v_obs > 0 else "BLOCKED"

        self.target_speed = vt
        err = vt - car.speed
        return steer, _clamp(err * 0.4, 0, 1), _clamp(-err * 0.25, 0, 1)
