"""Kinematic bicycle-model car.

Forward is (sin h, cos h), right is (-cos h, sin h); steering right decreases h.
"""

import math

from .track import ROAD_HALF_WIDTH

WHEELBASE = 2.6
MAX_STEER = math.radians(34)
ENGINE_ACCEL = 11.0       # m/s^2
BRAKE_DECEL = 16.0
REVERSE_MAX = 7.0
TOP_SPEED_ROAD = 58.0     # m/s (~210 km/h)
TOP_SPEED_GRASS = 18.0
AIR_DRAG = 0.0012
ROLL_ROAD = 0.15
ROLL_GRASS = 1.4
TREE_RADIUS = 1.1
BUILDING_RADIUS = 7.0
CAR_RADIUS = 1.1


class Car:
    def __init__(self, track):
        self.track = track
        self.reset(start=True)

    def reset(self, start=False):
        t = self.track
        if start:
            i = 3
            self.x, self.z = t.points[i]
            nx, nz = t.normals[i]
            self.x += nx * 2.5  # right lane
            self.z += nz * 2.5
        else:
            i = t.nearest(self.x, self.z)
            self.x, self.z = t.points[i]
        self.heading = t.heading_at(i)
        self.track_index = i
        self.speed = 0.0
        self.steer = 0.0           # front wheel angle, rad
        self.yaw_rate = 0.0
        self.lateral_accel = 0.0
        self.on_road = True
        self.wheel_spin = 0.0
        self.crashed = 0.0         # crash feedback timer, s

    def update(self, dt, steer_input, throttle, brake, handbrake=False):
        t = self.track

        # less lock at speed
        speed_factor = 1.0 / (1.0 + abs(self.speed) * 0.035)
        target = steer_input * MAX_STEER * speed_factor
        # rate-limited steering rack
        max_delta = math.radians(260) * dt
        self.steer += max(-max_delta, min(max_delta, target - self.steer))

        self.track_index = t.nearest(self.x, self.z, hint=self.track_index)
        off = abs(t.lateral_offset(self.track_index, self.x, self.z))
        self.on_road = off <= ROAD_HALF_WIDTH + 0.3
        top = TOP_SPEED_ROAD if self.on_road else TOP_SPEED_GRASS
        roll = ROLL_ROAD if self.on_road else ROLL_GRASS

        v = self.speed
        a = 0.0
        if throttle > 0:
            if v < -0.5:
                a += BRAKE_DECEL * throttle
            else:
                a += ENGINE_ACCEL * throttle * max(0.0, 1.0 - v / top)
        if brake > 0:
            if v > 0.5:
                a -= BRAKE_DECEL * brake
            elif v > -REVERSE_MAX:
                a -= ENGINE_ACCEL * 0.5 * brake
        if handbrake:
            a -= math.copysign(BRAKE_DECEL * 0.7, v) if abs(v) > 0.3 else 0
        a -= AIR_DRAG * v * abs(v) + roll * v
        if v > top:  # slow down on grass
            a -= (v - top) * 1.5

        new_v = v + a * dt
        if throttle == 0 and brake == 0 and v * new_v < 0:
            new_v = 0.0
        if brake > 0 and v > 0.5 and new_v < 0:
            new_v = 0.0  # stop, then reverse if held
        self.speed = new_v

        yaw_rate = self.speed / WHEELBASE * math.tan(self.steer)
        if handbrake and abs(self.speed) > 5:
            yaw_rate *= 1.5
        self.heading -= yaw_rate * dt
        self.yaw_rate = yaw_rate
        self.lateral_accel += (self.speed * yaw_rate - self.lateral_accel) * min(1.0, dt * 6)

        self.x += math.sin(self.heading) * self.speed * dt
        self.z += math.cos(self.heading) * self.speed * dt
        self.wheel_spin += self.speed * dt / 0.34

        self._collide()
        self.crashed = max(0.0, self.crashed - dt)

    def _collide(self):
        obstacles = [(o, TREE_RADIUS * o[2]) for o in self.track.trees]
        obstacles += [(o, BUILDING_RADIUS * o[2]) for o in self.track.buildings]
        for (tx, tz, _, _), radius in obstacles:
            rr = radius + CAR_RADIUS
            dx, dz = self.x - tx, self.z - tz
            if abs(dx) > rr or abs(dz) > rr:
                continue
            d = math.hypot(dx, dz)
            if 1e-6 < d < rr:
                self.x = tx + dx / d * rr
                self.z = tz + dz / d * rr
                if abs(self.speed) > 3:
                    self.crashed = 0.6
                self.speed *= 0.2

    @property
    def kmh(self):
        return abs(self.speed) * 3.6
