"""CARLA-but-worse: drive a 3D car with an Arduino + MPU6050 steering wheel."""

import argparse
import math
import os
import time

import numpy as np
import pygame
from OpenGL.GL import (GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_SCISSOR_TEST, glCallList,
                       glClear, glDisable, glEnable, glScissor, glViewport)

from .audio import SR, CarAudio
from .autopilot import Autopilot
from .car import Car
from .lidar import SCAN_HZ, Lidar, heat
from .render import Camera, Hud, build_world, draw_car, front_camera, init_gl
from .track import Track
from .vision import COLORS, LABELS, Detector
from .wheel_input import WheelInput

HELP = [
    "W/Up  gas      S/Down  brake/reverse   Space  handbrake",
    "A/D   steer (keyboard fallback)        R  reset to track",
    "C  re-centre wheel   M  mount mode   I  invert steering",
    "[ ]  wheel range     V  camera       H  hide help   Esc  quit",
    "B  horn   N  mute sound   F  FSD (or the D4 button)",
]


def fmt_time(t):
    if t is None:
        return "--:--.---"
    return f"{int(t // 60):02d}:{t % 60:06.3f}"


def rgb255(c):
    return tuple(int(v * 255) for v in c[:3])


class LapTimer:
    def __init__(self, n_samples):
        self.n = n_samples
        self.start = None
        self.last = None
        self.best = None
        self.laps = 0
        self.sectors = set()
        self.prev_index = None

    def update(self, index, moving, now):
        if self.start is None and moving:
            self.start = now
        if self.prev_index is not None and self.start is not None:
            self.sectors.add(index * 4 // self.n)
            crossed = self.prev_index > self.n * 0.9 and index < self.n * 0.1
            if crossed and len(self.sectors) >= 4:
                lap = now - self.start
                self.last = lap
                self.best = lap if self.best is None else min(self.best, lap)
                self.laps += 1
                self.start = now
                self.sectors = {0}
        self.prev_index = index

    def current(self, now):
        return None if self.start is None else now - self.start


def draw_camera_panel(hud, detector, x, y, w, h):
    hud.clip(x, y, w, h)
    for line, color in detector.lanes:
        hud.lines(line + (x, y), color, 3)
    for kind, conf, (x0, y0, x1, y1), _ in detector.detections:
        col = COLORS[kind]
        hud.outline(x + x0, y + y0, x + x1, y + y1, (*col, 0.95), 2)
        ly = y + y0 - 15 if y0 > 15 else y + y0 + 2
        hud.text(f"{LABELS[kind]} {conf:.2f}", x + x0 + 2, ly, hud.tiny, color=rgb255(col))
    hud.panel(x, y + h - 20, w, 20, 0.6)
    hud.text(f"FRONT CAMERA   {len(detector.detections)} objects   lanes "
             f"{'OK' if detector.lanes else '--'}", x + 6, y + h - 18, hud.tiny,
             color=(140, 230, 255))
    hud.unclip()
    hud.outline(x, y, x + w, y + h, (0.3, 0.85, 1, 0.9), 2)


def draw_lidar_panel(hud, lidar, car, x, y, size, now):
    view = 40.0
    scale = size / 2 / view
    cx, cy = x + size / 2, y + size * 0.62
    sh, ch = math.sin(car.heading), math.cos(car.heading)

    def to_px(world):
        dx, dz = world[:, 0] - car.x, world[:, 1] - car.z
        fwd, right = dx * sh + dz * ch, -dx * ch + dz * sh
        return np.column_stack([cx + right * scale, cy - fwd * scale]), np.hypot(fwd, right)

    hud.panel(x, y, size, size, 0.65)
    hud.clip(x, y, size, size)
    for r in (10, 20, 30, 40):
        hud.circle(cx, cy, r * scale, (0.3, 0.8, 1, 0.25))
    a = now * 2 * math.pi
    hud.lines(((cx, cy), (cx + math.sin(a) * size, cy - math.cos(a) * size)), (0.3, 0.9, 1, 0.35), 2)
    if len(lidar.markings):
        px, _ = to_px(lidar.markings)
        hud.cloud(px, np.tile([0.5, 0.85, 1.0, 0.45], (len(px), 1)), 2)
    if len(lidar.hits):
        px, d = to_px(lidar.hits)
        hud.cloud(px, heat(1 - d / view), 3)
    hud.outline(cx - 4, cy - 9, cx + 4, cy + 9, (1, 1, 1, 0.9), 2)
    hud.unclip()
    hud.text(f"LIDAR {len(lidar.points)} pts {SCAN_HZ} Hz", x + 6, y + 4, hud.tiny,
             color=(140, 230, 255))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", help="serial port (default: auto-detect /dev/ttyACM*, /dev/ttyUSB*, COM*)")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--range", type=float, default=None,
                   help="degrees of wheel rotation for full lock (default 90 wheel / 45 tilt)")
    p.add_argument("--invert", action="store_true", help="flip steering direction")
    p.add_argument("--no-serial", action="store_true", help="keyboard only")
    p.add_argument("--no-sound", action="store_true", help="disable sound effects")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--size", default="1280x720")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
    pygame.mixer.pre_init(SR, -16, 1, 512)  # small buffer for low latency
    pygame.init()
    pygame.display.set_caption("CARLA but worse")
    if args.fullscreen:
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN
        screen = pygame.display.set_mode((0, 0), flags)
    else:
        w, h = (int(v) for v in args.size.lower().split("x"))
        screen = pygame.display.set_mode((w, h), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
    width, height = screen.get_size()

    init_gl()
    track = Track()
    world = build_world(track)
    car = Car(track)
    camera = Camera()
    hud = Hud()
    timer = LapTimer(len(track.points))
    audio = CarAudio(enabled=not args.no_sound)
    lidar = Lidar(track)
    detector = Detector(track, lidar)
    pilot = Autopilot(track)

    wheel = None
    if not args.no_serial:
        wheel = WheelInput(port=args.port, baud=args.baud, steer_range_deg=args.range,
                           invert=args.invert)

    fsd = False
    fsd_seen = 0
    fsd_msg, fsd_msg_until = "", 0.0

    def set_fsd(on, msg):
        nonlocal fsd, fsd_msg, fsd_msg_until
        fsd = on
        fsd_msg, fsd_msg_until = msg, time.perf_counter() + 2.0
        lidar.timer = 0.0

    key_steer = 0.0
    show_help = True
    clock = pygame.time.Clock()
    running = True
    while running:
        dt = min(clock.tick(120) / 1000.0, 1 / 20)
        now = time.perf_counter()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.VIDEORESIZE and not args.fullscreen:
                width, height = ev.w, ev.h
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_r:
                    car.reset()
                elif k == pygame.K_v:
                    camera.cycle()
                elif k == pygame.K_h:
                    show_help = not show_help
                elif k == pygame.K_b:
                    audio.horn_on()
                elif k == pygame.K_n:
                    audio.toggle_mute()
                elif k == pygame.K_f:
                    set_fsd(not fsd, "FSD ENGAGED" if not fsd else "FSD OFF")
                elif wheel and k == pygame.K_c:
                    wheel.calibrate()
                elif wheel and k == pygame.K_m:
                    wheel.cycle_mode()
                elif wheel and k == pygame.K_i:
                    wheel.invert = not wheel.invert
                elif wheel and k == pygame.K_LEFTBRACKET:
                    wheel.steer_range_deg = max(15.0, wheel.range_deg() - 10)
                elif wheel and k == pygame.K_RIGHTBRACKET:
                    wheel.steer_range_deg = min(270.0, wheel.range_deg() + 10)
            elif ev.type == pygame.KEYUP and ev.key == pygame.K_b:
                audio.horn_off()

        if wheel and wheel.fsd_presses != fsd_seen:
            if (wheel.fsd_presses - fsd_seen) % 2:
                set_fsd(not fsd, "FSD ENGAGED" if not fsd else "FSD OFF")
            fsd_seen = wheel.fsd_presses

        keys = pygame.key.get_pressed()
        throttle = 1.0 if keys[pygame.K_w] or keys[pygame.K_UP] else 0.0
        brake = 1.0 if keys[pygame.K_s] or keys[pygame.K_DOWN] else 0.0
        handbrake = keys[pygame.K_SPACE]
        if wheel and wheel.connected:
            throttle = max(throttle, 1.0 if wheel.gas else 0.0)
            brake = max(brake, 1.0 if wheel.brake else 0.0)

        # ramp keyboard steering so taps aren't twitchy
        k_target = (1.0 if keys[pygame.K_d] or keys[pygame.K_RIGHT] else 0.0) - \
                   (1.0 if keys[pygame.K_a] or keys[pygame.K_LEFT] else 0.0)
        key_steer += (k_target - key_steer) * min(1.0, dt * (8 if k_target else 12))

        wheel_steer = wheel.steering() if wheel else None
        steer = wheel_steer if (wheel_steer is not None and k_target == 0) else key_steer

        if fsd and (brake > 0 or k_target != 0 or handbrake):
            set_fsd(False, "FSD DISENGAGED - driver took over")
        if fsd:
            lidar.update(dt, car)
            steer, throttle, brake = pilot.update(car, lidar)

        car.update(dt, steer, throttle, brake, handbrake)
        timer.update(car.track_index, abs(car.speed) > 0.5, now)
        audio.update(dt, car, throttle, brake, handbrake, timer.laps)

        # 3D
        glViewport(0, 0, width, height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        camera.apply(car, dt, width / max(1, height))
        glCallList(world)
        draw_car(car)
        if fsd:
            lidar.draw()

        # front camera inset for the vision sim
        cam_w = int(min(480, max(240, width - 680)))
        cam_h = cam_w * 9 // 16
        cam_x, cam_y = (width - cam_w) // 2, 16
        if fsd:
            gl_y = height - cam_y - cam_h
            glViewport(cam_x, gl_y, cam_w, cam_h)
            glEnable(GL_SCISSOR_TEST)
            glScissor(cam_x, gl_y, cam_w, cam_h)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            mv, proj = front_camera(car, cam_w / cam_h)
            glCallList(world)
            glDisable(GL_SCISSOR_TEST)
            glViewport(0, 0, width, height)
            detector.run(car, mv, proj, cam_w, cam_h, pilot.path(car))

        # HUD
        hud.begin(width, height)
        hud.panel(16, 16, 300, 150)
        hud.text(f"{car.kmh:5.0f} km/h", 28, 22, hud.big)
        hud.text(f"LAP  {fmt_time(timer.current(now))}", 28, 80)
        hud.text(f"LAST {fmt_time(timer.last)}", 28, 105, color=(200, 200, 200))
        hud.text(f"BEST {fmt_time(timer.best)}", 28, 130, color=(255, 215, 90))
        hud.text(f"laps: {timer.laps}", 304, 130, hud.small, anchor="tr")
        gear = "R" if car.speed < -0.3 else str(audio.drive.gear)
        hud.text(f"{gear}  {audio.drive.rpm:4.0f} rpm", 304, 30, hud.small, anchor="tr")

        hud.bar(16, 176, 300, 10, throttle, (0.2, 0.85, 0.3))
        hud.bar(16, 190, 300, 10, brake, (0.9, 0.25, 0.2))
        hud.panel(16, 204, 300, 10, 0.5)
        hud.bar(166 + min(0, steer) * 150, 204, abs(steer) * 150, 10, 1.0, (0.3, 0.6, 1.0))

        # FSD status
        hud.panel(16, 222, 300, 52 if fsd else 26)
        if fsd:
            hud.text("FSD ENGAGED", 28, 226, hud.small, color=(90, 220, 255))
            hud.text(pilot.status, 304, 226, hud.small, anchor="tr",
                     color=(255, 120, 90) if pilot.status in ("OBSTACLE", "BLOCKED") else (230, 230, 230))
            obstacle = f"{pilot.obstacle:4.1f} m" if pilot.obstacle is not None else "clear"
            hud.text(f"target {pilot.target_speed * 3.6:4.0f} km/h   path {obstacle}", 28, 248, hud.small)
        else:
            hud.text("FSD off  (F or D4 button)", 28, 226, hud.small, color=(170, 170, 170))

        alert_y = cam_y + cam_h + 16 if fsd else 60
        if not car.on_road:
            hud.text("OFF TRACK", width // 2, alert_y, hud.big, color=(255, 255, 255), anchor="tc")
        if car.crashed > 0:
            hud.text("BONK", width // 2, alert_y + 50, hud.big, color=(255, 80, 60), anchor="tc")
        if now < fsd_msg_until:
            hud.text(fsd_msg, width // 2, height - 140, hud.font, color=(90, 220, 255), anchor="tc")

        # wheel status, bottom-left
        status_y = height - 160
        using_wheel = wheel_steer is not None and k_target == 0 and not fsd
        if wheel:
            hud.steering_wheel(80, height - 85, 55, steer * wheel.range_deg(), using_wheel)
            hud.text(wheel.status, 150, status_y + 40, hud.small,
                     color=(150, 255, 150) if wheel.connected else (255, 190, 120))
            if wheel.connected:
                hud.text(f"angle {wheel.angle:+6.1f} deg  range +-{wheel.range_deg():.0f}  "
                         f"{wheel.sample_rate:.0f} Hz  {'INVERTED' if wheel.invert else ''}",
                         150, status_y + 62, hud.small)
        else:
            hud.text("keyboard only (--no-serial)", 16, height - 30, hud.small)

        hud.minimap(track, car, width - 216, 16, 200)
        hud.text(f"cam: {Camera.MODES[camera.mode]}   {clock.get_fps():.0f} fps",
                 width - 16, 222, hud.small, anchor="tr")
        if fsd:
            draw_camera_panel(hud, detector, cam_x, cam_y, cam_w, cam_h)
            draw_lidar_panel(hud, lidar, car, width - 216, 246, 200, now)
        if show_help:
            y = height - 100
            for line in HELP:
                hud.text(line, width - 16, y, hud.small, color=(230, 230, 230), anchor="tr")
                y += 20
        hud.end()

        pygame.display.flip()

    if wheel:
        wheel.close()
    audio.close()
    pygame.quit()


if __name__ == "__main__":
    main()
