"""Procedural car sounds, no audio files.

The engine is a train of per-cylinder exhaust pulses rendered on a background
thread. Other effects are short looped buffers.
"""

import math
import threading
import time

import numpy as np
import pygame

SR = 22050

IDLE_RPM = 850
REDLINE = 7200
# m/s at redline per gear
GEAR_TOP = [13.0, 23.0, 34.0, 46.0, 62.0]
SHIFT_UP = 6600
SHIFT_DOWN = 3000
CYLINDERS = 4


# engine

def _exhaust_pulse(rng, detune, sr):
    """One cylinder firing through the exhaust."""
    t = np.arange(int(0.07 * sr)) / sr
    p = (1.00 * np.exp(-t / 0.020) * np.sin(2 * np.pi * 85 * detune * t)
         + 0.70 * np.exp(-t / 0.011) * np.sin(2 * np.pi * 210 * detune * t + 0.6)
         + 0.40 * np.exp(-t / 0.006) * np.sin(2 * np.pi * 560 * detune * t + 1.3)
         + 0.20 * np.exp(-t / 0.003) * np.sin(2 * np.pi * 1450 * detune * t)
         + 0.35 * np.exp(-t / 0.0025) * rng.standard_normal(t.size))
    p *= 1 - np.exp(-t / 0.0006)  # soften the attack
    return p / np.abs(p).max()


def _filtered_noise(rng, seconds, smooth, sr):
    """Lowpassed white noise."""
    w = rng.standard_normal(int(seconds * sr))
    out = np.empty_like(w)
    acc = 0.0
    for i, x in enumerate(w):
        acc += (x - acc) * smooth
        out[i] = acc
    return out / np.abs(out).max()


class EngineSynth:
    def __init__(self, sr=SR):
        self.sr = sr
        self.rng = np.random.default_rng()
        self.pulses = [_exhaust_pulse(self.rng, d, sr) for d in (1.0, 1.05, 0.96, 1.02)]
        self.cyl_gain = np.array([1.0, 0.82, 0.93, 0.78])
        self.plen = len(self.pulses[0])
        self.acc = np.zeros(4096 + self.plen)   # overlap-add buffer
        self.intake = _filtered_noise(self.rng, 1.5, 0.12 * 44100 / sr, sr)
        self.lp_coef = 1 - math.exp(-2 * math.pi * 5000 / sr)  # ~5 kHz
        self.noise_pos = 0
        self.phase = 0.0       # four-stroke cycle, 0..1
        self.rpm = IDLE_RPM
        self.throttle = 0.0
        self.lp = 0.0
        # set by the game thread
        self.target_rpm = IDLE_RPM
        self.target_throttle = 0.0
        self.volume = 0.5

    def render(self, n):
        r0, r1 = self.rpm, self.target_rpm
        t0, t1 = self.throttle, self.target_throttle
        self.rpm, self.throttle = r1, t1
        rpm = np.linspace(r0, r1, n, endpoint=False)
        thr = np.linspace(t0, t1, n, endpoint=False)

        # crank angle -> firing events
        ph = self.phase + np.cumsum(rpm / 60.0 / 2.0 / self.sr)
        slot = np.floor(ph * CYLINDERS).astype(np.int64)
        prev = math.floor(self.phase * CYLINDERS)
        fire_idx = np.nonzero(np.diff(slot, prepend=prev))[0]
        self.phase = ph[-1] % 1.0

        if len(self.acc) < n + self.plen:
            grown = np.zeros(n + self.plen)
            grown[:len(self.acc)] = self.acc
            self.acc = grown
        acc = self.acc
        fire_hz = max(r1, 1.0) / 60.0 / 2.0 * CYLINDERS
        level = (28.0 / fire_hz) ** 0.45  # keep loudness even across rpm
        for i in fire_idx:
            cyl = int(slot[i] % CYLINDERS)
            load = thr[i]
            amp = self.cyl_gain[cyl] * (0.3 + 0.7 * load) * (1 + 0.14 * self.rng.standard_normal())
            if load > 0.5 and rpm[i] > REDLINE * 0.97 and self.rng.random() < 0.6:
                amp = 0.0  # rev limiter
            elif load < 0.1 and rpm[i] > 3200 and self.rng.random() < 0.035:
                amp = 2.2  # overrun pop
            acc[i:i + self.plen] += self.pulses[cyl] * (amp * level)

        out = acc[:n].copy()
        acc[:-n] = acc[n:]
        acc[-n:] = 0.0

        # intake hiss
        idx = (self.noise_pos + np.arange(n)) % len(self.intake)
        self.noise_pos = (self.noise_pos + n) % len(self.intake)
        out += self.intake[idx] * (0.04 + 0.18 * thr * rpm / REDLINE)

        out = np.tanh(out * (0.9 + 0.8 * thr))
        y = np.empty(n)
        lp = self.lp
        a = self.lp_coef
        for k in range(n):
            lp += (out[k] - lp) * a
            y[k] = lp
        self.lp = lp
        return (y * self.volume).astype(np.float32)


class EngineStream:
    """Queues EngineSynth chunks gaplessly on a mixer channel."""

    CHUNK = 512  # ~23 ms

    def __init__(self, synth, channel, n_out_channels):
        self.synth = synth
        self.ch = channel
        self.n_out = n_out_channels
        self.running = True
        self.ch.play(self._next())
        self.thread = threading.Thread(target=self._run, name="engine-audio", daemon=True)
        self.thread.start()

    def _next(self):
        pcm = (np.clip(self.synth.render(self.CHUNK), -1, 1) * 32000).astype(np.int16)
        if self.n_out > 1:
            pcm = np.repeat(pcm[:, None], self.n_out, axis=1)
        return pygame.mixer.Sound(buffer=pcm.tobytes())

    def _run(self):
        while self.running:
            try:
                if self.ch.get_queue() is None:
                    if self.ch.get_busy():
                        self.ch.queue(self._next())
                    else:  # underrun
                        self.ch.play(self._next())
            except pygame.error:
                break
            time.sleep(0.004)

    def close(self):
        self.running = False
        self.thread.join(timeout=0.5)
        self.ch.stop()


# effects

def _t(seconds):
    return np.arange(int(seconds * SR)) / SR


def _lowpass(x, smooth):
    out = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += (v - acc) * smooth
        out[i] = acc
    return out


def _norm(x, peak):
    return x * (peak / max(1e-9, np.abs(x).max()))


def _loop_noise(rng, n, lo, hi):
    """Band-limited noise that loops seamlessly (shaped in the FFT domain)."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(spec, n)
    return x / max(1e-9, np.abs(x).max())


def _squeal(rng):
    """Tyre squeal: wandering partials, stick-slip chatter and scrub hiss."""
    n = int(3.0 * SR)
    # pitch drift
    drift = 1 + 0.08 * _loop_noise(rng, n, 0.3, 2.5) + 0.015 * _loop_noise(rng, n, 8, 25)
    base = 980 * drift
    x = np.zeros(n)
    for mult, amp in ((1.0, 1.0), (1.47, 0.55), (2.03, 0.35), (2.9, 0.18)):
        ph = np.cumsum(2 * np.pi * base * mult / SR)
        # whole cycles so the loop closes
        ph *= np.round(ph[-1] / (2 * np.pi)) * 2 * np.pi / ph[-1]
        env = np.clip(0.6 + 0.5 * _loop_noise(rng, n, 0.5, 4), 0, None)
        x += amp * env * np.sin(ph + rng.uniform(0, 2 * np.pi))
    # stick-slip chatter
    chatter = np.clip(0.65 + 0.55 * _loop_noise(rng, n, 20, 70), 0.05, None)
    x *= chatter
    # scrub hiss
    x += 0.45 * _loop_noise(rng, n, 700, 3200) * (0.7 + 0.3 * chatter)
    x += 0.25 * _loop_noise(rng, n, 150, 700)
    x = np.tanh(1.4 * x / np.abs(x).max())
    return _norm(x, 0.7)


def _bonk(rng):
    t = _t(0.45)
    ph = np.cumsum(2 * np.pi * (110 * np.exp(-t * 6) + 45) / SR)
    x = (1.4 * np.sin(ph) + 0.8 * rng.uniform(-1, 1, t.size) * np.exp(-t * 30)) * np.exp(-t * 9)
    return _norm(np.tanh(1.5 * x), 0.9)


def _horn():
    t = _t(0.2)  # whole cycles -> seamless loop
    return _norm(np.tanh(1.8 * (np.sin(2 * np.pi * 400 * t) + np.sin(2 * np.pi * 500 * t))), 0.6)


def _chime():
    t = _t(0.9)
    x = np.zeros_like(t)
    for j, hz in enumerate((660, 880, 1320)):
        tt = np.clip(t - j * 0.12, 0, None)
        x += np.where(t >= j * 0.12, np.sin(2 * np.pi * hz * tt) * np.exp(-tt * 5), 0)
    return _norm(x, 0.6)


def _sound(x, channels):
    pcm = (np.clip(x, -1, 1) * 32000).astype(np.int16)
    if channels > 1:
        pcm = np.repeat(pcm[:, None], channels, axis=1)
    return pygame.mixer.Sound(buffer=pcm.tobytes())


# game side

class Drivetrain:
    """Speed + throttle -> gear and RPM."""

    def __init__(self):
        self.gear = 1
        self.rpm = IDLE_RPM
        self.shift_timer = 0.0

    def update(self, dt, speed, throttle):
        v = abs(speed)
        if speed < -0.3:
            self.gear = 1
        else:
            wheel_rpm = v / GEAR_TOP[self.gear - 1] * REDLINE
            if wheel_rpm > SHIFT_UP and self.gear < len(GEAR_TOP):
                self.gear += 1
                self.shift_timer = 0.22
            elif wheel_rpm < SHIFT_DOWN and self.gear > 1:
                self.gear -= 1
        target = v / GEAR_TOP[self.gear - 1] * REDLINE
        if v < 3.0:
            # clutch slip
            target = max(target, IDLE_RPM + throttle * 4500)
        target = max(IDLE_RPM, min(REDLINE, target))
        if self.shift_timer > 0:
            self.shift_timer -= dt
        rate = 9.0 if target > self.rpm else 6.0
        self.rpm += (target - self.rpm) * min(1.0, dt * rate)
        return self.rpm

    @property
    def shifting(self):
        return self.shift_timer > 0


class CarAudio:
    def __init__(self, enabled=True):
        self.drive = Drivetrain()
        self.ok = False
        self.engine = None
        self.muted = False
        self.prev_crashed = 0.0
        self.prev_laps = 0
        self.throttle = 0.0
        if not enabled:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(SR, -16, 1, 512)
            _, _, channels = pygame.mixer.get_init()
            pygame.mixer.set_num_channels(8)
        except pygame.error as e:
            print(f"[audio] disabled: {e}")
            return

        freq = pygame.mixer.get_init()[0]
        self.synth = EngineSynth(freq)
        self.engine = EngineStream(self.synth, pygame.mixer.Channel(6), channels)

        rng = np.random.default_rng(1)
        mk = lambda x: _sound(x, channels)

        def loop(idx, x):
            ch = pygame.mixer.Channel(idx)
            ch.play(mk(x), loops=-1)
            ch.set_volume(0.0)
            return ch

        self.squeal_ch = loop(0, _squeal(rng))
        self.grass_ch = loop(1, _norm(_lowpass(rng.uniform(-1, 1, SR), 0.04), 0.8))
        self.wind_ch = loop(2, _norm(_lowpass(rng.uniform(-1, 1, SR), 0.25), 0.8))
        self.horn_ch = pygame.mixer.Channel(3)
        self.fx_ch = pygame.mixer.Channel(4)
        self.chime_ch = pygame.mixer.Channel(5)
        self.horn = mk(_horn())
        self.bonk = mk(_bonk(rng))
        self.chime = mk(_chime())
        self.levels = {"squeal": 0.0, "grass": 0.0, "wind": 0.0}
        self.ok = True

    # controls
    def toggle_mute(self):
        self.muted = not self.muted
        if not self.ok:
            return
        if self.muted:
            pygame.mixer.pause()
        else:
            pygame.mixer.unpause()

    def horn_on(self):
        if self.ok and not self.muted:
            self.horn_ch.play(self.horn, loops=-1)

    def horn_off(self):
        if self.ok:
            self.horn_ch.stop()

    def close(self):
        if self.engine:
            self.engine.close()

    # per frame
    def update(self, dt, car, throttle, brake, handbrake, laps):
        rpm = self.drive.update(dt, car.speed, throttle)
        if not self.ok:
            return

        # lift off during upshifts
        load = 0.0 if self.drive.shifting else throttle
        self.throttle += (load - self.throttle) * min(1.0, dt * 15)
        self.synth.target_rpm = rpm
        self.synth.target_throttle = self.throttle

        crash_hit = car.crashed > self.prev_crashed + 0.1
        self.prev_crashed = car.crashed
        lap_done = laps > self.prev_laps
        self.prev_laps = laps
        if self.muted:
            return
        if crash_hit:
            self.fx_ch.play(self.bonk)
        if lap_done:
            self.chime_ch.play(self.chime)

        v = abs(car.speed)
        # squeal on hard cornering, handbrake slides or heavy braking
        slip = max(0.0, abs(car.lateral_accel) - 30.0) / 15.0
        if handbrake and v > 15:
            slip = max(slip, 0.5)
        if brake > 0.8 and car.speed > 40:
            slip = max(slip, 0.3 * (brake - 0.8) / 0.2 * min(1.0, (car.speed - 40) / 12))
        if not car.on_road:
            slip *= 0.3
        targets = {
            "squeal": min(0.6, slip) if v > 3 else 0.0,
            "grass": 0.0 if car.on_road else min(0.7, v / 15.0),
            "wind": min(0.3, (v / 58.0) ** 2 * 0.3),
        }
        chans = {"squeal": self.squeal_ch, "grass": self.grass_ch, "wind": self.wind_ch}
        for k, tgt in targets.items():
            self.levels[k] += (tgt - self.levels[k]) * min(1.0, dt * 12)
            chans[k].set_volume(self.levels[k])
