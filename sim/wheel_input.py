"""Reads the Arduino + MPU6050 over serial and turns it into a steering angle.

Modes: "wheel" (upright, rotate about Z) or "tilt-x"/"tilt-y" (flat, tilt like a tray).
A complementary filter blends gyro (smooth) with accel (no drift).
"""

import glob
import math
import sys
import threading
import time

try:
    import serial
except ImportError:
    serial = None

MODES = ("wheel", "tilt-x", "tilt-y")
GYRO_WEIGHT = 0.96
CALIBRATION_SAMPLES = 60    # ~0.6 s
FSD_DEBOUNCE = 0.3


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def _accel_angle(mode, ax, ay, az):
    """Angle (deg) about the mode's axis, same sign as the gyro."""
    if mode == "wheel":
        return math.degrees(math.atan2(ax, ay))
    if mode == "tilt-x":
        return math.degrees(math.atan2(ay, az))
    return math.degrees(math.atan2(-ax, az))


def _gyro_rate(mode, gx, gy, gz):
    return {"wheel": gz, "tilt-x": gx, "tilt-y": gy}[mode]


def find_port():
    candidates = []
    if sys.platform.startswith("win"):
        try:
            from serial.tools import list_ports
            candidates = [p.device for p in list_ports.comports()]
        except Exception:
            pass
    else:
        candidates = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
                            + glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/cu.usbserial*"))
    return candidates[0] if candidates else None


class WheelInput:
    def __init__(self, port=None, baud=115200, steer_range_deg=None, invert=False):
        self.port = port
        self.baud = baud
        self.steer_range_deg = steer_range_deg
        self.invert = invert
        self.deadzone_deg = 2.0

        self.mode = "wheel"
        self.connected = False
        self.status = "no board - keyboard only"
        self.gas = False
        self.brake = False
        self.fsd_presses = 0      # FSD button press count
        self.angle = 0.0          # degrees from centre
        self.sample_rate = 0.0

        self._lock = threading.Lock()
        self._ref = 0.0
        self._gyro_bias = (0.0, 0.0, 0.0)
        self._filtered_abs = None
        self._calib = []
        self._calibrating = True
        self._auto_mode = True
        self._last_t = None
        self._last_sample = None
        self._stop = False
        self._fsd_down = False
        self._fsd_last = 0.0

        if serial is None:
            self.status = "pyserial not installed - keyboard only"
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # public

    def calibrate(self):
        with self._lock:
            self._calib = []
            self._calibrating = True
            self._auto_mode = True

    def cycle_mode(self):
        with self._lock:
            self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
            self._auto_mode = False
            self._calib = []
            self._calibrating = True

    def range_deg(self):
        """Degrees for full lock."""
        if self.steer_range_deg:
            return self.steer_range_deg
        return 90.0 if self.mode == "wheel" else 45.0

    def steering(self):
        """-1 (left) .. +1 (right), or None if no board."""
        if not self.connected or self._calibrating:
            return None
        a = self.angle
        if abs(a) < self.deadzone_deg:
            return 0.0
        a -= math.copysign(self.deadzone_deg, a)
        # clockwise in wheel mode is negative about Z
        sign = -1.0 if self.mode == "wheel" else 1.0
        if self.invert:
            sign = -sign
        s = sign * a / (self.range_deg() - self.deadzone_deg)
        return max(-1.0, min(1.0, s))

    def close(self):
        self._stop = True

    # serial thread

    def _run(self):
        while not self._stop:
            port = self.port or find_port()
            if not port:
                self.connected = False
                self.status = "no board found - keyboard only"
                time.sleep(1.0)
                continue
            try:
                with serial.Serial(port, self.baud, timeout=1.0) as ser:
                    self.status = f"{port}: waiting for Arduino..."
                    self._read_loop(ser, port)
            except (serial.SerialException, OSError) as e:
                self.connected = False
                self.status = f"{port}: {e.__class__.__name__} - retrying"
                time.sleep(1.0)

    def _read_loop(self, ser, port):
        count, window_start = 0, time.perf_counter()
        while not self._stop:
            raw = ser.readline()
            if not raw:
                self.connected = False
                self.status = f"{port}: no data (is the sketch uploaded? baud {self.baud}?)"
                continue
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("E,"):
                self.connected = False
                self.status = line[2:]
                continue
            if not line.startswith("S,"):
                continue
            parts = line.split(",")
            if len(parts) not in (9, 10):  # 9 = old sketch without the FSD button
                continue
            try:
                ax, ay, az, gx, gy, gz = (float(p) for p in parts[1:7])
                gas, brake = parts[7] == "1", parts[8] == "1"
                fsd = len(parts) == 10 and parts[9] == "1"
            except ValueError:
                continue

            self._process(ax, ay, az, gx, gy, gz)
            self.gas, self.brake = gas, brake
            self._fsd_button(fsd)
            self.connected = True

            count += 1
            now = time.perf_counter()
            if now - window_start >= 1.0:
                self.sample_rate = count / (now - window_start)
                count, window_start = 0, now
            if not self._calibrating:
                self.status = f"{port} - {self.mode} mode"

    def _fsd_button(self, down):
        now = time.perf_counter()
        if down and not self._fsd_down and now - self._fsd_last > FSD_DEBOUNCE:
            self.fsd_presses += 1
            self._fsd_last = now
        self._fsd_down = down

    def _process(self, ax, ay, az, gx, gy, gz):
        now = time.perf_counter()
        with self._lock:
            if self._calibrating:
                self.status = "calibrating - hold the wheel centred and still"
                self._calib.append((ax, ay, az, gx, gy, gz))
                if len(self._calib) >= CALIBRATION_SAMPLES:
                    self._finish_calibration()
                self._last_t = now
                return

            dt = now - self._last_t if self._last_t else 0.01
            dt = min(dt, 0.05)
            self._last_t = now

            bx, by, bz = self._gyro_bias
            rate = _gyro_rate(self.mode, gx - bx, gy - by, gz - bz)
            acc = _accel_angle(self.mode, ax, ay, az)
            prev = self._filtered_abs
            gyro_est = prev + rate * dt
            # blend on the shortest arc
            self._filtered_abs = _wrap(gyro_est + (1 - GYRO_WEIGHT) * _wrap(acc - gyro_est))
            self.angle = _wrap(self._filtered_abs - self._ref)

    def _finish_calibration(self):
        n = len(self._calib)
        mean = [sum(s[i] for s in self._calib) / n for i in range(6)]
        ax, ay, az = mean[:3]
        if self._auto_mode:
            # upright -> wheel, flat -> tilt (default X)
            if abs(az) < math.hypot(ax, ay):
                self.mode = "wheel"
            else:
                self.mode = self.mode if self.mode.startswith("tilt") else "tilt-x"
        self._gyro_bias = tuple(mean[3:6])
        self._ref = _accel_angle(self.mode, ax, ay, az)
        self._filtered_abs = self._ref
        self.angle = 0.0
        self._calib = []
        self._calibrating = False
