# CARLA-but-worse

A 3D driving sim on your laptop, steered by an Arduino + MPU6050 on a breadboard that you hold and turn like a steering wheel.

```
 [breadboard: Arduino + MPU6050]  --USB serial, 100 Hz-->  [laptop: Python + OpenGL game]
   raw accel + gyro + 2 buttons                              filter -> angle -> car physics -> 3D
```

## Hardware

- Arduino Uno or Nano (anything with `Wire` works)
- MPU6050 breakout (GY-521)
- Optional: 3 push buttons for gas, brake and FSD
- Jumper wires and a USB cable

### Wiring (Uno / Nano)

| MPU6050 | Arduino |
|---------|---------|
| VCC     | 5V (GY-521 has a regulator; use 3.3V for a bare module) |
| GND     | GND     |
| SDA     | A4      |
| SCL     | A5      |

Optional buttons: gas **D2 → GND**, brake **D3 → GND**, FSD **D4 → GND**. Internal pull-ups are used, so no resistors are needed.

On a Leonardo or Micro, SDA/SCL are D2/D3, so move the buttons to other pins and change `GAS_PIN`/`BRAKE_PIN`/`FSD_PIN` in the sketch.

Keep the MPU6050 firmly seated. A wobbly sensor on a breadboard reads as steering input.

## Flash the Arduino

Open `arduino/steering_wheel/steering_wheel.ino` in the Arduino IDE, pick your board and port, and upload. No libraries are needed.

To check it, open the Serial Monitor at **115200** baud. You should see lines like `S,0.012,0.998,0.051,-0.42,0.88,0.10,0,0,0`. If you see `E,MPU6050 not found`, check SDA/SCL/VCC.

Close the Serial Monitor before starting the game, because only one program can hold the port.

## Run the game

```bash
/usr/bin/python3 -m venv .venv          # use the *system* python, see note below
.venv/bin/pip install -r requirements.txt
.venv/bin/python drive.py               # auto-detects /dev/ttyACM* or /dev/ttyUSB*
```

Useful flags: `--port /dev/ttyUSB0`, `--fullscreen`, `--invert`, `--range 120`, `--no-serial`.

On Linux your user needs to be in the `dialout` group to open the serial port.

> **Note for miniconda users:** a venv built from conda's Python fails with
> `Couldn't find matching GLX visual`. Conda ships an older `libstdc++` that the
> system Mesa GPU driver can't load. Build the venv from `/usr/bin/python3` instead.

The game works without the board too: steer with A/D.

## Holding the "wheel"

Calibration runs automatically when the board connects. Hold it **centred and still** for about a second. Press **C** any time to re-centre.

The game works out how you're holding it:

- **Upright, chip facing you** → *wheel mode*. Turn it like a steering wheel. ±90° is full lock.
- **Flat like a tray** → *tilt mode*. Tilt left/right. ±45° is full lock.

If the steering is backwards, press **I**. If tilting does nothing in flat mode, press **M** to switch which axis you tilt around. Use **[** and **]** to change how far you have to turn for full lock.

## Full self-driving

Press the **D4 button** (or **F**) to hand control to the autopilot. It follows the right-hand lane, slows for corners, and brakes for anything the lidar sees in its path. Press the brake, steer with A/D or pull the handbrake to take back control. Press the button again to switch it off.

While FSD is on you can see what it sees:

- **Lidar**: a 12-channel spinning lidar (240 steps per turn, 15 Hz, 80 m range) drawn as a point cloud in the world. Obstacles are coloured by distance (red is near). Ground returns are brighter on paint and kerbs, like real lidar intensity. The panel on the right shows a top-down view.
- **Camera**: an inset front-camera feed with object detection boxes (tree, building, cone and a confidence score), detected lane lines (green/yellow) and the planned path (blue). Objects the lidar says are hidden behind something closer are not boxed.

## Controls

| Key | Action |
|-----|--------|
| W / ↑ | gas (or the D2 button) |
| S / ↓ | brake, then hold to reverse (or the D3 button) |
| Space | handbrake |
| A / D | keyboard steering (overrides the wheel while held) |
| R | reset onto the track |
| V | camera: chase / hood / high |
| C | re-centre wheel |
| M | cycle mount mode (wheel / tilt-x / tilt-y) |
| I | invert steering |
| [ / ] | wheel range −/+ 10° |
| F | toggle FSD (or the D4 button) |
| H | hide help |
| Esc | quit |

## How it works

- **`arduino/steering_wheel/`**: reads the MPU6050 over I2C directly (no library) and prints accel (g), gyro (°/s) and the gas/brake/FSD button states at 100 Hz. The sketch does no maths, so tuning never requires re-flashing.
- **`sim/wheel_input.py`**: a background thread reads the serial port. It averages about 0.6 s of samples to find "centre" and the gyro bias, then runs a complementary filter: 96% integrated gyro (smooth) and 4% accelerometer angle (no drift).
- **`sim/car.py`**: kinematic bicycle model with speed-sensitive steering, grass slowdown and tree/building collisions.
- **`sim/track.py`**: a Catmull-Rom spline loop with scenery scattered off the road.
- **`sim/render.py`**: fixed-function OpenGL. The static world is compiled into one display list; the file also draws the car and the HUD.
- **`sim/lidar.py`**: vectorised numpy raycasts against trees, cones, buildings and the ground.
- **`sim/vision.py`**: projects scenery into the front camera for bounding boxes, plus lane-line detection.
- **`sim/autopilot.py`**: pure-pursuit steering. Its target speed is set by the corners ahead and by lidar returns in the predicted path.
- **`sim/main.py`**: the game loop, lap timer and HUD layout.
