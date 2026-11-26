// MPU6050 steering wheel for CARLA-but-worse.
// Streams raw accel + gyro + buttons at 100 Hz; all the maths is in Python.
//
// Wiring (Uno / Nano): VCC->5V, GND->GND, SDA->A4, SCL->A5
// Buttons to GND: gas D2, brake D3, FSD D4
//
// Output (115200 baud):
//   S,ax,ay,az,gx,gy,gz,gas,brake,fsd   accel g, gyro deg/s, buttons 0/1
//   I,<msg> info   E,<msg> error

#include <Wire.h>

const uint8_t MPU_ADDR = 0x68;   // 0x69 if AD0 is high
const uint8_t GAS_PIN = 2;
const uint8_t BRAKE_PIN = 3;
const uint8_t FSD_PIN = 4;
const unsigned long SAMPLE_US = 10000;  // 100 Hz

const float ACCEL_LSB_PER_G = 16384.0;  // +-2 g
const float GYRO_LSB_PER_DPS = 65.5;    // +-500 deg/s

bool writeReg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool initMpu() {
  if (!writeReg(0x6B, 0x01)) return false;  // wake, gyro clock
  delay(50);
  writeReg(0x1A, 0x03);  // low-pass ~44 Hz
  writeReg(0x1B, 0x08);  // gyro +-500 deg/s
  writeReg(0x1C, 0x00);  // accel +-2 g

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x75);  // WHO_AM_I
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, (uint8_t)1);
  if (Wire.available()) {
    uint8_t who = Wire.read();
    Serial.print(F("I,WHO_AM_I=0x"));
    Serial.println(who, HEX);  // clones may report 0x70/0x72/0x98
  }
  return true;
}

int16_t read16() {
  int16_t hi = Wire.read();
  return (hi << 8) | Wire.read();
}

int pressed(uint8_t pin) {
  return digitalRead(pin) == LOW ? 1 : 0;
}

void setup() {
  Serial.begin(115200);
  pinMode(GAS_PIN, INPUT_PULLUP);
  pinMode(BRAKE_PIN, INPUT_PULLUP);
  pinMode(FSD_PIN, INPUT_PULLUP);
  Wire.begin();
  Wire.setClock(400000);
#if defined(WIRE_HAS_TIMEOUT)
  Wire.setWireTimeout(3000, true);
#endif

  while (!initMpu()) {
    Serial.println(F("E,MPU6050 not found - check SDA/SCL/VCC wiring"));
    delay(1000);
  }
  Serial.println(F("I,READY"));
}

void loop() {
  static unsigned long next = micros();
  if ((long)(micros() - next) < 0) return;
  next += SAMPLE_US;

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);  // accel xyz, temp, gyro xyz
  if (Wire.endTransmission(false) != 0 || Wire.requestFrom(MPU_ADDR, (uint8_t)14) != 14) {
    Serial.println(F("E,MPU6050 read failed - loose wire?"));
    delay(200);
    initMpu();
    next = micros();
    return;
  }

  float ax = read16() / ACCEL_LSB_PER_G;
  float ay = read16() / ACCEL_LSB_PER_G;
  float az = read16() / ACCEL_LSB_PER_G;
  read16();  // temperature
  float gx = read16() / GYRO_LSB_PER_DPS;
  float gy = read16() / GYRO_LSB_PER_DPS;
  float gz = read16() / GYRO_LSB_PER_DPS;

  Serial.print('S');
  Serial.print(','); Serial.print(ax, 3);
  Serial.print(','); Serial.print(ay, 3);
  Serial.print(','); Serial.print(az, 3);
  Serial.print(','); Serial.print(gx, 2);
  Serial.print(','); Serial.print(gy, 2);
  Serial.print(','); Serial.print(gz, 2);
  Serial.print(','); Serial.print(pressed(GAS_PIN));
  Serial.print(','); Serial.print(pressed(BRAKE_PIN));
  Serial.print(','); Serial.println(pressed(FSD_PIN));
}
