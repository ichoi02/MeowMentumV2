#include <Encoder.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <cstdio>
#include <cstring>
#include <math.h>

const float M1GEAR = 9.68;
const float M2GEAR = 9.68;
const int TICKS_PER_REV = 48;

// Motor 1
const int M1INA = 2;
const int M1INB = 4;
const int M1PWM = 9;
const int M1EN = 6;
Encoder enc1(14, 15);

// Motor 2
const int M2INA = 7;
const int M2INB = 8;
const int M2PWM = 12;
const int M2EN = 10;
Encoder enc2(17, 16);

Adafruit_BNO08x bno08x;
sh2_SensorValue_t sensorValue;

// ==========================================
// 2. HARDWARE CALIBRATION
// ==========================================
bool MOTOR1_REVERSED = true;
bool ENCODER1_REVERSED = true;

bool MOTOR2_REVERSED = false;
bool ENCODER2_REVERSED = true;

// ==========================================
// 3. CONTROL SETTINGS
// ==========================================
// PD gains
double Kp1 = 2048.0;
double Kd1 = 204.8;
double Kp2 = 20480.0;
double Kd2 = 204.80;

// If the error is within this many ticks, motor stops
float deadband = 0.03;
const int minPWM = 100;
const int PWM_MAX = 1023;

// Fixed control frequency
const float CONTROL_HZ = 1000.0f;               // 1 kHz
const uint32_t CONTROL_PERIOD_US = 1000000UL / CONTROL_HZ;
const uint32_t PRINT_PERIOD_MS = 20;
static char s_telemBuf[128];

// ==========================================
// 4. STATE VARIABLES
// ==========================================
float targetPos1 = 0;
float targetPos2 = 0;
float lastErr1 = 0;
float lastErr2 = 0;

uint32_t lastControlMicros = 0;
uint32_t lastPrintMillis = 0;

// global IMU
float imu_qr = 1.0, imu_qi = 0.0, imu_qj = 0.0, imu_qk = 0.0;
float acc_mag = 0.0;

// Watchdog variables for I2C freeze recovery
static uint32_t lastGameRotEventMs = 0;
static const uint32_t IMU_STALE_MS = 1500;
static const uint32_t IMU_RECOVER_COOLDOWN_MS = 2500;

// ==========================================
// 5. I2C RECOVERY FUNCTIONS
// ==========================================

void clearI2CBus() {
  Wire.end();
  
  // Take manual control of the Teensy 4.0 default I2C pins
  pinMode(18, INPUT_PULLUP); // SDA
  pinMode(19, OUTPUT);       // SCL
  digitalWrite(19, HIGH);
  delay(1);
  
  // Pulse SCL until the sensor releases the SDA line (max 20 pulses)
  for (int i = 0; i < 20; i++) {
    if (digitalRead(18) == HIGH) {
      break; 
    }
    digitalWrite(19, LOW);
    delayMicroseconds(10);
    digitalWrite(19, HIGH);
    delayMicroseconds(10);
  }
  
  // Release SCL to float HIGH
  pinMode(19, INPUT_PULLUP); 
  delay(10);
  
  Wire.begin();
  
  // Force a hardware timeout so the Wire library CANNOT freeze
  Wire.setTimeout(10); 
  Wire.setClock(50000); 
}

static bool bno08x_begin_and_enable() {
  bool ok = false;
  
  for (int attempt = 0; attempt < 5 && !ok; attempt++) {
    if (bno08x.begin_I2C(0x4A, &Wire)) ok = true;
    else delay(50);
  }
  
  if (!ok) {
    return false;
  }
  
  bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, 10000);
  bno08x.enableReport(SH2_ACCELEROMETER, 10000);
  return true;
}

static void recover_bno08x_from_stall() {
  clearI2CBus();
  delay(50);
  
  if (bno08x_begin_and_enable()) {
    lastGameRotEventMs = millis();
  } 
}

// ==========================================
// 6. SETUP
// ==========================================
void setup() {
  Serial.begin(115200);

  pinMode(M1INA, OUTPUT);
  pinMode(M1INB, OUTPUT);
  pinMode(M1PWM, OUTPUT);
  pinMode(M1EN, OUTPUT);

  pinMode(M2INA, OUTPUT);
  pinMode(M2INB, OUTPUT);
  pinMode(M2PWM, OUTPUT);
  pinMode(M2EN, OUTPUT);

  digitalWrite(M1EN, LOW);
  digitalWrite(M2EN, LOW);

  analogWriteFrequency(M1PWM, 20000);
  analogWriteFrequency(M2PWM, 20000);
  analogWriteRes(10);

  stopMotor1();
  stopMotor2();

  targetPos1 = readEncoder1();
  targetPos2 = readEncoder2();

  lastControlMicros = micros();
  lastPrintMillis = millis();

  // IMU Setup (retry: transient I2C glitches on power-up)
  Wire.begin();
  Wire.setTimeout(10); 
  Wire.setClock(50000); // 50kHz for long wire stability
  
  if (!bno08x_begin_and_enable()) {
    while (1) { delay(100); } // Hang if totally dead on boot
  }

  lastGameRotEventMs = millis();
}

// ==========================================
// 7. MAIN LOOP
// ==========================================
void loop() {
  handleSerialInput();

  // 1. Check for silent internal sensor resets
  if (bno08x.wasReset()) {
    bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, 10000);
    bno08x.enableReport(SH2_ACCELEROMETER, 10000);
    lastGameRotEventMs = millis();
  }

  int imu_budget = 5; 
  while ((imu_budget-- > 0) && bno08x.getSensorEvent(&sensorValue)) {
    switch (sensorValue.sensorId) {
      case SH2_GAME_ROTATION_VECTOR:
        imu_qr = sensorValue.un.gameRotationVector.real;
        imu_qi = sensorValue.un.gameRotationVector.i;
        imu_qj = sensorValue.un.gameRotationVector.j;
        imu_qk = sensorValue.un.gameRotationVector.k;
        lastGameRotEventMs = millis(); 
        break;
        
      case SH2_ACCELEROMETER:
        float ax = sensorValue.un.accelerometer.x;
        float ay = sensorValue.un.accelerometer.y;
        float az = sensorValue.un.accelerometer.z;
        acc_mag = sqrt((ax * ax) + (ay * ay) + (az * az));
        lastGameRotEventMs = millis(); 
        break;
    }
  }

  // 4. Run Controller
  uint32_t nowMicros = micros();
  if ((uint32_t)(nowMicros - lastControlMicros) >= CONTROL_PERIOD_US) {
    double dt = (nowMicros - lastControlMicros) / 1000000.0;
    lastControlMicros = nowMicros;
    runController(dt);
  }

  // 5. Print Telemetry
  uint32_t nowMillis = millis();
  if ((uint32_t)(nowMillis - lastPrintMillis) >= PRINT_PERIOD_MS) {
    lastPrintMillis = nowMillis;
    
    float angle1 = readEncoder1();
    float angle2 = readEncoder2();

    int n = snprintf(s_telemBuf, sizeof(s_telemBuf),
                     "%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f\n",
                     imu_qr, imu_qi, imu_qj, imu_qk, angle1, angle2, acc_mag);
    if (n > 0 && n < (int)sizeof(s_telemBuf) &&
        Serial.availableForWrite() >= n) {
      Serial.write((const uint8_t *)s_telemBuf, (size_t)n);
    }
  }
}

// ==========================================
// 8. CONTROL LOOP
// ==========================================
void runController(double dt) {
  if (dt <= 0.0) {
    dt = 0.001; // fallback
  }

  float currentPos1 = readEncoder1();
  float currentPos2 = readEncoder2();

  // ----- Motor 1 -----
  float err1 = targetPos1 - currentPos1;
  if (abs(err1) <= deadband) {
    stopMotor1();
  } else {
    double deriv1 = (err1 - lastErr1) / dt;
    double cmd1 = (Kp1 * err1) + (Kd1 * deriv1);
    if (MOTOR1_REVERSED) cmd1 = -cmd1;
    driveMotor1(cmd1);
  }

  // ----- Motor 2 -----
  float err2 = targetPos2 - currentPos2;
  if (abs(err2) <= deadband) {
    stopMotor2();
  } else {
    double deriv2 = (err2 - lastErr2) / dt;
    double cmd2 = (Kp2 * err2) + (Kd2 * deriv2);
    if (MOTOR2_REVERSED) cmd2 = -cmd2;
    driveMotor2(cmd2);
  }

  lastErr1 = err1;
  lastErr2 = err2;
}

// ==========================================
// 9. SERIAL INPUT (non-blocking: no readStringUntil timeout stalls)
// ==========================================
static void processSerialLine(char *line) {
  while (*line == ' ' || *line == '\t') line++;
  size_t len = strlen(line);
  while (len > 0 && (line[len - 1] == ' ' || line[len - 1] == '\t')) {
    line[--len] = '\0';
  }
  if (len == 0) return;

  if (strcmp(line, "RESET") == 0) {
    enc1.write(0);
    enc2.write(0);
  } else if (strcmp(line, "START") == 0) {
    digitalWrite(M1EN, HIGH);
    digitalWrite(M2EN, HIGH);
  } else if (strcmp(line, "STOP") == 0) {
    digitalWrite(M1EN, LOW);
    digitalWrite(M2EN, LOW);
  } else if (strcmp(line, "REBOOT") == 0) {
    const char msg[] = "Rebooting...\n";
    if ((int)Serial.availableForWrite() >= (int)sizeof(msg) - 1) {
      Serial.write((const uint8_t *)msg, sizeof(msg) - 1);
    }
    delay(100);
    SCB_AIRCR = 0x05FA0004;
  } else if (strcmp(line, "RESET_IMU") == 0) {
    sh2_setTareNow(SH2_TARE_X | SH2_TARE_Y | SH2_TARE_Z, SH2_TARE_BASIS_GAMING_ROTATION_VECTOR);
  } else if (strcmp(line, "RESET_I2C") == 0) {
    recover_bno08x_from_stall();
  } else {
    float a, b;
    if (sscanf(line, "%f,%f", &a, &b) == 2) {
      targetPos1 = a;
      targetPos2 = b;
    }
  }
}

void handleSerialInput() {
  static char lineBuf[80];
  static size_t lineLen = 0;
  int budget = 64;
  while (Serial.available() && budget-- > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineLen > 0) {
        lineBuf[lineLen] = '\0';
        processSerialLine(lineBuf);
        lineLen = 0;
      }
      continue;
    }
    if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    } else {
      lineLen = 0;
    }
  }
}

// ==========================================
// 10. TELEMETRY
// ==========================================
float readEncoder1() {
  long pos = enc1.read();
  if (ENCODER1_REVERSED) pos = -pos;
  return (float)pos/TICKS_PER_REV/M1GEAR*2*PI;
}

float readEncoder2() {
  long pos = enc2.read();
  if (ENCODER2_REVERSED) pos = -pos;
  return (float)pos/TICKS_PER_REV/M2GEAR*2*PI;
}

// ==========================================
// 11. MOTOR DRIVE FUNCTIONS
// ==========================================
void driveMotor1(double speed) {
  int pwmVal = constrain((int)abs(speed), 0, PWM_MAX);
  if (pwmVal > 0 && pwmVal < minPWM) pwmVal = minPWM;

  if (speed > 0) {
    digitalWrite(M1INA, HIGH);
    digitalWrite(M1INB, LOW);
  } else {
    digitalWrite(M1INA, LOW);
    digitalWrite(M1INB, HIGH);
  }
  analogWrite(M1PWM, pwmVal);
}

void stopMotor1() {
  digitalWrite(M1INA, LOW);
  digitalWrite(M1INB, LOW);
  analogWrite(M1PWM, 0);
}

void driveMotor2(double speed) {
  int pwmVal = constrain((int)abs(speed), 0, PWM_MAX);
  if (pwmVal > 0 && pwmVal < minPWM) pwmVal = minPWM;

  if (speed > 0) {
    digitalWrite(M2INA, HIGH);
    digitalWrite(M2INB, LOW);
  } else {
    digitalWrite(M2INA, LOW);
    digitalWrite(M2INB, HIGH);
  }
  analogWrite(M2PWM, pwmVal);
}

void stopMotor2() {
  digitalWrite(M2INA, LOW);
  digitalWrite(M2INB, LOW);
  analogWrite(M2PWM, 0);
}