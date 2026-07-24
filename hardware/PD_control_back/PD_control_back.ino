#include <Encoder.h>
#include <cstdio>
#include <cstring>
#include <math.h>

const float M1GEAR = 9.68;
const float M2GEAR = 9.68;
const int TICKS_PER_REV = 48;

// Motor 1
const int M1INA = 2;
const int M1INB = 4;
const int M1PWM = 6;
const int M1EN = 9;
Encoder enc1(13, 14);

// Motor 2
const int M2INA = 7;
const int M2INB = 8;
const int M2PWM = 10;
const int M2EN = 12;
Encoder enc2(15, 16);

// ==========================================
// 2. HARDWARE CALIBRATION
// ==========================================
bool MOTOR1_REVERSED = false;
bool ENCODER1_REVERSED = false;

bool MOTOR2_REVERSED = true;
bool ENCODER2_REVERSED = true;

// ==========================================
// 3. CONTROL SETTINGS
// ==========================================
// PD gains
double Kp1 = 1024.0;
double Kd1 = 10.24;
double Kp2 = 1024.0;
double Kd2 = 102.4;

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

float imu_qr = 1.0, imu_qi = 0.0, imu_qj = 0.0, imu_qk = 0.0;
float acc_mag = 0.0;

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
}

// ==========================================
// 7. MAIN LOOP
// ==========================================
void loop() {
  handleSerialInput();

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
// 9. SERIAL INPUT (non-blocking)
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
    
  } else if (strcmp(line, "RESET_I2C") == 0) {
    
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
// 10. TELEMETRY ENCODERS
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