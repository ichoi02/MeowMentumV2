#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <Encoder.h>

// --- BNO08x Setup ---
Adafruit_BNO08x bno08x;
sh2_SensorValue_t sensorValue;
const unsigned long PRINT_INTERVAL_MS = 20; // 50Hz
unsigned long lastPrint = 0;

// --- Hardware & Calibration Constants ---
const float M1GEAR = 9.68;
const float M2GEAR = 46.85; // Fixed: M2 gear ratio
const int TICKS_PER_REV = 48;

bool MOTOR1_REVERSED = true;
bool ENCODER1_REVERSED = true;
bool MOTOR2_REVERSED = false;
bool ENCODER2_REVERSED = false;

// Motor 1
const int M1INA = 2;
const int M1INB = 4;
const int M1PWM = 9;
const int M1EN = 6;
Encoder enc1(14, 15);

// Motor 2
const int M2INA = 7;
const int M2INB = 8;
const int M2PWM = 10;
const int M2EN = 12;
Encoder enc2(17, 20);

// PWM Settings
const int PWM_MAX = 1023;
const int minPWM = 100;

// Safety Watchdog
unsigned long lastCommandTime = 0;
const unsigned long TIMEOUT_MS = 500;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(5);

  // Motor 1 pins
  pinMode(M1INA, OUTPUT); pinMode(M1INB, OUTPUT);
  pinMode(M1PWM, OUTPUT); pinMode(M1EN, OUTPUT);

  // Motor 2 pins
  pinMode(M2INA, OUTPUT); pinMode(M2INB, OUTPUT);
  pinMode(M2PWM, OUTPUT); pinMode(M2EN, OUTPUT);

  // Enable drivers
  digitalWrite(M1EN, HIGH);
  digitalWrite(M2EN, HIGH);

  // Teensy 4.0 PWM setup
  analogWriteFrequency(M1PWM, 20000);
  analogWriteFrequency(M2PWM, 20000);
  analogWriteRes(10);

  stopMotor1();
  stopMotor2();

  // IMU Setup
  Wire.begin();
  Wire.setClock(400000);
  if (!bno08x.begin_I2C(0x4A, &Wire)) {
    while (1) { delay(100); }
  }
  if (!bno08x.enableReport(SH2_ROTATION_VECTOR, 10000)) {
    while (1) { delay(100); }
  }
}

void loop() {
  unsigned long currentMillis = millis();

  // 1. READ SERIAL COMMANDS
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    if (input == "RESET") {
      enc1.write(0);
      enc2.write(0);
    } 
    else {
      int commaIndex = input.indexOf(',');
      if (commaIndex > 0) {
        float m1_cmd = input.substring(0, commaIndex).toFloat();
        float m2_cmd = input.substring(commaIndex + 1).toFloat();
        
        driveMotor1(m1_cmd);
        driveMotor2(m2_cmd);
        lastCommandTime = currentMillis;
      }
    }
  }

  // 2. SAFETY WATCHDOG
  if (currentMillis - lastCommandTime > TIMEOUT_MS) {
    stopMotor1();
    stopMotor2();
  }

  // 3. READ AND SEND TELEMETRY
  if (currentMillis - lastPrint >= PRINT_INTERVAL_MS) {
    lastPrint = currentMillis;

    if (bno08x.getSensorEvent(&sensorValue)) {
      if (sensorValue.sensorId == SH2_ROTATION_VECTOR) {
        float qr = sensorValue.un.rotationVector.real;
        float qi = sensorValue.un.rotationVector.i;
        float qj = sensorValue.un.rotationVector.j;
        float qk = sensorValue.un.rotationVector.k;
        float acc = sensorValue.un.rotationVector.accuracy;

        float angle1 = readEncoder1();
        float angle2 = readEncoder2();

        Serial.print(qr, 6); Serial.print(",");
        Serial.print(qi, 6); Serial.print(",");
        Serial.print(qj, 6); Serial.print(",");
        Serial.print(qk, 6); Serial.print(",");
        Serial.print(acc, 6); Serial.print(",");
        Serial.print(angle1, 4); Serial.print(",");
        Serial.println(angle2, 4);
      }
    }
  }
}

// --- Encoder Helpers ---
float readEncoder1() {
  long pos = enc1.read();
  if (ENCODER1_REVERSED) pos = -pos;
  return (float)pos / (TICKS_PER_REV * M1GEAR) * 2.0 * PI;
}

float readEncoder2() {
  long pos = enc2.read();
  if (ENCODER2_REVERSED) pos = -pos;
  return (float)pos / (TICKS_PER_REV * M2GEAR) * 2.0 * PI; // Fixed bug here!
}

// --- Motor Control Helpers ---
void driveMotor1(double speed) {
  if (MOTOR1_REVERSED) speed = -speed;
  
  int pwmVal = constrain((int)abs(speed), 0, PWM_MAX);
  if (pwmVal > 0 && pwmVal < minPWM) pwmVal = minPWM;

  if (pwmVal == 0) {
    stopMotor1();
    return;
  }

  if (speed > 0) {
    digitalWrite(M1INA, HIGH); digitalWrite(M1INB, LOW);
  } else {
    digitalWrite(M1INA, LOW); digitalWrite(M1INB, HIGH);
  }
  analogWrite(M1PWM, pwmVal);
}

void stopMotor1() {
  digitalWrite(M1INA, LOW); digitalWrite(M1INB, LOW); analogWrite(M1PWM, 0);
}

void driveMotor2(double speed) {
  if (MOTOR2_REVERSED) speed = -speed;

  int pwmVal = constrain((int)abs(speed), 0, PWM_MAX);
  if (pwmVal > 0 && pwmVal < minPWM) pwmVal = minPWM;

  if (pwmVal == 0) {
    stopMotor2();
    return;
  }

  if (speed > 0) {
    digitalWrite(M2INA, HIGH); digitalWrite(M2INB, LOW);
  } else {
    digitalWrite(M2INA, LOW); digitalWrite(M2INB, HIGH);
  }
  analogWrite(M2PWM, pwmVal);
}

void stopMotor2() {
  digitalWrite(M2INA, LOW); digitalWrite(M2INB, LOW); analogWrite(M2PWM, 0);
}