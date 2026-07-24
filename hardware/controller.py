import os
from scipy.spatial.transform import Rotation as R
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Edge / Pi: ONNX probes GPU on load; keep stderr quiet (4 = fatal in typical ORT builds).
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "4")

# Some Pi/conda setups set LD_PRELOAD to a missing OpenBLAS path; drop broken entries
# before NumPy so this process and ctypes-loaded libs don't trip over them.
def _strip_broken_ld_preload():
    p = os.environ.get("LD_PRELOAD", "")
    if not p:
        return
    sep = ":" if ":" in p else " "
    kept = []
    for x in p.split(sep):
        x = x.strip()
        if not x:
            continue
        if x.startswith("-") or x.startswith(" "):
            kept.append(x)
            continue
        if ("/" in x or x.endswith(".so") or ".so." in x) and not os.path.isfile(x):
            continue
        kept.append(x)
    if kept:
        os.environ["LD_PRELOAD"] = sep.join(kept)
    else:
        os.environ.pop("LD_PRELOAD", None)


_strip_broken_ld_preload()

import serial
import serial.tools.list_ports
import termios
import time
import math
import io
import argparse
import numpy as np
import csv
import threading
import cat_env.env_util as util
import socket

# --- CONFIGURATION ---
SN_FRONT = "18451300" 
SN_BACK  = "18452630"

BAUD_RATE = 115200
LOOP_HZ = 50

CONTROL_DURATION = 0.7

# Bound serial draining per tick (~50Hz telemetry per board); unbounded reads starve the Python loop.
SERIAL_DRAIN_MAX_LINES = 32

# Teleplot
# UDP_IP = "127.0.0.1" 
# UDP_PORT = 47269
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def _open_teensy_serial(port_path: str) -> serial.Serial:
    # write_timeout=0: do not block indefinitely if USB TX is wedged (pair with non-blocking Teensy TX).
    return serial.Serial(
        port_path,
        BAUD_RATE,
        timeout=0.005,
        write_timeout=0,
    )

debug_action = [0.0, 0.0, 0.0, 0.0]

class TeensyInterface:
    def __init__(self, port_path, name):
        self.port_path = port_path
        self.name = name
        self._rx_remainder = b""
        self.ser = _open_teensy_serial(port_path)

        self.quat = [1.0, 0.0, 0.0, 0.0]
        self.m1_rad = 0.0
        self.m2_rad = 0.0
        self.acc_mag = 0.0

    def reopen_serial(self):
        try:
            self.ser.close()
        except (OSError, AttributeError, ValueError):
            pass
        time.sleep(0.2)
        self.ser = _open_teensy_serial(self.port_path)
        self._rx_remainder = b""
        time.sleep(0.05)

    def _is_serial_gone(self, err):
        en = getattr(err, "errno", None)
        if en is None and getattr(err, "args", None):
            a0 = err.args[0]
            if isinstance(a0, OSError):
                en = a0.errno
        # EIO, EBADF, ENODEV, EPIPE — typical when USB CDC drops or device resets.
        return en in (5, 9, 19, 32)

    def flush_input_safe(self):
        """tcflush can raise EIO if USB CDC reset/re-enumerated; reopen and retry once."""
        try:
            self.ser.reset_input_buffer()
        except (termios.error, OSError) as e:
            print(f"{self.name}: input flush failed ({e!r}); reopening serial…")
            self.reopen_serial()
            try:
                self.ser.reset_input_buffer()
            except (termios.error, OSError):
                pass
        self._rx_remainder = b""
        self.quat = [1.0, 0.0, 0.0, 0.0] 
        self.m1_rad = 0.0
        self.m2_rad = 0.0
        self.acc_mag = 9.8

    def reset_encoders(self):
        """Sends the command to zero out the hardware encoder counts."""
        try:
            self.ser.write(b"RESET\n")
            print(f"{self.name}: Reset encoders.")
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                self.reopen_serial()
    
    def reset_IMU(self):
        try:
            self.ser.write(b"RESET_IMU\n")
            print(f"{self.name}: Reset IMU.")
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                self.reopen_serial()
    
    # def reset_I2C(self):
    #     try:
    #         self.ser.write(b"RESET_I2C\n")
    #         print(f"{self.name}: Reset I2C.")
    #     except (OSError, serial.SerialException) as e:
    #         if self._is_serial_gone(e):
    #             self.reopen_serial()

    def reboot_teensy(self):
        try:
            self.ser.write(b"REBOOT\n")
            print(f"{self.name}: Reboot.")
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                self.reopen_serial()

    def stop_all_motors(self):
        try:
            self.ser.write(b"STOP\n")
            print(f"{self.name}: Stopping motors.")
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                self.reopen_serial()

    def start_all_motors(self):
        try:
            self.ser.write(b"START\n")
            print(f"{self.name}: Starting motors.")
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                self.reopen_serial()

    def update_sensor_data(self):
        """Reads buffered lines to get the freshest IMU and encoder data."""
        try:
            waiting = self.ser.in_waiting
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                print(f"{self.name}: serial I/O in update_sensor_data; reopening…")
                self.reopen_serial()
            return

        if waiting <= 0:
            # print(f"waiting {self.name}")
            return

        max_bytes = min(waiting, 4096)
        try:
            raw = self.ser.read(max_bytes)
        except (OSError, serial.SerialException) as e:
            # print(f"serialexception")
            if self._is_serial_gone(e):
                self.reopen_serial()
            return

        if not raw:
            # print(f"raw")
            return

        self._rx_remainder += raw
        if b"\n" not in self._rx_remainder:
            # print(f"remainder")
            if len(self._rx_remainder) > 8192:
                self._rx_remainder = self._rx_remainder[-4096:]
            return

        chunks = self._rx_remainder.split(b"\n")
        self._rx_remainder = chunks[-1]
        complete = chunks[:-1][-SERIAL_DRAIN_MAX_LINES:]
        latest_line = None
        for c in complete:
            s = c.decode("utf-8", errors="ignore").strip()
            if s:
                latest_line = s
                
        if latest_line:
            parts = latest_line.split(',')
            if len(parts) == 7:
                try:
                    self.quat = [float(x) for x in parts[:4]]
                    self.quat = self.align_imu_quaternions(np.array([self.quat]), self.name)
                    self.m1_rad = float(parts[4])
                    self.m2_rad = float(parts[5])
                    self.acc_mag = float(parts[6])
                except ValueError:
                    pass

    def set_motors(self, m1, m2):
        """Sends target motor angles (2dp) to Teensy: 'm1,m2\\n'."""
        cmd = f"{m1:.2f},{m2:.2f}\n"
        try:
            self.ser.write(cmd.encode('utf-8'))
        except (OSError, serial.SerialException) as e:
            if self._is_serial_gone(e):
                print(f"{self.name}: serial I/O error in set_motors; reopening…")
                self.reopen_serial()
    
    def align_imu_quaternions(self, quats_wxyz, imu_type):
        r_raw = R.from_quat(quats_wxyz, scalar_first=True)
        
        if imu_type == 'Front':
            r_align = R.from_euler('xyz', [0, 0, 90], degrees=True)
            
        elif imu_type == 'Back':
            r_align = R.from_euler('xyz', [180, 0, -90], degrees=True)
            
        r_global = r_align.inv() * r_raw * r_align
        
        aligned_wxyz = r_global.as_quat(scalar_first=True)
        return aligned_wxyz.squeeze(0)

def get_port_by_sn(serial_number):
    for port in serial.tools.list_ports.comports():
        if port.serial_number == serial_number:
            return port.device
    return None

def keyboard_input_thread():
    global debug_action
    while True:
        try:
            user_in = input().split()
            if len(user_in) == 2:
                motor_num = int(user_in[0])
                target_angle = float(user_in[1])
                debug_action[motor_num - 1] = target_angle
                print(f"Updated Motor {motor_num} to {target_angle}")
        except Exception as e:
            print(f"Invalid input. Try again. ({e})")

def main():
    parser = argparse.ArgumentParser(description="Teensy Control Loop with Telemetry Logging")
    parser.add_argument(
        '--debug',
        action='store_true'
    )
    parser.add_argument(
        '--motoroff',
        action='store_true'
    )
    args = parser.parse_args()

    path_front = get_port_by_sn(SN_FRONT)
    path_back = get_port_by_sn(SN_BACK)
    
    if not path_front or not path_back:
        print(f"Error finding boards! Front: {path_front}, Back: {path_back}")
        return

    print("Connecting to boards...")
    front = TeensyInterface(path_front, "Front")
    back = TeensyInterface(path_back, "Back")
    time.sleep(1) # Wait for serial connection to establish

    print("Rebooting Teensy")
    front.reboot_teensy()
    front.ser.close()
    time.sleep(2)
    back.reboot_teensy()
    back.ser.close()
    time.sleep(2)

    path_front = get_port_by_sn(SN_FRONT)
    path_back = get_port_by_sn(SN_BACK)
    front = TeensyInterface(path_front, "Front")
    back = TeensyInterface(path_back, "Back")
    print(f"{path_front}, {path_back}")
    time.sleep(1) # Wait for serial connection to establish

    # Stop all motors
    front.stop_all_motors()
    back.stop_all_motors()

    # --- ZERO THE ENCODERS ---
    print("Zeroing motor encoders...")
    front.reset_encoders()
    back.reset_encoders()
    time.sleep(0.5)  # Let Teensy drain RESET; CDC can return EIO on flush if too early

    print("Zeroing IMU quaternions...")
    front.reset_IMU()
    back.reset_IMU()
    time.sleep(0.5)
    
    # Flush any stale data that was transmitted before the reset happened
    front.flush_input_safe()
    back.flush_input_safe()
    time.sleep(0.5)
    
    input_name = None
    if not args.debug:
        print("Loading ONNX Model...")
        import onnxruntime as ort
        try:
            ort.set_default_logger_severity(3)
        except (AttributeError, TypeError):
            pass # FIXED: Changed 'parse_args' to 'pass'
        
        ort_session = ort.InferenceSession("cat_controller.onnx")
        # Fetch the exact input name defined during your PyTorch export
        input_name = ort_session.get_inputs()[0].name 

    if not args.motoroff:
        front.start_all_motors()
        back.start_all_motors()
    print("Press enter to start.")
    input()

    loop_period = 1.0 / LOOP_HZ
    start_time = time.time()
    log = []
    
    # FIXED: Initialize drop_started so debug mode doesn't crash
    drop_started = float('inf') 

    if args.debug:
        print("Debug mode initiated: enter '[motor num] [target angle]'")
        threading.Thread(target=keyboard_input_thread, daemon=True).start()
    else:
        print("Waiting for drop..")
        while(True):
            loop_start = time.time()
            front.update_sensor_data()
            back.update_sensor_data()
            # print(front.acc_mag)
            if front.acc_mag < 3.5:
                print("Drop")
                drop_started = time.time()
                break

            r_front = R.from_quat(front.quat, scalar_first=True)
            print(r_front.as_euler('xyz', degrees=True))
            
            elapsed = time.time() - loop_start
            if elapsed < loop_period:
                time.sleep(loop_period - elapsed)
            else:
                print(f"WARNING: Loop missed deadline! Took {elapsed:.4f}s")

    print(f"Starting loop at {LOOP_HZ}Hz. Press Ctrl+C to quit.")

    try:
        while(True): 
            loop_start = time.time()
            t = loop_start - start_time

            front.update_sensor_data()
            back.update_sensor_data()
            
            action = [0, 0, 0, 0]
            # To rotation matrices
            front_rot = util.to_rotation_matrix(front.quat)
            back_rot = util.to_rotation_matrix(back.quat)

            if args.debug:
                action = list(debug_action)
            else:
                # Ensure the joint array matches the exact order used in simulation
                joints = np.array([front.m1_rad, front.m2_rad, back.m2_rad, back.m1_rad])
                
                # Stack to create a flat 1D array
                obs = np.hstack((front_rot, joints)) #FIXME
                
                # Cast to float32 and add the batch dimension (1, N) for ONNX
                obs_tensor = np.array(obs, dtype=np.float32).reshape(1, -1)
                
                # Run inference
                # [0] gets the first output node, [0] unpacks the batch dimension
                raw_action = ort_session.run(None, {input_name: obs_tensor})[0][0]
                
                roll = util.map_value(float(raw_action[0]), -1, 1, -np.pi*2, np.pi*2) # roll
                pitch = util.map_value(float(raw_action[1]), -1, 1, -np.pi/2, np.pi/2) # pitch
                tail = util.map_value(float(raw_action[2]), -1, 1, -np.pi/2, np.pi/2) # tail

                action = [roll, pitch, tail, -roll]
                # print(action)

            front.set_motors(action[0], action[1])
            back.set_motors(action[2], action[3])

            log.append([
                round(t, 4),
                front.quat[0], front.quat[1], front.quat[2], front.quat[3],
                front.m1_rad, front.m2_rad,
                front.acc_mag,
                action[0], action[1],
                back.quat[0], back.quat[1], back.quat[2], back.quat[3],
                back.m1_rad, back.m2_rad,
                back.acc_mag,
                action[2], action[3],
            ])

            # Teleplot format: "VariableName:Value\n"
            # teleplot_data = [
            #     f"front_w:{front.quat[0]}", f"front_x:{front.quat[1]}", f"front_y:{front.quat[2]}", f"front_z:{front.quat[3]}",
            #     f"front_m1:{front.m1_rad}", f"front_m2:{front.m2_rad}",
            #     f"front_acc:{front.acc_mag}",
            #     f"action_0:{action[0]}", f"action_1:{action[1]}",
            #     f"back_w:{back.quat[0]}", f"back_x:{back.quat[1]}", f"back_y:{back.quat[2]}", f"back_z:{back.quat[3]}",
            #     f"back_m1:{back.m1_rad}", f"back_m2:{back.m2_rad}",
            #     f"back_acc:{back.acc_mag}",
            #     f"action_2:{action[2]}", f"action_3:{action[3]}"
            # ]

            # Join with newlines and add the final newline
            # teleplot_string = "\n".join(teleplot_data) + "\n"

            # Send the data via your UDP socket
            # sock.sendto(teleplot_string.encode(), (UDP_IP, UDP_PORT))

            # Enforce timing
            elapsed = time.time() - loop_start
            if elapsed < loop_period:
                time.sleep(loop_period - elapsed)
            else:
                print(f"WARNING: Loop missed deadline! Took {elapsed:.4f}s")
                
            if time.time() - drop_started > CONTROL_DURATION:
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\nExiting...")
        front.stop_all_motors()
        back.stop_all_motors()
        if log:
            filename = f"telemetry/telemetry_{int(time.time())}.csv"
            print(f"Saving {len(log)} records to {filename}...")
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time", "F_Q0", "F_Q1", "F_Q2", "F_Q3", "F_M1", "F_M2", "F_ACC", "Cmd_F1", "Cmd_F2", 
                                 "B_Q0", "B_Q1", "B_Q2", "B_Q3", "B_M1", "B_M2", "B_ACC", "Cmd_B1", "Cmd_B2"])
                writer.writerows(log)
            print("Done.")

if __name__ == "__main__":
    main()