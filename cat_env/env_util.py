import numpy as np
from scipy.spatial.transform import Rotation as R

# Teensy inner-loop nonlinearities (hardware/PD_control_{front,back}/*.ino).
# The board zeroes the motor inside `deadband` and floors any nonzero command to
# `minPWM`, so the sim PD must reproduce both -- otherwise training relies on
# fine torques the real actuator never produces.
MOTOR_DEADBAND = 0.03           # rad, firmware `deadband`
MOTOR_MIN_CMD = 100.0 / 1023.0  # firmware `minPWM` / `PWM_MAX`

# Encoder model. The boards do not measure joint angle -- they count quadrature
# ticks on the MOTOR shaft and divide by the gear ratio (readEncoder1/2 in
# hardware/PD_control_*.ino), so the position the PD sees is quantized, and the
# velocity it differentiates is quantized far more harshly still.
#
# This is a first-class part of the plant, not a detail: at 48 CPR the joint-side
# step is 13.5 mrad on roll -- 45% of the 30 mrad deadband -- and a single tick
# inside one 1 ms cycle reads as 13.5 rad/s. It is the reason the velocity filter
# exists and the reason kd cannot be raised freely. Modelling it here (rather than
# bounding kd with a side formula) means the ordinary step-response and oscillation
# metrics see the real effect and reject unusable gains on their own.
ENCODER_TICKS_PER_REV = 48      # quadrature counts per motor revolution
GEAR_RATIO = {"rot1": 9.68, "pitch": 34.014, "rot2": 9.68, "tail": 34.014}

def encoder_resolution(joint):
    """Smallest joint-angle change the encoder can report, in rad."""
    return 2.0 * np.pi / (ENCODER_TICKS_PER_REV * GEAR_RATIO[joint])

# Per-joint override of the deadband. The firmware keeps one `deadband` per board
# today, but it is applied per channel, so these may legitimately differ per joint
# if a sweep justifies it. Anything not listed falls back to MOTOR_DEADBAND.
JOINT_DEADBAND = {}

def joint_deadband(joint):
    return JOINT_DEADBAND.get(joint, MOTOR_DEADBAND)

def apply_motor_deadband(norm_torque, pos_err, deadband=None):
    """Mirror the Teensy deadband + minimum-PWM floor on a normalized torque."""
    if deadband is None:
        deadband = MOTOR_DEADBAND
    if abs(pos_err) <= deadband:
        return 0.0
    if 0.0 < abs(norm_torque) < MOTOR_MIN_CMD:
        return np.copysign(MOTOR_MIN_CMD, norm_torque)
    return norm_torque

# One-pole low-pass on the measured joint velocity, applied every 1 ms control
# cycle: v_f += alpha * (v - v_f). Mirrors the same filter in the Teensy firmware,
# where it is not optional -- the boards differentiate a 48 CPR encoder, which at
# the joint is 13.5 mrad (roll) / 3.8 mrad (pitch, tail) per tick, so a single tick
# in one 1 ms sample reads as a 13.5 rad/s velocity spike and saturates the D term
# outright. The filter exists to make that usable.
#
# It is mirrored here because the velocity the sim PD differentiates is quantized
# exactly as the real one is (see `resolution` below), so the filter is doing the
# same job on both sides -- and because the phase lag it adds is part of the
# effective damping, which gains tuned without it would not transfer through.
#
# 0.1 puts the averaging window (~2/alpha - 1 = 19 samples) at one 20 ms control
# period; smoothing over longer than the gap between commands buys nothing.
# MUST equal VEL_ALPHA in hardware/PD_control_{front,back}.ino.
VEL_FILTER_ALPHA = 0.1

class PDController:
    """The Teensy inner loop, step for step.

    With `resolution` set, this reproduces runController() exactly: quantize the
    encoder, difference the quantized position for velocity, low-pass it, then
    kp*err - kd*v. The quantization is what bounds kd -- differentiating a coarse
    encoder at 1 kHz turns one tick into a large apparent velocity, so too much kd
    makes the motor chatter on sensor noise alone rather than on real motion.
    """

    def __init__(self, kp: float, kd: float, resolution: float = 0.0,
                 vel_alpha: float = VEL_FILTER_ALPHA, dt: float = 0.001):
        self.kp = kp
        self.kd = kd
        self.resolution = resolution   # rad per encoder tick; 0 disables quantization
        self.vel_alpha = vel_alpha
        self.dt = dt
        self.reset()

    def _measure(self, current_pos):
        """What the encoder would report for this true joint angle."""
        if self.resolution <= 0.0:
            return current_pos
        return np.round(current_pos / self.resolution) * self.resolution

    def reset(self, current_pos=0.0):
        """Clear filter state and seed the position history. Once per episode.

        Seeded with the actual starting angle: with randomized initial joint
        positions, differencing against a default of 0 would read as a huge
        first-sample velocity.
        """
        self.meas_pos = self._measure(current_pos)
        self.vel_filt = 0.0

    def get_torque(self, target_pos, current_pos, current_vel=None):
        """Normalized torque in [-1, 1]. `current_vel` is used only when the
        encoder model is off; otherwise velocity comes from the quantized
        position, as on hardware."""
        prev = self.meas_pos
        self.meas_pos = self._measure(current_pos)
        vel = ((self.meas_pos - prev) / self.dt if self.resolution > 0.0
               else current_vel)
        self.vel_filt += self.vel_alpha * (vel - self.vel_filt)
        raw_torque = self.kp * (target_pos - self.meas_pos) - self.kd * self.vel_filt
        return np.clip(raw_torque, -1.0, 1.0)

def map_value(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def add_gaussian_noise(values, magnitude):
    noise = np.random.normal(loc=0.0, scale=magnitude, size=np.array(values).shape)
    return values + noise

def to_rotation_matrix(quat):
    return R.from_quat(quat, scalar_first=True).as_matrix().flatten()

def to_projected_gravity(quat):
    """World gravity direction (-z) expressed in the body frame.

    Yaw-invariant orientation cue: a rotation about world-z leaves world-z
    unchanged, so this 3-vector does not depend on the robot's heading. Returns
    a unit vector (g_body = R^T @ [0, 0, -1], where R is body-to-world).
    """
    r = R.from_quat(quat, scalar_first=True)
    return r.apply([0.0, 0.0, -1.0], inverse=True)

def add_rotational_noise(rot_matrices_flat, std_dev=0.01):
    rot_matrices = rot_matrices_flat.reshape(-1, 3, 3)
    noise_vecs = np.random.normal(scale=std_dev, size=(rot_matrices.shape[0], 3))
    noise_rotations = R.from_rotvec(noise_vecs)
    original_rotations = R.from_matrix(rot_matrices)
    noisy_rotations = noise_rotations * original_rotations
    return noisy_rotations.as_matrix().flatten()

def reverse_align_imu_quaternions(aligned_wxyz, imu_type):
    r_global = R.from_quat(aligned_wxyz, scalar_first=True)
    
    if imu_type == 'Front':
        r_align = R.from_euler('xyz', [0, 0, 90], degrees=True)
        
    elif imu_type == 'Back':
        r_align = R.from_euler('xyz', [180, 0, -90], degrees=True)
        
    r_raw = r_global * r_align.inv()
    
    raw_wxyz = r_raw.as_quat(scalar_first=True)
    return raw_wxyz