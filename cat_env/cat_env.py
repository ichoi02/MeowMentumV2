import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle
from scipy.spatial.transform import Rotation as R
import cat_env.env_util as util
import mujoco
import os

# ---- train parameters ----
def _w(name, default):
    """Reward weight, overridable as CAT_W_<NAME> so a sweep needs no code edit."""
    return float(os.environ.get(f"CAT_W_{name}", default))

# Task terms.
w_pos = _w("POS", 1.0)      # uprightness reward weight
w_bonus = _w("BONUS", 1.0)  # bonus when BOTH bodies are upright (up_f, up_r > up_thresh)
up_thresh = 0.95  # per-body uprightness threshold for the success bonus (~26 deg tilt)

# Penalty terms, PER VARIANT. Every one trades directly against w_pos: set too
# high, a variant that cannot *reliably* right itself scores better holding still
# than trying, and training collapses to a passive policy instead of a
# bad-but-trying one. The no-tail variant does exactly that at 36% of task reward
# (6.8% success, 86 deg final tilt = no better than doing nothing).
#
# Tuned from all-zero by sweeping each term alone over a 4-point grid and then
# combining -- 77 training runs, both variants, docs/REWARD_TUNING.md. Three
# things that sweep settled and that any future edit should respect:
#
#   1. What binds is the TOTAL penalty budget, not any single weight. Grid
#      positions are sized in reward units (fraction of that variant's baseline
#      task reward), because a raw weight means nothing on its own: m_av ~ 18 and
#      m_en ~ 0.5, so one weight is crushing where the other is negligible.
#   2. The two variants need DIFFERENT weights, not a shared shape scaled down.
#      The ratios that came out of the sweep are en 0.69, av 0.00, jv 0.78,
#      time 0.32 -- no single scale factor produces that, which is why the old
#      notail_penalty_scale is gone.
#   3. w_av is the weak term. Swept alone it raises the magnitudes it does NOT
#      price: contact is disabled, angular momentum is conserved, and the policy
#      cannot shed rotation -- it only thrashes trying. Tail keeps it at the
#      cheapest useful setting; no-tail drops it entirely.
#
# Budgets: tail 54% of task reward (the ceiling -- 60% and above lands at 41%),
# no-tail 24% (its ceiling is between 24% and 36%, where success halves).
#
# Both were measured with the action-rate term `w_sm` still in the reward. It has
# since moved into the ACTOR loss (smooth_sac.py::SmoothSAC), and the four weights
# below are left at their swept values rather than rescaled: the budget is a
# CEILING, so spending less of it is the safe direction. `w_sm` was worth exactly
# one 6% grid position in each variant, so what is left here is 48% of task reward
# for tail (89% of the swept budget) and 18% for no-tail (75%). No-tail is
# comfortably fine -- 12% and 24% score 26.3% and 26.6%, inside noise of each
# other. Tail is the one to watch: 54% was where it *gained* 8 pp, and the nearest
# measured point below (36%) gave back half of that. If the tail variant regresses,
# that is the first thing to check, not the new actor term.
PENALTY_WEIGHTS = {
    #           torque      body omega   joint vel     simultaneity
    "tail":   {"en": 0.646, "av": 0.00547, "jv": 0.000815, "time": 0.848},
    "notail": {"en": 0.447, "av": 0.0,     "jv": 0.000633, "time": 0.271},
}

init_ang_vel_max = 0.5  # rad/s, per-axis range for random initial tumble
init_joint_pos_max = 0.2  # rad, per-joint range for random initial joint angles
filter_alpha = 0.3  # action low-pass gain (1.0 = no filter, smaller = smoother)
# --------------------------

# ---- privileged (teacher-only) domain-randomization block ----
# The teacher is a privileged policy: it may see things no sensor measures. The
# per-episode DR draw is exactly that -- the student cannot read its own rotor
# inertia, but it CAN infer it from how the joints responded over the last few
# frames, which is what distillation forces it to learn. Handing the teacher the
# true draw means it no longer has to infer the plant from the same 25 numbers it
# uses to act, so its labels are the actions of a policy that *knows* the robot.
#
# Appended at the END of the observation so every existing index is unchanged --
# distillation.py's student slices (front gravity 0:3, joint angles 12:16) and
# the hardware obs layout keep working untouched.
#
# Layout (all bodies below exclude the world body, index 0; 5 real bodies):
#   [ 0: 5]  body mass multipliers
#   [ 5:20]  body COM (ipos) offsets, xyz
#   [20:35]  body inertia multipliers, xyz
#   [35:47]  per motor GROUP (2) x [damping, armature, friction, ctrlrange, kp, kd]
#   [   47]  action delay
# Both model variants have the same 6 bodies / 4 joints / 2 motor groups, so this
# width is identical for tail and no-tail.
#
# Every entry is normalized to put nominal at exactly 0 and the sampled range at
# about [-1, 1]. Raw multipliers would hand SAC a block of inputs sitting at 1.0
# with 1e-3-scale variation (armature is 1.5e-4 Nm.s^2 nominal), which the first
# layer resolves about as poorly as it resolves the frame differences the student
# stacker exists to avoid.
DR_DIM = 48
BASE_OBS_DIM = 25


def _dr_scale(x, half_range, center=1.0):
    """Multiplier (or offset) -> ~[-1, 1], nominal at 0."""
    return (np.asarray(x) - center) / half_range

class CatEnv(MujocoEnv, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 50}

    def __init__(self, model_path="model/cat.xml", render_mode=None, privileged=True):
        model_path = os.path.abspath(model_path)

        # privileged=False reproduces the old 25-dim observation, so teachers
        # trained before the DR block was added still load and evaluate.
        self.privileged = privileged
        self._dr_vec = np.zeros(DR_DIM)
        obs_dim = BASE_OBS_DIM + (DR_DIM if privileged else 0)

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=20,
            observation_space=observation_space,
            default_camera_config={"distance": 3.0, "lookat": np.array([0.0, 0.0, 2])},
            render_mode=render_mode
        )
        self.action_space = action_space
        EzPickle.__init__(self, model_path=model_path, render_mode=render_mode,
                          privileged=privileged)
        
        # Cache objects
        self._body_idx = {}
        self._joint_qpos_idx = {}
        self._joint_qvel_idx = {}
        
        for name in ["front_body", "rear_body", "spine_1", "spine_2", "tail"]:
            self._body_idx[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            
        self._joint_id = {}
        self._actuator_idx = {}
        for name in ["rot1", "pitch", "rot2", "tail"]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._joint_id[name] = jid
            self._joint_qpos_idx[name] = self.model.jnt_qposadr[jid]
            self._joint_qvel_idx[name] = self.model.jnt_dofadr[jid]
            self._actuator_idx[name] = int(np.flatnonzero(self.model.actuator_trnid[:, 0] == jid)[0])

        # Motor groups: joints driven by the SAME motor + gearbox type. rot1/rot2 are
        # both the 9.68:1 roll motor; pitch/tail are both the 34.014:1 motor. The
        # randomization multipliers below (damping, armature, friction, ctrlrange, PD
        # gain) all model properties of the motor/gearbox -- rotor inertia, gear
        # friction, stall torque, torque constant -- so identical hardware must draw
        # ONE multiplier per group, not one per joint. Drawing them independently
        # would let training see a robot whose two roll motors behave differently,
        # which the real robot never does, and would wash out the shared-parameter
        # error the policy actually has to survive.
        self._motor_groups = (("rot1", "rot2"), ("pitch", "tail"))

        # Cache nomical physics parameters
        self.nominal_mass = self.model.body_mass.copy()
        self.nominal_damping = self.model.dof_damping.copy()
        self.nominal_ipos = self.model.body_ipos.copy()
        self.nominal_inertia = self.model.body_inertia.copy()
        self.nominal_armature = self.model.dof_armature.copy()
        self.nominal_frictionloss = self.model.dof_frictionloss.copy()
        self.nominal_ctrlrange = self.model.actuator_ctrlrange.copy()

        # Initialize variables
        self.steps = 0
        self.max_steps = 37
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

        # Per-joint PD gains. Order: [rot1, pitch, rot2, tail].
        # kp*(err) - kd*filtered_vel -> normalized [-1,1].
        #
        # Set by tools/tune_pd_gains.py: a real step response through this exact
        # control path (PD -> deadband + minimum-PWM floor -> ctrlrange -> 1 ms
        # mj_step) on the free-floating robot, taking the fastest settling time among
        # gains that show no overshoot and no limit cycle *in the worst case over the
        # domain-randomization range*, not merely at nominal -- the armature spread
        # alone is 6x, and gains that look critically damped at nominal ring inside
        # it. Settling (90th pct over DR, worst of 3 seeds): roll 0.432 s,
        # pitch 0.245 s, tail 0.246 s. Retuned after the tail mass went 172 -> 150 g:
        # the lighter tail made the previous (40, 2.0) overshoot past the gate on
        # both pitch and tail, and halving to (20, 1.0) is fractionally faster as
        # well as cleaner -- worth re-running the tuner after any inertia edit.
        #
        # The roll pair is damped harder than a noise-free model would ask for. With
        # exact qvel, kd = 0.2 looks clean; once the PD differentiates the real 48 CPR
        # encoder (13.5 mrad at the roll joint, only 2.2 ticks across the deadband) it
        # is visibly under-damped -- 6.2% overshoot at the 95th percentile, 2 velocity
        # sign changes, and 3 of 256 DR draws never settling. kd = 0.4 removes all of
        # that. kp = 4 is then the fastest roll pair that still passes every gate
        # on every seed; kp = 5 settles 17% quicker but overshoots 2.8% at p95 and
        # leaves a draw unsettled.
        #
        # rot1/rot2 must share gains: the policy drives rot2 as -rot1, so asymmetric
        # roll gains would turn the counter-twist into a net torque. The roll pair is
        # unchanged by the sweep -- it is torque-limited, not gain-limited (0.17 Nm
        # against the body inertia), so raising kp only saturates sooner. pitch and
        # pitch and tail share a gain pair because their step responses land within
        # 1% of each other, not because they must.
        self.pd_nominal = [(4.0, 0.4), (20.0, 1.0), (4.0, 0.4), (20.0, 1.0)]
        self.joint_names = ("rot1", "pitch", "rot2", "tail")
        # Each controller carries its own encoder resolution, so the inner loop
        # differentiates a quantized angle exactly as the Teensy does.
        self.pd = [util.PDController(kp, kd,
                                     resolution=util.encoder_resolution(name),
                                     dt=self.model.opt.timestep)
                   for (kp, kd), name in zip(self.pd_nominal, self.joint_names)]

        # Penalty weights are per-instance and per-variant: a variant with less
        # authority reaches the passive optimum at a lower budget, and at the
        # tailed weights the no-tail robot collapses outright (6.6% success,
        # 91 deg tilt -- docs/REWARD_TUNING.md). CAT_W_* still overrides, so a
        # sweep needs no code edit.
        variant = "notail" if "notail" in os.path.basename(model_path) else "tail"
        w = PENALTY_WEIGHTS[variant]
        self.w_en = _w("EN", w["en"])
        self.w_av = _w("AV", w["av"])
        self.w_jv = _w("JV", w["jv"])
        self.w_time = _w("TIME", w["time"])

        self.ctrls = []

    def step(self, action):
        self.steps += 1
        
        action = np.clip(action, -1, 1)

        # Random delay
        if self.action_delay > 0:
            self.action_buffer.append(action.copy())
            executed_action = self.action_buffer.pop(0)
        else:
            executed_action = action.copy()

        # First-order low-pass on the commanded (normalized) target: suppresses
        # high-frequency action reversals / jitter and mirrors the hardware's
        # actuator + PD low-pass. State persists across steps, reset per episode.
        self.action_filt = filter_alpha * executed_action + (1.0 - filter_alpha) * self.action_filt
        executed_action = self.action_filt.copy()

        # PD control
        roll_range = self.model.jnt_range[1][1]
        pitch_range = self.model.jnt_range[2][1]
        tail_range = self.model.jnt_range[4][1]
        executed_action[0] = util.map_value(executed_action[0], -1, 1, -roll_range, roll_range) # roll
        executed_action[1] = util.map_value(executed_action[1], -1, 1, -pitch_range, pitch_range) # pitch
        executed_action[2] = util.map_value(executed_action[2], -1, 1, -tail_range, tail_range) # tail
        
        # Joint targets in actuator order; rot2 counter-twists rot1.
        targets = (executed_action[0], executed_action[1],
                   -executed_action[0], executed_action[2])

        for _ in range(self.frame_skip):
            norm_torque = np.zeros(4)

            # Recalculate torque based on CURRENT micro-state, then apply the
            # Teensy deadband / minimum-PWM floor so the sim cannot exploit
            # fine torques the real inner loop is incapable of commanding.
            for i, (name, target) in enumerate(zip(self.joint_names, targets)):
                pos = self.data.qpos[self._joint_qpos_idx[name]]
                tau = self.pd[i].get_torque(target, pos)
                # The firmware's deadband test uses the encoder reading, not truth,
                # so the error compared here is the quantized one.
                norm_torque[i] = util.apply_motor_deadband(
                    tau, target - self.pd[i].meas_pos, util.joint_deadband(name))


            # Map normalized torque to physical torque
            physical_torque = np.zeros(4)
            for i in range(4):
                ctrl_min, ctrl_max = self.model.actuator_ctrlrange[i]
                physical_torque[i] = util.map_value(norm_torque[i], -1.0, 1.0, ctrl_min, ctrl_max)
            
            # Apply to MuJoCo and advance physics by exactly 1 ms
            self.data.ctrl[:] = physical_torque
            mujoco.mj_step(self.model, self.data)

        self.ctrls.append(np.hstack([executed_action, self.data.qpos[7:]]))
        # self.ctrls.append(physical_torque)
        observation = self._get_obs()
        reward, reward_info = self._get_reward(action)
        info = reward_info 
        
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        self.prev_action = action

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def reset_model(self):
        self.steps = 0
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        # Per-episode command log: rows of [mapped target (roll, pitch, tail),
        # joint qpos (4)]. Cleared here rather than at episode end so the finished
        # episode stays readable -- plots/plot_joint_tracking.py reads it.
        self.ctrls = []
        # print("---")
        # Domain randomization
        # Mass
        mass_noise = np.random.uniform(0.8, 1.2, size=self.nominal_mass.shape)
        self.model.body_mass[:] = self.nominal_mass * mass_noise

        # Motor/gearbox parameters. One draw per motor GROUP (see _motor_groups):
        # the two roll joints are the same motor, as are pitch and tail, so their
        # multipliers are shared. Start from nominal so the root free-joint dofs
        # (which no motor drives) stay untouched.
        self.model.dof_damping[:] = self.nominal_damping
        self.model.dof_armature[:] = self.nominal_armature
        self.model.dof_frictionloss[:] = self.nominal_frictionloss
        self.model.actuator_ctrlrange[:] = self.nominal_ctrlrange

        dr_motor = []
        for group in self._motor_groups:
            # Damping is a nominal estimate (from motor no-load speed / stall torque),
            # so randomize it wider (+/-30%) than the other params.
            damping_noise = np.random.uniform(0.7, 1.3)
            # Armature is reflected rotor inertia estimated from motor class (no datasheet
            # rotor inertia). It is a pure estimate, so randomize it log-symmetrically
            # over ~2.5x either way rather than the narrower band the other params use.
            armature_noise = np.random.uniform(0.4, 2.5)
            friction_noise = np.random.uniform(0.7, 1.3)
            # Actuator authority. ctrlrange encodes stall torque, which moves with LiPo
            # sag across the drop, motor temperature and unit-to-unit spread. Scaling
            # both limits keeps the range symmetric, so this is a pure torque scale.
            ctrlrange_noise = np.random.uniform(0.8, 1.2)
            # PD gains. The flashed inner-loop gains equal pd_nominal by construction,
            # but the PWM -> physical torque constant they act through is not measured,
            # so the effective closed-loop gain still carries real uncertainty. That
            # constant is a property of the motor, hence also per-group.
            kp_noise = np.random.uniform(0.8, 1.2)
            kd_noise = np.random.uniform(0.8, 1.2)

            # Armature is the one draw that is log-uniform-ish rather than a narrow
            # band around 1 (0.4 = 1/2.5), so it is normalized in log space; a linear
            # scaling would put nominal at -0.6 instead of 0 and squash the whole
            # lower half of the range into a fifth of the axis.
            dr_motor.extend([
                _dr_scale(damping_noise, 0.3),
                np.log(armature_noise) / np.log(2.5),
                _dr_scale(friction_noise, 0.3),
                _dr_scale(ctrlrange_noise, 0.2),
                _dr_scale(kp_noise, 0.2),
                _dr_scale(kd_noise, 0.2),
            ])

            for name in group:
                dof = self._joint_qvel_idx[name]
                act = self._actuator_idx[name]
                i = self.joint_names.index(name)
                self.model.dof_damping[dof] = self.nominal_damping[dof] * damping_noise
                self.model.dof_armature[dof] = self.nominal_armature[dof] * armature_noise
                self.model.dof_frictionloss[dof] = self.nominal_frictionloss[dof] * friction_noise
                self.model.actuator_ctrlrange[act] = self.nominal_ctrlrange[act] * ctrlrange_noise
                self.pd[i].kp = self.pd_nominal[i][0] * kp_noise
                self.pd[i].kd = self.pd_nominal[i][1] * kd_noise

        # COM position
        ipos_noise = np.random.uniform(-0.04, 0.04, size=self.nominal_ipos.shape)
        ipos_noise[0] = 0.0  # Crucial: Do not move the world body (index 0)
        self.model.body_ipos[:] = self.nominal_ipos + ipos_noise

        # Inertia tensor
        inertia_noise = np.random.uniform(0.8, 1.2, size=self.nominal_inertia.shape)
        self.model.body_inertia[:] = self.nominal_inertia * inertia_noise

        # Delay
        self.action_delay = np.random.randint(0, 3)
        zero_action = np.zeros(self.action_space.shape)
        self.action_buffer = [zero_action.copy() for _ in range(self.action_delay)]

        # Action low-pass filter state (starts neutral)
        self.action_filt = np.zeros(self.action_space.shape, dtype=np.float32)

        # Freeze this episode's draw as the privileged block of the observation.
        # Constant for the whole episode by construction -- the DR is resampled
        # only here -- so the teacher reads the same 48 numbers every step.
        self._dr_vec = np.concatenate([
            _dr_scale(mass_noise[1:], 0.2),
            (ipos_noise[1:] / 0.04).ravel(),
            _dr_scale(inertia_noise[1:], 0.2).ravel(),
            np.array(dr_motor),
            [self.action_delay - 1.0],  # {0,1,2} steps -> {-1,0,1}
        ])
        assert self._dr_vec.shape == (DR_DIM,), \
            f"DR_DIM={DR_DIM} does not match this model ({self._dr_vec.shape[0]})"

        # Set physics params
        mujoco.mj_setConst(self.model, self.data)

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Randomize initial orientation. Policy and reward are yaw-invariant, so we
        # sample a uniformly random attitude (SO(3)): full roll/pitch coverage, with
        # heading a free dimension. The robot may be dropped in any orientation.
        r = R.random()
        quat_xyzw = r.as_quat()
        qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

        # Randomize initial angular velocity: the robot is dropped already tumbling.
        # Root free-joint angular velocity lives in qvel[3:6]. This is what puts
        # nonzero angular momentum in the episode -- contact is disabled, so L is
        # conserved throughout, and at exactly 0 the policy would only ever see the
        # (easier, and unreachable on a real release) zero-momentum problem.
        qvel[3:6] = np.random.uniform(-init_ang_vel_max, init_ang_vel_max, size=3)

        # Randomize initial joint angles: the release pose is never exactly the
        # nominal home pose (servo zero offset, backlash, hand placement), so the
        # policy must not assume it starts folded exactly at zero. Clipped to each
        # joint's limits so no episode starts outside its range.
        joint_noise = np.random.uniform(-init_joint_pos_max, init_joint_pos_max,
                                        size=len(self.joint_names))
        for name, noise in zip(self.joint_names, joint_noise):
            adr = self._joint_qpos_idx[name]
            lo, hi = self.model.jnt_range[self._joint_id[name]]
            qpos[adr] = np.clip(qpos[adr] + noise, lo, hi)

        self.set_state(qpos, qvel)

        # Per-episode inner-loop state, seeded from the pose actually being started
        # from: initial joint angles are randomized, so differencing against 0 would
        # read as a large first-sample velocity.
        for pd, name in zip(self.pd, self.joint_names):
            pd.reset(self.data.qpos[self._joint_qpos_idx[name]])
        
        return self._get_obs()

    def _body_gyro(self, name):
        """Body-frame angular velocity of a body (what its IMU gyro would read)."""
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 self._body_idx[name], vel, 1)
        return vel[:3]

    def _get_obs(self):
        # The observation is strictly yaw-invariant: nothing below depends on the
        # robot's heading about world-z. Absolute root position/orientation and
        # world-frame root velocities are dropped (heading-dependent and useless
        # for a contact-free mid-air righting task).

        # Projected gravity (yaw-invariant orientation cue): world -z in body frame.
        # Captures exactly the non-yaw part of each body's orientation.
        front_proj_grav = util.to_projected_gravity(self.data.xquat[self._body_idx["front_body"]])
        rear_proj_grav = util.to_projected_gravity(self.data.xquat[self._body_idx["rear_body"]])

        # Gyro (body-frame angular velocity, yaw-invariant)
        front_gyro = self._body_gyro("front_body")
        rear_gyro = self._body_gyro("rear_body")

        # Joint state only (root free-joint qpos[:7]/qvel[:6] dropped): 4 joints in
        # XML order [rot1, pitch, rot2, tail]. Joint-space -> yaw-invariant.
        joint_qpos = self.data.qpos[7:]
        joint_qvel = self.data.qvel[6:]

        # Control signal (joint-space torques, yaw-invariant)
        ctrl = self.data.ctrl
        step = np.array([self.steps / self.max_steps])

        obs = np.concatenate([
            front_proj_grav, rear_proj_grav,
            front_gyro, rear_gyro,
            joint_qpos, joint_qvel,
            ctrl, step
        ])

        # Privileged tail: this episode's domain-randomization draw. Teacher-only
        # by construction -- the student's obs is built from slices of the block
        # above (distillation.py), so nothing downstream of distillation ever sees
        # it, and it must stay LAST so those slice indices never move.
        if self.privileged:
            obs = np.concatenate([obs, self._dr_vec])
        return obs.astype(np.float32)
    
    def _get_reward(self, action):
        front_quat = self.data.xquat[self._body_idx["front_body"]]
        rear_quat = self.data.xquat[self._body_idx["rear_body"]]

        # World-frame body up-vectors (local +z mapped to world).
        front_up = R.from_quat(front_quat, scalar_first=True).apply([0, 0, 1])
        rear_up = R.from_quat(rear_quat, scalar_first=True).apply([0, 0, 1])

        # Per-body uprightness in [0, 1] (up[2] = cos(tilt); 1 = perfectly upright).
        up_f = 0.5 * (front_up[2] + 1.0)
        up_r = 0.5 * (rear_up[2] + 1.0)

        # Dense, per-step, NO time ramp: every upright step counts, so under the SAC
        # discount (gamma < 1) the optimal policy rights ASAP and holds. The sum term
        # gives each body an independent gradient (no vanishing when the other is
        # inverted); the product term rewards true "both upright" simultaneously.
        r_pos = w_pos * (0.5 * (up_f + up_r) + up_f * up_r)

        # Success bonus: crisp "reach upright and stay" signal once both are upright.
        r_bonus = w_bonus if (up_f > up_thresh and up_r > up_thresh) else 0.0

        # --- penalty magnitudes, unweighted ---
        # Each is the raw physical quantity the matching weight prices. They are
        # reported in `info` as m_* alongside the weighted r_*, because the tuning
        # question ("did raising w_jv actually reduce joint velocity, and by how
        # much?") is about the magnitude, not the reward it contributes -- and the
        # weighted term is uninformative at w = 0. docs/evaluate.py --stats reads
        # these; keeping the definition here means the tuning instrument cannot
        # drift from what the reward actually optimizes.
        d_action = np.abs(action - self.prev_action)

        # Applied torque: energy.
        m_en = np.mean(self.data.ctrl ** 2)

        # NOTE: the action-rate term (`w_sm * mean(d_action ** 2)`) used to sit
        # here. It is now an actor-loss term on the DETERMINISTIC mean action --
        # see smooth_sac.py -- so the quantity smoothed is the commanded
        # trajectory rather than the exploration-noised action the buffer holds.
        # d_action survives because m_time still needs it.

        # Body angular velocity: an upright pose reached while still tumbling is not
        # a landing. Contact is disabled and angular momentum is conserved, so the
        # policy cannot shed omega -- it can only park the *bodies* still by putting
        # the residual rotation into the joints. This term asks for exactly that.
        # It opposes the maneuver itself (reorienting at zero momentum REQUIRES the
        # bodies to rotate), so it is the term most likely to cause a passive
        # collapse and is weighted accordingly.
        omega = np.concatenate([self._body_gyro("front_body"), self._body_gyro("rear_body")])
        m_av = np.mean(omega ** 2)

        # Joint velocity: suppress motion that does not buy reorientation -- flailing
        # inside the joint limits, and the fast reversals that a real geared motor
        # cannot track. Penalizing velocity rather than the commanded rate (which
        # the actor-loss smoothness term now covers) catches motion the PD loop
        # produces on its own, e.g. ringing after a step.
        m_jv = np.mean(self.data.qvel[6:] ** 2)

        # Timing: penalize the three channels moving AT THE SAME TIME. The pairwise
        # product is zero whenever only one channel is moving and grows only when two
        # or more move together, so a sequential roll -> pitch -> tail maneuver is
        # free while a simultaneous one is not. This is a sim2real term: coupled
        # multi-joint motion is where the model is least trustworthy (unmodeled
        # spine flex, shared battery sag across four motors at once), whereas a
        # step-by-step maneuver moves through states the sim gets right.
        m_time = (d_action[0] * d_action[1] + d_action[0] * d_action[2]
                  + d_action[1] * d_action[2])

        r_en = self.w_en * m_en
        r_av, r_jv, r_time = self.w_av * m_av, self.w_jv * m_jv, self.w_time * m_time

        final_reward = r_pos + r_bonus - r_en - r_av - r_jv - r_time
        reward_info = {
            "r_pos": r_pos,
            "r_bonus": r_bonus,
            "r_en": r_en,
            "r_av": r_av,
            "r_jv": r_jv,
            "r_time": r_time,
            "m_en": m_en,
            "m_av": m_av,
            "m_jv": m_jv,
            "m_time": m_time,
            "up_mean": 0.5 * (up_f + up_r),
        }

        return final_reward, reward_info
    
    def _is_terminated(self):
        return False

    def _is_truncated(self):
        return self.steps >= self.max_steps
