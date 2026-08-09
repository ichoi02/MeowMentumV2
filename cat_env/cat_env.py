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

# Task terms. There is only one: the success BONUS was removed.
#
# It was `w_bonus * 1[both bodies > up_thresh]`, a step at 25.8 deg of tilt -- stricter
# than the 30 deg that `docs/evaluate.py` actually scores success at, so a policy could
# succeed by the reported metric while never earning the term meant to encourage it.
# Worse, it was badly morphology-biased: measured over 200 episodes it fired on 19.3% of
# tail steps but 5.8% of no-tail steps against a 4.5% PASSIVE rate -- i.e. for no-tail it
# was statistically indistinguishable from doing nothing, so it carried no gradient for
# the arm that needed it. Removing it cuts the tail:no-tail asymmetry in task-reward gain
# from 2.15x to 1.82x.
#
# w_pos is 1.5 rather than 1.0 to restore scale, not to change the balance: dropping the
# bonus and linearising `up` (see _get_reward) together cut no-tail's reward gain over a
# passive policy from 0.347/step to 0.248, and 1.4-1.6x puts it back. The penalty weights
# below are sized against task reward, so leaving w_pos at 1.0 would have quietly made
# every penalty ~40% more expensive.
w_pos = _w("POS", 1.5)      # uprightness reward weight

# Penalty terms, SHARED BY BOTH VARIANTS. Every one trades directly against w_pos:
# set too high, a variant that cannot *reliably* right itself scores better holding
# still than trying, and training collapses to a passive policy instead of a
# bad-but-trying one.
#
# One vector for both arms is a deliberate choice, and it costs something. This is
# a MORPHOLOGY ablation: if the arms run different rewards, the tail-vs-no-tail gap
# mixes the tail's physical contribution with a reward-budget difference, which is
# what the per-variant weights that used to live here did. Identical raw weights
# make the reward function literally the same; the two robots then spend different
# FRACTIONS of their task reward on it, and that difference is a consequence of the
# morphology rather than a confound in the comparison.
#
# Found by scaling the previous tail vector by a single factor k and evaluating on
# fixed release attitudes -- roll 180/90/45/0 at pitch 0, 300 drops each, held-out
# seed 4242 (docs/shared_k_runs.jsonl). No-tail is the binding arm, so k was chosen
# on it. Mean success rose monotonically from 47.2% at k=0 to 58.6% at k=1, and k=1
# is what is below.
#
# Two things any future edit should know:
#
#   1. The budget is far smaller than the raw numbers suggest, because the grid
#      that produced them was sized against baselines that no longer hold. Action
#      smoothness moved into the ACTOR loss (smooth_sac.py), which suppresses
#      action increments unconditionally, and m_time is quadratic in those -- the
#      penalty-free no-tail baseline fell from m_time 0.304 to 0.044. This vector
#      costs the trained no-tail policy ~12% of its task reward, not the ~43% the
#      same numbers cost when they were tuned. The old collapse ceiling (~18-30%)
#      was never reached anywhere in the k sweep.
#   2. The mean over release angles hides a real regression at roll 180. Success
#      there falls monotonically as k rises -- 24.0% at k=0 to 16.3% at k=1 --
#      while roll 90 climbs 34.0% to 72.3%. The budget buys success at the easy
#      attitudes and pays for it upside-down, which is the case a falling-cat
#      robot arguably exists for. If that matters more than the mean, k=0.25 is
#      the better pick and this vector is the wrong one.
#
# `w_time` IS NOW ZERO for both variants. The simultaneity term is computed on the
# RAW action (`d_action` in _get_reward), but the per-channel low-pass above now sits
# between that action and the plant, so raw action rate no longer corresponds to
# physical joint motion -- the filter absorbs it. The term was therefore pricing a
# quantity the robot does not feel. The tell showed up immediately on the first
# filtered teachers: no-tail's `r_time` reached 0.531/step, ~31% of its task reward
# and 10x the tail's, against ~0.01 on every pre-filter run. Whatever the policy was
# buying with that spend, it was not smoother hardware motion.
#
# If simultaneity is wanted back, price it on something the filter cannot absorb --
# joint velocity products from `self.data.qvel[6:]` rather than action differences.
#
# CURRENT VECTOR: the 3% budget under the exponential pose reward. Re-sized from
# scratch against that reward's own measured penalty-free baseline (m_en, m_av, m_jv
# and r_pos over 3 seeds), not carried over -- carrying a vector across a reward change
# is what previously left it costing 12% of task reward when it had been tuned for 43%.
#
# 26 runs, 1M steps each, evaluated on roll 180/90/45/0 x 300 drops at held-out seed
# 4242 (docs/exp_budget_runs.jsonl). Budget ladder, delta vs penalty-free:
#
#   budget     3%     6%    12%    18%    24%    36%
#   tail    + 6.3  + 5.0  + 2.9  - 0.5  - 6.5  - 9.5
#   notail  - 3.8  - 4.6  - 5.8  - 7.8  - 3.2  - 3.2
#
# NO budget is free. No-tail is degraded at every point, and the damage is NOT
# proportional to the budget -- 3% costs 3.8 pp and 36% costs 3.2 pp -- so this is a
# fixed cost of having any penalty at all, not a scale that can be tuned down. 3% is
# chosen as the cheapest point that still prices torque, joint velocity and body
# rotation at all; those terms exist for hardware transfer, not for sim score, and
# penalty-free would leave the commanded trajectory unconstrained.
#
# Two things not to over-read:
#   1. The tail row is single-seed. Penalty-free tail spans 53.7 / 58.7 / 68.8 across
#      three seeds -- a 15.2 pp spread -- so every tail number above sits inside seed
#      noise, INCLUDING the +6.3 that makes 3% look best. No-tail is the trustworthy
#      arm here: its penalty-free spread is 2.0 pp.
#   2. Inverted righting is nearly gone for no-tail under this reward: roll 180 is
#      4.9% penalty-free and 0.7% here. It was 2.0% under cos+bonus. That is the case
#      a falling-cat robot exists for, and no configuration measured so far solves it.
PENALTY_WEIGHTS = {
    #        torque      body omega    joint vel      simultaneity
    "tail":   {"en": 0.03859, "av": 0.002123, "jv": 0.0003087, "time": 0.0},
    "notail": {"en": 0.03859, "av": 0.002123, "jv": 0.0003087, "time": 0.0},
}

init_ang_vel_max = 0.5  # rad/s, per-axis range for random initial tumble
init_joint_pos_max = 0.2  # rad, per-joint range for random initial joint angles
init_pitch_max_deg = 45.0  # deg, release pitch range (roll is full; yaw is irrelevant)

# Per-channel first-order low-pass on the NORMALIZED action, applied before the
# joint-range mapping. Order matches the action vector: [roll, pitch, tail].
#
# Sized against measured hardware, not chosen by feel. Over 7 logged drops the
# unfiltered policy commanded 69 rad of roll travel per 0.74 s episode while the joint
# physically delivered 6.9 -- a factor of 10. One alpha cannot fix that without
# over-damping pitch: roll runs +-7.28 rad through 9.68:1 gearing, pitch +-1.57 rad
# through 34:1, so in range-units roll is far slower. Applying each candidate alpha to
# the logged action sequence gives commanded/achievable travel of:
#
#         alpha    roll     pitch
#         1.00     10.0x     2.1x     (what actually flew)
#         0.30      4.0x     0.9x
#         0.10      1.7x     0.5x
#
# Hence 0.3 on pitch and 0.1 on roll/tail. Tail follows roll rather than pitch because
# it shares pitch's gearing but a wider range, and because on the no-tail variant it is
# zeroed outright (see hardware/controller.py).
#
# This is state: it persists across steps within an episode and resets per episode, so
# the plant the policy acts on is no longer memoryless. The student observation carries
# the previous action for exactly this reason (distillation.py) -- without it the
# filter state is unobservable and the student's problem stops being Markov.
filter_alpha = np.array([0.1, 0.3, 0.1])  # [roll, pitch, tail]
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
DR_DIM = 48 # domain randomization vector width, teacher-only; see above
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
        # 50 steps x 20 ms = 1.0 s. Raised from 37 (0.74 s) because the drop was
        # ending mid-manoeuvre: released inverted, the no-tail teacher was still
        # righting at 119 deg/s when the episode was truncated, finishing at 50 deg
        # against a 30 deg threshold -- it needed roughly another 0.17 s. This is a
        # taller drop, so hardware/controller.py::CONTROL_DURATION moves with it.
        self.max_steps = 50
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

        # Per-channel low-pass on the commanded (normalized) target. Applied AFTER the
        # delay buffer so the filter sees the action in the order the plant does, and
        # BEFORE the range mapping so alpha is expressed in normalized units and does
        # not silently change meaning if a joint range is edited.
        self.action_filt = (filter_alpha * executed_action
                            + (1.0 - filter_alpha) * self.action_filt)
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

        # Action low-pass state, neutral at release (the robot is dropped with its
        # joints wherever they were, but with no command history).
        self.action_filt = np.zeros(self.action_space.shape, dtype=np.float64)

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

        # Randomize initial orientation: FULL roll, but pitch only within +-45 deg.
        #
        # This is the release the robot is actually given -- held by hand, rolled to
        # some angle about its long axis, and let go roughly level fore-aft. A uniform
        # SO(3) draw (the previous behaviour) spends most of its mass on nose-up /
        # nose-down attitudes that never occur in a drop test, so the policy was
        # spending capacity on a distribution it is never evaluated or deployed in.
        #
        # Yaw is fixed at 0 rather than sampled: the observation is projected gravity
        # and the reward is a tilt angle, both invariant to heading, so a yaw draw adds
        # nothing but variance. Euler order matches docs/evaluate.py::force_attitude,
        # so `--roll X` there lands exactly on this distribution.
        roll_deg = np.random.uniform(-180.0, 180.0)
        pitch_deg = np.random.uniform(-init_pitch_max_deg, init_pitch_max_deg)
        quat_xyzw = R.from_euler("xyz", [roll_deg, pitch_deg, 0.0], degrees=True).as_quat()
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

        # Per-body uprightness, LINEAR in tilt angle: 1 at upright, 0 at inverted, with
        # a constant gradient of 1/pi everywhere.
        #
        # The clip is load-bearing: `apply` routinely returns 1.0 + 1e-16, and arccos
        # of that is NaN, which would silently poison a whole episode's reward.
        #
        # Shapes measured, teacher mean success over roll 180/90/45/0, 300 drops each at
        # held-out seed 4242, all under the OLD uniform-SO(3) release (tail / no-tail):
        #
        #   0.5*(cos(tilt)+1) + step bonus, penalties on   63.5% / 35.8%
        #   1 - tilt/pi, no bonus, penalties on            47.8% / 32.3%
        #   1 - tilt/pi, no bonus, penalties OFF           56.6% / 36.9%
        #   exp(-tilt), no bonus, penalties OFF (3 seeds)  60.4% / 39.2%
        #   exp(-tilt), no bonus, 3% budget                66.7% / 35.3%
        #
        # Gradient is what distinguishes them. cos gives 0.5*sin(tilt), which VANISHES at
        # both ends (0.04 at 5 deg and at 175 deg, against 0.50 at 90 deg) -- flattest
        # exactly where the policy is stuck. Linear gives a constant 1/pi. exp(-k*tilt)
        # gives k*exp(-k*tilt): 1.00 at 0 deg and 0.59 at 30 deg for k=1, against linear's
        # 0.32, but only 0.21 at 90 deg and 0.04 at 180 deg.
        #
        # Two things the search established and this file should not lose:
        #   - Offline scoring of a shape on FIXED trajectories does not predict what it
        #     TRAINS. It called linear roughly neutral; linear trained 15.7 pp worse on
        #     the tail. Shapes are compared by training runs here, not by analysis.
        #   - Steepening near upright amplifies the morphology asymmetry, because
        #     reaching upright is exactly what no-tail cannot do. Scored on real
        #     trajectories the tail:no-tail gain ratio runs 1.82x (cos), 1.90x (linear),
        #     2.33x (exp k=1), 2.87x (exp k=2). It is the same bias the deleted step
        #     bonus had, just smoothed.
        up_f = 1.0 - np.arccos(np.clip(front_up[2], -1.0, 1.0)) / np.pi
        up_r = 1.0 - np.arccos(np.clip(rear_up[2], -1.0, 1.0)) / np.pi

        # Dense, per-step, NO time ramp: every upright step counts, so under the SAC
        # discount (gamma < 1) the optimal policy rights ASAP and holds. The sum term
        # gives each body an independent gradient (no vanishing when the other is
        # inverted); the product term rewards true "both upright" simultaneously.
        r_pos = w_pos * (0.5 * (up_f + up_r) + up_f * up_r)

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

        final_reward = r_pos - r_en - r_av - r_jv - r_time
        reward_info = {
            "r_pos": r_pos,
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
