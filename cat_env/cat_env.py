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
w_pos = 1.0
w_sm = 0.1
w_en = 1.0
w_av = 0.0  # angular velocity penalty weight (temporarily disabled)
k = 0.2 # tanh gain param
init_ang_vel_max = 0.0  # rad/s, per-axis range for random initial tumble
# --------------------------

class CatEnv(MujocoEnv, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        model_path = os.path.abspath("model/cat.xml")
        
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)
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
        EzPickle.__init__(self)
        
        # Cache objects
        self._body_idx = {}
        self._joint_qpos_idx = {}
        self._joint_qvel_idx = {}
        
        for name in ["front_body", "rear_body", "spine_1", "spine_2", "tail"]:
            self._body_idx[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            
        for name in ["rot1", "pitch", "rot2", "tail"]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._joint_qpos_idx[name] = self.model.jnt_qposadr[jid]
            self._joint_qvel_idx[name] = self.model.jnt_dofadr[jid]

        # Cache nomical physics parameters
        self.nominal_mass = self.model.body_mass.copy()
        self.nominal_damping = self.model.dof_damping.copy()
        self.nominal_ipos = self.model.body_ipos.copy()
        self.nominal_inertia = self.model.body_inertia.copy()
        self.nominal_armature = self.model.dof_armature.copy()
        self.nominal_frictionloss = self.model.dof_frictionloss.copy()

        # Initialize variables
        self.steps = 0
        self.max_steps = 37
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

        # Per-joint PD gains, tuned in sim for a critically-damped (zeta~1) step
        # response: fastest settling with no overshoot or Coulomb limit cycle, on
        # each joint's composite inertia / stall torque / damping / friction.
        # Order: [rot1, pitch, rot2, tail]. kp*(err) - kd*vel -> normalized [-1,1].
        self.pd_nominal = [(2.0, 0.2), (20.0, 2.0), (2.0, 0.2), (20.0, 2.0)]
        self.pd = [util.PDController(kp, kd) for kp, kd in self.pd_nominal]

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

        # PD control
        roll_range = self.model.jnt_range[1][1]
        pitch_range = self.model.jnt_range[2][1]
        tail_range = self.model.jnt_range[4][1]
        executed_action[0] = util.map_value(executed_action[0], -1, 1, -roll_range, roll_range) # roll
        executed_action[1] = util.map_value(executed_action[1], -1, 1, -pitch_range, pitch_range) # pitch
        executed_action[2] = util.map_value(executed_action[2], -1, 1, -tail_range, tail_range) # tail
        
        for _ in range(self.frame_skip):
            norm_torque = np.zeros(4)

            # Recalculate torque based on CURRENT micro-state
            norm_torque[0] = self.pd[0].get_torque(executed_action[0],
                                              self.data.qpos[self._joint_qpos_idx["rot1"]], 
                                              self.data.qvel[self._joint_qvel_idx["rot1"]])
            norm_torque[1] = self.pd[1].get_torque(executed_action[1],
                                              self.data.qpos[self._joint_qpos_idx["pitch"]], 
                                              self.data.qvel[self._joint_qvel_idx["pitch"]])
            norm_torque[2] = self.pd[2].get_torque(-executed_action[0],
                                              self.data.qpos[self._joint_qpos_idx["rot2"]], 
                                              self.data.qvel[self._joint_qvel_idx["rot2"]])
            norm_torque[3] = self.pd[3].get_torque(executed_action[2],
                                              self.data.qpos[self._joint_qpos_idx["tail"]], 
                                              self.data.qvel[self._joint_qvel_idx["tail"]])
            
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

        if terminated or truncated:
            # np.save(f"control.npy", np.array(self.ctrls))
            self.ctrls = []
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def reset_model(self):
        self.steps = 0
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        # print("---")
        # Domain randomization
        # Mass
        mass_noise = np.random.uniform(0.8, 1.2, size=self.nominal_mass.shape)
        self.model.body_mass[:] = self.nominal_mass * mass_noise

        # Joint
        # Damping is a nominal estimate (from motor no-load speed / stall torque),
        # so randomize it wider (+/-30%) than the other params.
        damping_noise = np.random.uniform(0.7, 1.3, size=self.nominal_damping.shape)
        self.model.dof_damping[:] = self.nominal_damping * damping_noise
        # Armature is reflected rotor inertia estimated from motor class (no datasheet
        # rotor inertia), so ~+/-2x uncertainty -> randomize wide (0.5-1.5).
        armature_noise = np.random.uniform(0.5, 1.5, size=self.nominal_armature.shape)
        self.model.dof_armature[:] = self.nominal_armature * armature_noise
        friction_noise = np.random.uniform(0.7, 1.3, size=self.nominal_frictionloss.shape)
        self.model.dof_frictionloss[:] = self.nominal_frictionloss * friction_noise

        # PD gains are NOT randomized: the exact sim-tuned gains are flashed to the
        # hardware inner loop, so there is no gain uncertainty to be robust to.

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
        # Root free-joint angular velocity lives in qvel[3:6].
        qvel[3:6] = np.random.uniform(-init_ang_vel_max, init_ang_vel_max, size=3)

        self.set_state(qpos, qvel)
        
        return self._get_obs()

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
        front_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self._body_idx["front_body"], front_vel, 1)
        front_gyro = front_vel[:3]
        rear_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self._body_idx["rear_body"], rear_vel, 1)
        rear_gyro = rear_vel[:3]

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
        return obs.astype(np.float32)
    
    def _get_reward(self, action):
        front_quat = self.data.xquat[self._body_idx["front_body"]]
        rear_quat = self.data.xquat[self._body_idx["rear_body"]]

        # rotation matricies
        r_front = R.from_quat(front_quat, scalar_first=True)
        r_rear = R.from_quat(rear_quat, scalar_first=True)

        # transform local z vectors to global
        front_up = r_front.apply([0, 0, 1])
        rear_up = r_rear.apply([0, 0, 1])

        angle_front = np.arccos(np.clip(front_up[2], -1.0, 1.0))
        angle_rear = np.arccos(np.clip(rear_up[2], -1.0, 1.0))
        
        # get z component and scale
        reward_front = 1.0 - (angle_front / np.pi)
        reward_rear = 1.0 - (angle_rear / np.pi)
        
        r_pos = reward_front*reward_rear
        r_pos *= np.tanh(self.steps*k)
        
        delta = action - self.prev_action
        r_sm = np.mean(delta**2) * w_sm

        ctrl = self.data.ctrl
        r_en = np.mean(ctrl**2) * w_en

        # Angular velocity penalty (time-scaled: tolerated during rotation phase,
        # penalized as we approach landing)
        front_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 self._body_idx["front_body"], front_vel, 1)
        rear_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 self._body_idx["rear_body"], rear_vel, 1)
        front_ang_vel = front_vel[:3]
        rear_ang_vel = rear_vel[:3]
        ang_vel_sq = np.mean(front_ang_vel**2) + np.mean(rear_ang_vel**2)
        r_av = ang_vel_sq * w_av * np.tanh(self.steps * k)

        penalty_factor = np.exp(-(r_sm + r_en + r_av))

        final_reward = r_pos * penalty_factor
        reward_info = {
            "r_pos": r_pos,
            "r_sm": r_sm,
            "r_en": r_en,
            "r_av": r_av,
            "penalty_factor": penalty_factor
        }

        return final_reward, reward_info
    
    def _is_terminated(self):
        return False

    def _is_truncated(self):
        return self.steps >= self.max_steps
