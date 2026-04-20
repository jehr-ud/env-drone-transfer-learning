import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, Physics, DroneModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class SingleDroneEnv(BaseRLAviary):

    def __init__(self, gui=False):

        self.NUM_DRONES = 1

        self.goal = np.array([2.5, 2.5, 1.5])

        self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X)]

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            neighbourhood_radius=10,
            initial_rpys=np.zeros((1, 3)),
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=48,
            gui=gui,
            obs=ObservationType.KIN,
            act=ActionType.VEL
        )

        # -------------------------------
        # SPACES
        # -------------------------------
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(15,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # -------------------------------
        # CONSTANTES
        # -------------------------------
        self.MAX_SPEED = 2.0
        self.MAX_DIST = 10.0

        self.step_counter = 0
        self.prev_dist = 0.0

    # =========================================================
    # OBSERVATION
    # =========================================================
    def _computeObs(self):

        state = self._getDroneStateVector(0)

        pos = state[0:3]
        vel = state[10:13]
        rpy = state[7:10]

        # Goal relative
        goal_vec = self.goal - pos
        goal_rel = np.clip(goal_vec / 5.0, -1, 1)

        # Velocity
        vel_norm = np.clip(vel / 3.0, -1, 1)

        # Attitude
        rp = np.clip(rpy[0:2] / 0.5, -1, 1)

        # Yaw sin/cos
        yaw = np.array([
            np.sin(rpy[2]),
            np.cos(rpy[2])
        ])

        obs = np.concatenate([
            goal_rel,   # 3
            vel_norm,   # 3
            rp,         # 2
            yaw,        # 2
            [np.linalg.norm(goal_vec) / self.MAX_DIST]  # 1
        ])

        obs = np.clip(obs, -1.0, 1.0)

        return obs.astype(np.float32)

    # =========================================================
    # REWARD
    # =========================================================
    def _computeReward(self):

        state = self._getDroneStateVector(0)

        pos = state[0:3]
        vel = state[10:13]

        dist = np.linalg.norm(self.goal - pos)
        speed = np.linalg.norm(vel)

        # -------------------------
        # PROGRESS
        # -------------------------
        progress = self.prev_dist - dist
        progress_r = 5.0 * progress

        # -------------------------
        # DIRECTION
        # -------------------------
        goal_dir = (self.goal - pos)
        goal_dir /= (np.linalg.norm(goal_dir) + 1e-6)

        vel_dir = vel / (np.linalg.norm(vel) + 1e-6)

        alignment = np.dot(goal_dir, vel_dir)

        # -------------------------
        # DISTANCE SHAPING
        # -------------------------
        dist_reward = 1.0 - np.clip(dist / self.MAX_DIST, 0, 1)

        # -------------------------
        # MOVEMENT
        # -------------------------
        move_penalty = -0.2 if speed < 0.05 else 0.0

        # -------------------------
        # GOAL
        # -------------------------
        goal_bonus = 3.0 if dist < 0.3 else 0.0

        # -------------------------
        # TIME
        # -------------------------
        time_penalty = -0.01

        reward = (
            progress_r
            + 0.5 * alignment
            + 0.5 * dist_reward
            + move_penalty
            + goal_bonus
            + time_penalty
        )

        self.prev_dist = dist

        return reward

    # =========================================================
    # TERMINATION
    # =========================================================
    def _computeTerminated(self):

        state = self._getDroneStateVector(0)

        pos = state[0:3]
        roll, pitch = state[7], state[8]

        dist = np.linalg.norm(pos - self.goal)

        # goal
        if dist < 0.3:
            return True

        # crash
        if pos[2] < 0.05:
            return True

        if abs(roll) > 1.5 or abs(pitch) > 1.5:
            return True

        return False

    def _computeTruncated(self):

        if self.step_counter > 2000:
            return True

        return False

    # =========================================================
    # ACTION
    # =========================================================
    def _preprocessAction(self, action):

        state = self._getDroneStateVector(0)

        pos = state[0:3]
        vel = state[10:13]

        direction = np.clip(action[0:3], -1, 1)
        speed = (action[3] + 1) / 2  # [0,1]

        target_vel = direction * speed * 1.0

        target_pos = pos + target_vel * 0.1

        rpm, _, _ = self.ctrl[0].computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            cur_pos=pos,
            cur_quat=state[3:7],
            cur_vel=vel,
            cur_ang_vel=state[13:16],
            target_pos=target_pos,
            target_rpy=np.array([0, 0, state[9]]),
            target_vel=target_vel
        )

        return np.array([rpm])

    # =========================================================
    # STEP
    # =========================================================
    def step(self, action):

        self.step_counter += 1

        obs, _, term, trunc, info = super().step(action)

        reward = self._computeReward()

        obs = self._computeObs()

        return obs, reward, term, trunc, {}

    # =========================================================
    # RESET
    # =========================================================
    def reset(self, seed=None, options=None):

        self.step_counter = 0

        self.INIT_XYZS = np.array([[0.0, 0.0, 1.2]])

        obs, info = super().reset(seed=seed, options=options)

        pos = self._getDroneStateVector(0)[0:3]
        self.prev_dist = np.linalg.norm(self.goal - pos)

        return self._computeObs(), {}