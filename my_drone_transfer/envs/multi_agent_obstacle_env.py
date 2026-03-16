import numpy as np
import pybullet as p

from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, Physics, DroneModel


class MultiAgentObstacleEnv(BaseRLAviary):

    def __init__(
        self,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        gui=False
    ):

        self.colors = [
            [0.86, 0.37, 0.34, 1],
            [0.35, 0.70, 0.90, 1],
            [0.50, 0.78, 0.50, 1],
            [0.95, 0.77, 0.35, 1],
            [0.72, 0.56, 0.87, 1],
            [0.60, 0.60, 0.60, 1]
        ]

        self.obstacles = [
            ([3, 0, 1.5], [0.2, 3, 1.5], 3, "wall"),
            ([-3, 0, 1.5], [0.2, 3, 1.5], 4, "wall"),

            ([0, 0, 1.5], 3, 0, "cube"),
            ([0, 1.5, 1.0], 2, 1, "cube"),
            ([0, -1.5, 1.0], 2, 2, "cube"),

            ([1.5, 0.8, 1.0], 2, 5, "cube"),
            ([-1.5, -0.8, 1.0], 2, 0, "cube"),

            ([1.2, -2.0, 1.0], 2, 2, "cube"),
            ([-1.2, 2.0, 1.0], 2, 1, "cube"),

            ([2.2, 1.5, 2.0], 4, 1, "cylinder"),
            ([-2.2, -1.5, 2.0], 2, 2, "cylinder"),

            ([2.0, -1.5, 2.0], 4, 4, "cylinder"),
            ([-2.0, 1.5, 2.0], 2, 5, "cylinder"),

            ([0.8, 2.5, 2.0], 2, 1, "cylinder"),
            ([-0.8, -2.5, 2.0], 3, 2, "cylinder")
        ]

        self.goals = np.array([
            [2.5, 2.5, 1.8],
            [-2.5, -2.5, 1.8]
        ])

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=2,
            neighbourhood_radius=10,
            initial_xyzs=np.array([
                [0, -2, 1],
                [0, 2, 1]
            ]),
            initial_rpys=np.zeros((2, 3)),
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=48,
            gui=gui,
            obs=obs,
            act=act
        )

        self.prev_goal_dist = np.zeros(self.NUM_DRONES)
        self.prev_action = np.zeros(self.NUM_DRONES * 3)
        self.last_action = np.zeros(self.NUM_DRONES * 3)
        
        # (goal_rel + vel + attitude + ang_vel + other_rel + obstacle_distances) * total_drones
        # obs_dim = self.NUM_DRONES * (3 + 3 + 3 + 3 + 3 + len(self.obstacles))

        self.max_steps = 800

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(60,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.NUM_DRONES * 3,),
            dtype=np.float32
        )

        self.episode_reward = 0.0
        self.step_counter = 0

    def _addObstacles(self):

        for pos, size, color_idx, obstacle_type in self.obstacles:

            if obstacle_type == "cube":

                obstacle_id = p.loadURDF(
                    "cube_small.urdf",
                    pos,
                    globalScaling=size,
                    physicsClientId=self.CLIENT
                )

                p.changeVisualShape(
                    obstacle_id,
                    -1,
                    rgbaColor=self.colors[color_idx],
                    physicsClientId=self.CLIENT
                )

            elif obstacle_type == "cylinder":

                collision = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=0.25,
                    height=size,
                    physicsClientId=self.CLIENT
                )

                visual = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=0.25,
                    length=size,
                    rgbaColor=self.colors[color_idx],
                    physicsClientId=self.CLIENT
                )

                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=visual,
                    basePosition=pos,
                    physicsClientId=self.CLIENT
                )

            elif obstacle_type == "wall":

                collision = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=size,
                    physicsClientId=self.CLIENT
                )

                visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=size,
                    rgbaColor=self.colors[color_idx],
                    physicsClientId=self.CLIENT
                )

                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=visual,
                    basePosition=pos,
                    physicsClientId=self.CLIENT
                )

        self._addGoals()
        self._colorDrones()

    def _computeObs(self):

        obs = []

        for i in range(self.NUM_DRONES):

            state = self._getDroneStateVector(i)

            pos = state[0:3]
            vel = state[10:13]

            attitude = state[7:10]
            ang_vel = state[13:16]

            # -------------------------------
            # Relative goal position
            # -------------------------------
            goal_rel = self.goals[i] - pos
            goal_rel = goal_rel / 5.0

            # -------------------------------
            # Relative other drone position
            # -------------------------------
            other = self._getDroneStateVector(1 - i)[0:3]
            other_rel = other - pos
            other_rel = other_rel / 5.0

            # -------------------------------
            # Obstacle distances
            # -------------------------------
            obstacle_distances = []

            for obstacle, size, _, obstacle_type in self.obstacles:

                if obstacle_type == "wall":
                    dist = self._distance_to_wall(pos, obstacle, size)
                else:
                    dist = np.linalg.norm(pos - np.array(obstacle))

                obstacle_distances.append(dist)

            obstacle_distances = np.array(obstacle_distances, dtype=np.float32)

            # normalize obstacle distances
            obstacle_distances = np.clip(obstacle_distances / 5.0, 0, 1)

            # -------------------------------
            # Normalize velocity
            # -------------------------------
            vel = vel / 3.0

            # -------------------------------
            # Normalize angular velocity
            # -------------------------------
            ang_vel = ang_vel / 10.0

            # -------------------------------
            # Attitude normalization
            # -------------------------------
            attitude = attitude / np.pi

            # -------------------------------
            # Final observation vector
            # -------------------------------
            drone_obs = np.concatenate([
                goal_rel,
                vel,
                attitude,
                ang_vel,
                other_rel,
                obstacle_distances
            ])

            obs.append(drone_obs)

        return np.array(obs, dtype=np.float32).flatten()

    def _computeReward(self):
        rewards = []

        for i in range(self.NUM_DRONES):

            state = self._getDroneStateVector(i)

            pos = state[0:3]
            vel = state[10:13]

            other = self._getDroneStateVector(1 - i)[0:3]

            reward = -0.01

            # -------------------------------
            # Distance to goal
            # -------------------------------
            current_dist = np.linalg.norm(pos - self.goals[i])
            delta = self.prev_goal_dist[i] - current_dist

            if delta > 0:
                reward += 3.0 * delta
            else:
                reward += 5.0 * delta

            reward += 1.5 / (current_dist + 0.2)

            self.prev_goal_dist[i] = current_dist

            # -------------------------------
            # Velocity alignment
            # -------------------------------
            goal_dir = self.goals[i] - pos
            goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-6)

            vel_alignment = np.dot(vel, goal_dir)

            reward += 1.5 * vel_alignment

            # -------------------------------
            # Speed penalty
            # -------------------------------
            speed = np.linalg.norm(vel)

            if speed > 0.8:
                reward -= speed * 2.0

            # -------------------------------
            # Altitude stabilization
            # -------------------------------
            z_error = abs(pos[2] - self.goals[i][2])

            reward += 0.8 / (z_error + 0.1)

            if pos[2] < 0.7:
                reward -= (0.7 - pos[2]) * 20

            if 0.9 < pos[2] < 1.3:
                reward += 2

            # -------------------------------
            # Strong attitude stabilization
            # -------------------------------
            roll = abs(state[7])
            pitch = abs(state[8])

            reward -= 2.0 * roll
            reward -= 2.0 * pitch

            # -------------------------------
            # Angular velocity penalty
            # -------------------------------
            ang_vel = np.linalg.norm(state[13:16])

            if ang_vel > 1.5:
                reward -= ang_vel * 2.0

            # -------------------------------
            # Obstacle avoidance
            # -------------------------------
            min_obstacle_dist = np.inf

            for obstacle, size, _, obstacle_type in self.obstacles:

                if obstacle_type == "wall":
                    d = self._distance_to_wall(pos, obstacle, size)
                else:
                    d = np.linalg.norm(pos - np.array(obstacle))

                if d < min_obstacle_dist:
                    min_obstacle_dist = d

                if d < 1.5:
                    reward -= (1.5 - d) * 10

                    if speed > 0.5:
                        reward -= speed * 3

            # -------------------------------
            # Drone separation
            # -------------------------------
            d_drone = np.linalg.norm(pos - other)

            if d_drone < 1.5:
                reward -= (1.5 - d_drone) * 10

            # -------------------------------
            # Goal bonus progressive
            # -------------------------------
            if current_dist < 0.5:
                reward += 5

            if current_dist < 0.3:
                reward += 10

            if current_dist < 0.15:
                reward += 20

            rewards.append(reward)

        return np.mean(rewards)

    def _computeTerminated(self):

        # -------------------------------
        # Goal reached by all drones
        # -------------------------------
        goal_done = all(
            np.linalg.norm(self._getDroneStateVector(i)[0:3] - self.goals[i]) < 0.2
            for i in range(self.NUM_DRONES)
        )

        if goal_done:
            return True

        # -------------------------------
        # Per-drone safety checks
        # -------------------------------
        for i in range(self.NUM_DRONES):

            state = self._getDroneStateVector(i)

            pos = state[0:3]

            roll = abs(state[7])
            pitch = abs(state[8])

            # excessive inclination only if really unstable
            if roll > 1.3 or pitch > 1.3:
                return True

            # floor collision
            if pos[2] < 0.08:
                return True

            # obstacle collision
            for obstacle, size, _, obstacle_type in self.obstacles:

                if obstacle_type == "wall":
                    d = self._distance_to_wall(pos, obstacle, size)
                else:
                    d = np.linalg.norm(pos - np.array(obstacle))

                if d < 0.08:
                    return True

        # -------------------------------
        # Drone-drone collision
        # -------------------------------
        d_drone = np.linalg.norm(
            self._getDroneStateVector(0)[0:3] -
            self._getDroneStateVector(1)[0:3]
        )

        if d_drone < 0.08:
            return True

        return False

    def _computeTruncated(self):

        return self.step_counter >= self.max_steps

    def _computeInfo(self):
        return {}

    def _distance_to_wall(self, pos, center, half_extents):

        dx = max(abs(pos[0] - center[0]) - half_extents[0], 0)
        dy = max(abs(pos[1] - center[1]) - half_extents[1], 0)
        dz = max(abs(pos[2] - center[2]) - half_extents[2], 0)

        return np.linalg.norm([dx, dy, dz])

    def _preprocessAction(self, action):
        rpm = np.zeros((self.NUM_DRONES, 4))

        for i in range(self.NUM_DRONES):

            ax = np.clip(action[i*3 + 0], -1, 1)
            ay = np.clip(action[i*3 + 1], -1, 1)
            az = np.clip(action[i*3 + 2], -1, 1)

            hover = self.HOVER_RPM

            thrust_delta = 0.05 * self.HOVER_RPM
            tilt_delta = 0.01 * self.HOVER_RPM

            thrust = hover + az * thrust_delta

            thrust = np.clip(
                thrust,
                0.95 * self.HOVER_RPM,
                1.05 * self.HOVER_RPM
            )

            roll = ax * tilt_delta
            pitch = ay * tilt_delta

            rpm[i] = np.array([
                thrust + roll - pitch,
                thrust - roll - pitch,
                thrust - roll + pitch,
                thrust + roll + pitch
            ])

        return rpm

    def step(self, action):

        self.step_counter += 1
        self.last_action = action.copy()

        obs, reward, terminated, truncated, info = super().step(action)

        self.episode_reward += float(reward)

        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.step_counter
            }

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.step_counter = 0
        self.prev_action = np.zeros(self.NUM_DRONES * 3)

        self.episode_reward = 0.0

        obs, info = super().reset(seed=seed, options=options)

        for i in range(self.NUM_DRONES):
            pos = self._getDroneStateVector(i)[0:3]
            self.prev_goal_dist[i] = np.linalg.norm(pos - self.goals[i])

        return obs.astype(np.float32), info

    def _addGoals(self):

        goal_colors = [
            [1.0, 0.55, 0.0, 1],
            [0.65, 0.35, 0.85, 1]
        ]

        labels = ["G1", "G2"]

        for i, goal in enumerate(self.goals):

            collision_sphere = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=0.25,
                physicsClientId=self.CLIENT
            )

            visual_sphere = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.25,
                rgbaColor=goal_colors[i],
                physicsClientId=self.CLIENT
            )

            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_sphere,
                baseVisualShapeIndex=visual_sphere,
                basePosition=goal,
                physicsClientId=self.CLIENT
            )

            collision_cyl = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=0.05,
                height=1.0,
                physicsClientId=self.CLIENT
            )

            visual_cyl = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.05,
                length=1.0,
                rgbaColor=goal_colors[i],
                physicsClientId=self.CLIENT
            )

            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_cyl,
                baseVisualShapeIndex=visual_cyl,
                basePosition=[goal[0], goal[1], goal[2] - 0.5],
                physicsClientId=self.CLIENT
            )

            p.addUserDebugText(
                text=labels[i],
                textPosition=[goal[0], goal[1], goal[2] + 0.4],
                textColorRGB=goal_colors[i][:3],
                textSize=1.5,
                physicsClientId=self.CLIENT
            )

    def _colorDrones(self):

        drone_colors = [
            [1.0, 0.55, 0.0, 1],
            [0.65, 0.35, 0.85, 1]
        ]

        for i in range(self.NUM_DRONES):

            for link in range(-1, 5):

                p.changeVisualShape(
                    self.DRONE_IDS[i],
                    link,
                    rgbaColor=drone_colors[i],
                    physicsClientId=self.CLIENT
                )