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

            speed = np.linalg.norm(vel)

            if delta >= 0:
                reward += 5 * delta
            else:
                reward += 8 * delta

            reward += 1 / (current_dist + 0.3)

            # global speed penalty
            if speed > 1.0:
                reward -= speed * 2

            # -------------------------------
            # Velocity alignment
            # -------------------------------
            goal_dir = self.goals[i] - pos
            goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-6)

            vel_alignment = np.dot(vel, goal_dir)

            reward += 2 * vel_alignment

            # -------------------------------
            # Altitude alignment
            # -------------------------------
            z_error = abs(pos[2] - self.goals[i][2])

            reward += 0.5 / (z_error + 0.05)

            if z_error < 0.1:
                reward += 10

            self.prev_goal_dist[i] = current_dist

            # -------------------------------
            # Fall penalization / hover reward
            # -------------------------------
            if pos[2] < 0.6:
                reward -= (0.6 - pos[2]) * 20

            if abs(pos[2] - self.goals[i][2]) < 0.2:
                reward += 2

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
                    reward -= (1.5 - d) * 8

            # speed penalty only if obstacle nearby
            if min_obstacle_dist < 1.5:
                reward -= speed * 3

            # -------------------------------
            # Drone movement brusco
            # -------------------------------
            action_change = np.linalg.norm(self.last_action - self.prev_action)
            reward -= 0.5 * action_change

            # -------------------------------
            # Drone separation
            # -------------------------------
            d_drone = np.linalg.norm(pos - other)

            if d_drone < 1.5:
                reward -= (1.5 - d_drone) * 10

            # -------------------------------
            # Stability penalty
            # -------------------------------
            roll = abs(state[7])
            pitch_angle = abs(state[8])

            reward -= 0.5 * (roll + pitch_angle)

            # angular velocity penalty
            ang_vel = np.linalg.norm(state[13:16])

            reward -= 0.1 * ang_vel

            # -------------------------------
            # Goal bonus
            # -------------------------------
            if current_dist < 0.5:
                reward += 5

            if current_dist < 0.3:
                reward += 15

            if current_dist < 0.15:
                reward += 25

            rewards.append(reward)

        self.prev_action = self.last_action.copy()
        return np.mean(rewards)

    def _computeTerminated(self):

        # goal reached by all drones
        goal_done = all(
            np.linalg.norm(self._getDroneStateVector(i)[0:3] - self.goals[i]) < 0.2
            for i in range(self.NUM_DRONES)
        )

        if goal_done:
            return True

        # obstacle collision
        for i in range(self.NUM_DRONES):

            pos = self._getDroneStateVector(i)[0:3]

            if pos[2] < 0.1:
                return True

            for obstacle, size, _, obstacle_type in self.obstacles:

                if obstacle_type == "wall":
                    d = self._distance_to_wall(pos, obstacle, size)
                else:
                    d = np.linalg.norm(pos - np.array(obstacle))

                if d < 0.10:
                    return True

        # drone collision
        d_drone = np.linalg.norm(
            self._getDroneStateVector(0)[0:3] -
            self._getDroneStateVector(1)[0:3]
        )

        if d_drone < 0.10:
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

            ax = action[i*3 + 0]
            ay = action[i*3 + 1]
            az = action[i*3 + 2]

            hover = self.HOVER_RPM

            delta = 0.15 * self.HOVER_RPM

            thrust = hover + az * delta

            thrust = np.clip(
                thrust,
                0.8 * self.HOVER_RPM,
                1.2 * self.HOVER_RPM
            )

            roll = ax * 0.05 * self.HOVER_RPM
            pitch = ay * 0.05 * self.HOVER_RPM

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

        return super().step(action)

    def reset(self, seed=None, options=None):
        self.step_counter = 0
        self.prev_action = np.zeros(self.NUM_DRONES * 3)

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