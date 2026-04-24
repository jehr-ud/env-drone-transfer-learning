import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, Physics, DroneModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class SingleDroneEnv(BaseRLAviary):

    def __init__(
            self,
            gui=False,
            obstacles = [
                # Position, Radius_RL, Size_PyBullet, Type
                # CUBES (Radius = Size * 0.05)
                ([0, 0, 1.5],     2, 3, "cube"),
                ([0, 1.5, 1.0],   2, 2, "cube"),
                ([0, -1.5, 1.0],  2, 2, "cube"),
                ([1.5, 0.8, 1.0], 2, 2, "cube"),
                ([-1.5, -0.8, 1.0], 2, 2, "cube"),
                ([1.2, -2.0, 1.0], 2, 2, "cube"),
                ([-1.2, 2.0, 1.0], 2, 2, "cube"),

                # CYLINDERS (Fixed radius = 0.25, Size is the height)
                ([2.2, 1.5, 2.0],   0.25, 4, "cylinder"),
                ([-2.2, -1.5, 2.0], 0.25, 2, "cylinder"),
                ([2.0, -1.5, 2.0],  0.25, 4, "cylinder"),
                ([-2.0, 1.5, 2.0],  0.25, 2, "cylinder"),
                ([0.8, 2.5, 2.0],   0.25, 2, "cylinder"),
                ([-0.8, -2.5, 2.0], 0.25, 3, "cylinder")
            ],
            goal=np.array([2.5, 2.5, 1.5])
        ):

        self.NUM_DRONES = 1

        self.goal = goal
        self.obstacles = obstacles

        self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X)]

        self.obstacle_ids = []

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            neighbourhood_radius=10,
            initial_rpys=np.zeros((1, 3)),
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=60,
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
            shape=(25,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # -------------------------------
        # DEFINITIONS
        # -------------------------------
        
        self.MAX_SPEED = 2.0
        self.MAX_DIST = 10.0
        self.MAX_OBS_RADIUS = 0.35 
        self.step_counter = 0
        self.prev_dist = 0.0
        self.episode_reward = 0.0

    def _addObstacles(self):

        for pos, size, color_idx, obstacle_type in self.obstacles:

            if obstacle_type == "cube":
                obstacle_id = p.loadURDF(
                    "cube_small.urdf",
                    pos,
                    globalScaling=size,
                    physicsClientId=self.CLIENT
                )

                # Visual
                p.changeVisualShape(
                    obstacle_id,
                    -1,
                    rgbaColor=[0.8, 0.2, 0.2, 1],
                    physicsClientId=self.CLIENT
                )

                p.changeDynamics(
                    obstacle_id,
                    -1,
                    mass=0,  # estático
                    lateralFriction=1.0,
                    restitution=0.0,
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
                    rgbaColor=[0.2, 0.2, 0.8, 1],
                    physicsClientId=self.CLIENT
                )

                obstacle_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=visual,
                    basePosition=pos,
                    physicsClientId=self.CLIENT
                )

            self.obstacle_ids.append((obstacle_id, pos))


    def _colorDrone(self):
        drone_colors = [
            [1.0, 0.55, 0.0, 1],  # Orange
            [0.65, 0.35, 0.85, 1]   # Purple
        ]

        for i in range(self.NUM_DRONES):
            # Link -1 is the drone's body.
            # Links 0 to 4 are usually the motors/propellers.
            for link in range(-1, 5): 
                p.changeVisualShape(
                    self.DRONE_IDS[i],
                    link,
                    rgbaColor=drone_colors[i % len(drone_colors)], # The percentage avoids errors if there are more drones than colors.
                    physicsClientId=self.CLIENT
                )

    def _addGoal(self):
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.2,
            rgbaColor=[0, 1, 0, 1],
            physicsClientId=self.CLIENT
        )

        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual,
            basePosition=self.goal,
            physicsClientId=self.CLIENT
        )

        p.addUserDebugText(
            text="GOAL",
            textPosition=[self.goal[0], self.goal[1], self.goal[2] + 0.3],
            textColorRGB=[0, 1, 0],
            textSize=1.5,
            physicsClientId=self.CLIENT
        )

    # =========================================================
    # OBSERVATION
    # =========================================================
    def _computeObs(self):

        state = self._getDroneStateVector(0)

        pos = state[0:3]
        vel = state[10:13]
        rpy = state[7:10]
        ang_vel = state[13:16]

        # =========================
        # GOAL
        # =========================
        goal_vec = self.goal - pos
        goal_rel = np.clip(goal_vec / 5.0, -1, 1)

        # =========================
        # VELOCITY
        # =========================
        vel_norm = np.clip(vel / 3.0, -1, 1)

        # =========================
        # ATTITUDE
        # =========================
        rp = np.clip(rpy[0:2] / 0.5, -1, 1)

        # =========================
        # YAW
        # =========================
        yaw = np.array([
            np.sin(rpy[2]),
            np.cos(rpy[2])
        ])

        # =========================
        # ANGULAR VELOCITY
        # =========================
        ang_vel_norm = np.clip(ang_vel / 10.0, -1, 1)

        # =========================
        # DISTANCE + ALIGNMENT
        # =========================
        dist = np.linalg.norm(goal_vec) / self.MAX_DIST

        goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)
        vel_dir = vel / (np.linalg.norm(vel) + 1e-6)
        alignment = np.dot(goal_dir, vel_dir)
        alignment = np.clip(alignment, -1.0, 1.0)

        # =========================
        # OBSTACLES (ONLY TOP 2)
        # =========================
        obs_features = []

        if len(self.obstacles) > 0:

            obs_temp = []

            for obs_data in self.obstacles:
                obs_pos = np.array(obs_data[0])

                rel_vec = obs_pos - pos
                dist_obs = np.linalg.norm(rel_vec) + 1e-6

                obs_temp.append({
                    "rel_vec": rel_vec,
                    "dist": dist_obs,
                    "size": obs_data[1]
                })

            # sort by distances
            obs_temp.sort(key=lambda x: x["dist"])

            # get top 2
            for i in range(2):
                if i < len(obs_temp):

                    rel_vec = obs_temp[i]["rel_vec"]
                    dist_obs = obs_temp[i]["dist"]
                    size_obstacle = obs_temp[i]["size"]

                    rel_norm = np.clip(rel_vec / self.MAX_DIST, -1, 1)
                    dist_norm = np.clip(dist_obs / self.MAX_DIST, 0, 1)

                    obs_features.extend(rel_norm)   # 3
                    obs_features.append(dist_norm)  # 1

                    size_norm = np.clip(size_obstacle / self.MAX_OBS_RADIUS, 0, 1)
                    obs_features.append(size_norm)

                else:
                    # padding
                    obs_features.extend([0.0, 0.0, 0.0, 1.0, 0.0])

        else:
            # no obstacles → padding
            obs_features = [0.0, 0.0, 0.0, 1.0, 0.0] * 2

        obs_features = np.array(obs_features)

        # =========================
        # FINAL OBS
        # =========================
        obs = np.concatenate([
            goal_rel,        # 3
            vel_norm,        # 3
            rp,              # 2
            yaw,             # 2
            ang_vel_norm,    # 3
            [alignment],     # 1
            [dist],          # 1
            obs_features     # 10
        ])

        return np.clip(obs, -1.0, 1.0).astype(np.float32)


    def set_difficulty(self, level):
        if level == 0:
            self.obstacles = []
        elif level == 1:
            self.obstacles = self.obstacles[:2]
        elif level == 2:
            self.obstacles = self.obstacles[:5]
        else:
            pass

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
        # PROGRESS (normalized)
        # -------------------------
        progress = self.prev_dist - dist
        progress_norm = np.clip(progress / (self.MAX_DIST + 1e-6), -1.0, 1.0)

        # -------------------------
        # DIRECTION (alignment)
        # -------------------------
        goal_dir = (self.goal - pos)
        goal_dir /= (np.linalg.norm(goal_dir) + 1e-6)

        if speed < 0.05:
            alignment = 0.0
        else:
            vel_dir = vel / (speed + 1e-6)
            alignment = np.dot(goal_dir, vel_dir)

        alignment_norm = np.clip(alignment, -1.0, 1.0)

        # -------------------------
        # DISTANCE (normalized)
        # -------------------------
        dist_norm = 1.0 - np.clip(dist / (self.MAX_DIST + 1e-6), 0.0, 1.0)

        # -------------------------
        # MOVEMENT
        # -------------------------
        move_norm = -1.0 if speed < 0.05 else 1.0

        # -------------------------
        # GOAL
        # -------------------------
        goal_norm = 1.0 if dist < 0.3 else 0.0

        # -------------------------
        # OBSTACLES
        # -------------------------
        obstacle_penalty = 0.0
        min_dist = float("inf")

        for obs in self.obstacles:
            obs_pos = np.array(obs[0])
            obs_radius = obs[1]

            d = np.linalg.norm(pos - obs_pos) - obs_radius
            min_dist = min(min_dist, d)

        if self.obstacles:
            if min_dist < 0.5:
                obstacle_penalty = - (1.0 - (np.clip(min_dist, 0, 0.5) / 0.5))**2

            if min_dist < 0.15:
                obstacle_penalty += -1.0

        obstacle_norm = np.clip(obstacle_penalty, -1.0, 0.0)

        # -------------------------
        # TIME
        # -------------------------
        time_norm = -1.0  # constante

        # -------------------------
        # WEIGHTS (percentages)
        # -------------------------
        w = {
            "progress": 0.35,
            "alignment": 0.20,
            "distance": 0.8,
            "movement": 0.05,
            "goal": 0.18,
            "time": 0.05,
            "obstacle": 0.10
        }

        # -------------------------
        # FINAL REWARD
        # -------------------------
        reward = (
            w["progress"] * progress_norm
            + w["alignment"] * alignment_norm
            + w["distance"] * dist_norm
            + w["movement"] * move_norm
            + w["goal"] * goal_norm
            + w["time"] * time_norm
            + w["obstacle"] * obstacle_norm
        )

        # update state
        self.prev_dist = dist

        return reward

    # =========================================================
    # TERMINATION (MDP STATES: GOAL OR CRASH)
    # =========================================================
    def _computeTerminated(self):
        """
        Returns True if the agent reaches a terminal state 
        defined by the MDP (Success or Failure).
        """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        roll, pitch = state[7], state[8]
        dist = np.linalg.norm(pos - self.goal)

        # 1. POSITIVE TERMINAL STATE: Goal Reached
        if dist < 0.3:
            print("🎯 TERMINATED: GOAL REACHED (Positive)")
            return True

        # 2. NEGATIVE TERMINAL STATE: Ground Collision
        if pos[2] < 0.05:
            print("💥 TERMINATED: GROUND COLLISION (Negative)")
            return True

        # 3. NEGATIVE TERMINAL STATE: Rollover/Instability
        if abs(roll) > 1.5 or abs(pitch) > 1.5:
            print("🙃 TERMINATED: DRONE FLIPPED (Negative)")
            return True
        
        # 4. NEGATIVE TERMINAL STATE: Obstacle Collision
        for body_id, obs_pos in self.obstacle_ids:
            aabb_min, aabb_max = p.getAABB(body_id, physicsClientId=self.CLIENT)

            aabb_min = np.array(aabb_min)
            aabb_max = np.array(aabb_max)

            # Check: if the drone is inside the box
            margin = 0.15  # size of the dron aprox
            if np.all(pos >= (aabb_min - margin)) and np.all(pos <= (aabb_max + margin)):
                print(f"💥 TERMINATED: OBSTACLE COLLISION at {obs_pos}")
                return True

        return False

    # =========================================================
    # TRUNCATION (OUTSIDE MDP SCOPE)
    # =========================================================
    def _computeTruncated(self):
        """
        Returns True if the episode ends prematurely due to 
        external constraints (Time or Out of Bounds).
        """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        dist_to_goal = np.linalg.norm(pos - self.goal)

        # 1. TIME LIMIT: Maximum episode steps
        if self.step_counter > (2500 * len(self.obstacles)) :
            print("🕒 TRUNCATED: TIME LIMIT REACHED")
            return True

        # 2. OUT OF BOUNDS: Agent wandered too far away
        if dist_to_goal > self.MAX_DIST:
            print(f"🛰️ TRUNCATED: OUT OF BOUNDS ({dist_to_goal:.2f}m)")
            return True

        return False

    # =========================================================
    # ACTION
    # =========================================================
    def _preprocessAction(self, action):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        quat = state[3:7]
        vel = state[10:13]
        
        # 1. Map action to velocity (Action 0 to 3: VX, VY, VZ, Magnitude)
        # action[0:3] is the normalized direction.
        direction = action[0:3]
        speed_scale = (action[3] + 1) / 2 # from 0 to 1
        
        target_vel = direction * self.MAX_SPEED * speed_scale
        
        # 2. Limiting speed to prevent rollover (Crucial)
        target_vel = np.clip(target_vel, -1.0, 1.0) 

        # 3. Calculate soft target position (where it should be in the next frame)
        target_pos = pos + target_vel * self.CTRL_TIMESTEP

        # 4. PID Control
        rpm, _, _ = self.ctrl[0].computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            cur_pos=pos,
            cur_quat=quat,
            cur_vel=vel,
            cur_ang_vel=state[13:16],
            target_pos=target_pos,
            target_rpy=np.array([0, 0, state[9]]), # Maintain current yaw
            target_vel=target_vel
        )

        return np.array([rpm])

    # =========================================================
    # STEP
    # =========================================================
    def step(self, action):
        self.step_counter += 1
        obs, reward, term, trunc, info = super().step(action)
        self.episode_reward += reward
        return obs, reward, term, trunc, info

    # =========================================================
    # RESET
    # =========================================================
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        self.obstacle_ids = []

        self._addGoal()
        self._addObstacles()
        self._colorDrone()

        self.step_counter = 0

        state = self._getDroneStateVector(0)
        pos = state[0:3]

        self.prev_dist = np.linalg.norm(self.goal - pos)
        self.episode_reward = 0.0

        return self._computeObs(), {}

    def _computeInfo(self):
        return {}
    

    def _highlightObstacle(self, target_pos, color=[1,1,0,1]):
        target_pos = np.array(target_pos)

        for body_id, pos in self.obstacle_ids:
            if np.linalg.norm(np.array(pos) - target_pos) < 0.2:
                p.changeVisualShape(body_id, -1, rgbaColor=color, physicsClientId=self.CLIENT)