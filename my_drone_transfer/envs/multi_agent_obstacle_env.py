import numpy as np
import csv
import os
import pybullet as p

from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    ObservationType,
    ActionType,
    Physics,
    DroneModel
)
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class MultiAgentObstacleEnv(BaseRLAviary):

    def __init__(
        self,
        obs=ObservationType.KIN,
        act=ActionType.VEL,
        gui=False,
        with_obstacles=False,
        debug=False
    ):
        self.colors = [
            [0.86, 0.37, 0.34, 1], [0.35, 0.70, 0.90, 1],
            [0.50, 0.78, 0.50, 1], [0.95, 0.77, 0.35, 1],
            [0.72, 0.56, 0.87, 1], [0.60, 0.60, 0.60, 1]
        ]

        # goals
        self.goals = np.array([
            [2.5, 3.5, 1.8],
            [-2.5, -3.5, 1.8]
        ])

        self.obstacles = []

        self.with_obstacles = with_obstacles

        if with_obstacles:
            self.obstacles = [
                # Position, Radius_RL, Size_PyBullet, Type
                # CUBES (Radius = Size * 0.05)
                ([0, 0, 1.5],     0.15, 3, "cube"),
                ([0, 1.5, 1.0],   0.10, 2, "cube"),
                ([0, -1.5, 1.0],  0.10, 2, "cube"),
                ([1.5, 0.8, 1.0], 0.10, 2, "cube"),
                ([-1.5, -0.8, 1.0], 0.10, 2, "cube"),
                ([1.2, -2.0, 1.0], 0.10, 2, "cube"),
                ([-1.2, 2.0, 1.0], 0.10, 2, "cube"),

                # CYLINDERS (Fixed radius = 0.25, Size is the height)
                ([2.2, 1.5, 2.0],   0.25, 4, "cylinder"),
                ([-2.2, -1.5, 2.0], 0.25, 2, "cylinder"),
                ([2.0, -1.5, 2.0],  0.25, 4, "cylinder"),
                ([-2.0, 1.5, 2.0],  0.25, 2, "cylinder"),
                ([0.8, 2.5, 2.0],   0.25, 2, "cylinder"),
                ([-0.8, -2.5, 2.0], 0.25, 3, "cylinder")
            ]

        self.NUM_DRONES = 1
        self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(self.NUM_DRONES)]

        self.debug = debug
        self.reward_log_path = "reward_debug.csv"

        # debug file
        if not os.path.exists(self.reward_log_path):
            with open(self.reward_log_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step", "drone", "dist",
                    "progress", "distance_reward",
                    "speed_penalty", "bonus",
                    "time_penalty", "total"
                ])


        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=self.NUM_DRONES,
            neighbourhood_radius=10,
            initial_rpys=np.zeros((self.NUM_DRONES, 3)),
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=48,
            gui=gui,
            obs=obs,
            act=act
        )

        # Setting limits and states
        self.EPISODE_LEN_SEC = 120
        self.step_counter = 0
        self.episode_reward = 0.0
        self.reached = [False] * self.NUM_DRONES
        self.prev_goal_dist = np.zeros(self.NUM_DRONES)
        
        # Observaions spaces (Gymnasium)
        self.observation_space = spaces.Dict({
            "goal": spaces.Box(-1, 1, shape=(self.NUM_DRONES, 3), dtype=np.float32),
            "velocity": spaces.Box(-1, 1, shape=(self.NUM_DRONES, 3), dtype=np.float32),
            "attitude": spaces.Box(-1, 1, shape=(self.NUM_DRONES, 2), dtype=np.float32),
            "yaw": spaces.Box(-1, 1, shape=(self.NUM_DRONES, 2), dtype=np.float32),
            "angular_velocity": spaces.Box(-1, 1, shape=(self.NUM_DRONES, 3), dtype=np.float32),
            "other": spaces.Box(-1, 1, shape=(self.NUM_DRONES, 3), dtype=np.float32),
            "obstacles": spaces.Box(-1, 1, shape=(self.NUM_DRONES, 12), dtype=np.float32)
        })

        radius = 1.5  # maximum 2 at the start

        noise = np.random.uniform(-radius, radius, size=(1, 3))
        self.INIT_XYZS = self.goals[:1] + noise
        self.INIT_XYZS[:, 2] = 1.2

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.NUM_DRONES * 4,), dtype=np.float32
        )

        self._last_reward_info = None

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
        goal_list = []
        vel_list = []
        rp_list = []
        yaw_list = []
        ang_vel_list = []
        other_list = []
        obs_list = []

        all_states = [self._getDroneStateVector(i) for i in range(self.NUM_DRONES)]

        for i in range(self.NUM_DRONES):
            state = all_states[i]
            pos = state[0:3]
            rpy = state[7:10]
            vel = state[10:13]
            ang_vel = state[13:16]

            # 1. Relative Goal (Normalized)
            goal_rel = (self.goals[i] - pos) / 5.0
            goal_list.append(np.clip(goal_rel, -1, 1))

            # 2. Linear Speed
            vel_list.append(np.clip(vel / 3.0, -1, 1))

            # 3. Attitude (Roll and Pitch)
            rp_list.append(np.clip(rpy[0:2] / 0.5, -1, 1))

            # 4. Yaw (Sin/Cos representation to avoid discontinuity at 2pi)
            yaw_list.append([np.sin(rpy[2]), np.cos(rpy[2])])

            # 5. Angular Velocity
            ang_vel_list.append(np.clip(ang_vel / 10.0, -1, 1))

            # 6. Relationship with the other drone
            if self.NUM_DRONES > 1:
                other_idx = 1 - i 
                other_pos = all_states[other_idx][0:3]
                other_rel = (other_pos - pos) / 5.0
                other_list.append(np.clip(other_rel, -1, 1))
            else:
                other_list.append(np.zeros(3))

            # 7. Obstacles (The 3 closest)

            current_drone_obs_flat = []
            if len(self.obstacles) > 0:
                obs_temp_list = []
                for obs_data in self.obstacles:
                    obs_pos = np.array(obs_data[0])
                    size = obs_data[1]
                    
                    # Relative vector and Euclidean distance to the center
                    r_vector = (obs_pos - pos)
                    dist_to_center = np.linalg.norm(r_vector)
                    
                    # We save the info to sort by actual proximity
                    obs_temp_list.append({
                        "rel_vec": np.clip(r_vector / 5.0, -1, 1),     # Normalized to the viewing range (5m)
                        "size": np.clip(size / 5.0, 0, 1),            # Standardized size
                        "dist": dist_to_center
                    })

                # Sort by actual distance to center
                obs_temp_list.sort(key=lambda x: x["dist"])

                for j in range(3):
                    if j < len(obs_temp_list):
                        item = obs_temp_list[j]
                        # We add x, y, z (relative) + the size
                        current_drone_obs_flat.extend(item["rel_vec"]) 
                        current_drone_obs_flat.append(item["size"])
                    else:
                        # Padding: Far position (1,1,1) and size 0
                        current_drone_obs_flat.extend([1.0, 1.0, 1.0, 0.0])
            else:
                current_drone_obs_flat = [0.0] * 12 # Padding if there are no obstacles

            obs_list.append(current_drone_obs_flat)

        # Return the dictionary with shapes (NUM_DRONES, N)
        return {
            "goal": np.array(goal_list, dtype=np.float32),
            "velocity": np.array(vel_list, dtype=np.float32),
            "attitude": np.array(rp_list, dtype=np.float32),
            "yaw": np.array(yaw_list, dtype=np.float32),
            "angular_velocity": np.array(ang_vel_list, dtype=np.float32),
            "other": np.array(other_list, dtype=np.float32),
            "obstacles": np.array(obs_list, dtype=np.float32),
        }

    def _is_pos_safe(self, pos, margin=0.3):
        """
        Check if a position is at a safe distance from all obstacles.
        """
        for obs_pos_list, size, _, obs_type in self.obstacles:
            obs_pos = np.array(obs_pos_list)
            dist = np.linalg.norm(pos - obs_pos)
            
            if obs_type == "cube":
                # Approximation by surrounding sphere radius for cubes
                # size is the global scaling of the urdf cube_small
                safe_dist = (size * 0.1) + margin 
            elif obs_type == "cylinder":
                # The radius of the cylinder is 0.25 according to the definition of obstacles.
                safe_dist = 0.25 + margin
            else:
                safe_dist = margin
                
            if dist < safe_dist:
                return False
        return True

    def _computeReward(self):

        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

        total_reward = 0
        reward_info = []

        for i in range(self.NUM_DRONES):

            pos = states[i][0:3]
            vel = states[i][10:13]

            dist = np.linalg.norm(self.goals[i] - pos)
            speed = np.linalg.norm(vel)

            # =====================================================
            # 1. PROGRESS (VERY IMPORTANT - DOMINANT)
            # =====================================================
            progress = self.prev_goal_dist[i] - dist
            progress_r = 20.0 * progress

            # =====================================================
            # 2. DISTANCE SHAPING
            # =====================================================
            distance_r = 2.0 / (1.0 + dist)

            # =====================================================
            # 3. EXTRA REWARD NEAR THE FINISH LINE
            # =====================================================
            goal_shaping = 0.0
            if dist < 1.0:
                goal_shaping = 2.0 * (1.0 - dist)

            # =====================================================
            # 4. CORRECT DIRECTION (ANTI-ZIGZAG)
            # =====================================================
            goal_dir = self.goals[i] - pos
            goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-6)

            vel_dir = vel / (np.linalg.norm(vel) + 1e-6)

            alignment = np.dot(goal_dir, vel_dir)
            direction_bonus = 0.5 * alignment  # [-0.5, 0.5]

            # =====================================================
            # 5. PENALIZE STILLNESS
            # =====================================================
            movement_penalty = 0.0
            if speed < 0.05:
                movement_penalty = -0.5

            # =====================================================
            # 6. SPEED CONTROL NEAR THE FINISH LINE
            # =====================================================
            speed_penalty = 0.0
            if dist < 1.0:
                speed_penalty = -0.1 * speed

            # =====================================================
            # 7. BONUS FOR ARRIVING
            # =====================================================
            bonus = 0.0
            if dist < 0.3:
                bonus = 10.0
                self.reached[i] = True

            # =====================================================
            # 8. OBSTACLES (SMOOTH AND STABLE)
            # =====================================================
            obstacle_penalty = 0.0
            min_dist = float("inf")

            for obs in self.obstacles:
                obs_pos = np.array(obs[0])
                obs_radius = obs[1]
                
                dist_to_center = np.linalg.norm(pos - obs_pos)
                d_to_surface = dist_to_center - obs_radius
                
                if d_to_surface < min_dist:
                    min_dist = d_to_surface

            # Penalty if you approach within 0.5m of the surface
            if min_dist < 0.5:
                # We use a quadratic function so that it's smooth at the beginning and strong at the end
                obstacle_penalty = -2.0 * (1.0 - (np.clip(min_dist, 0, 0.5) / 0.5))**2

            # =====================================================
            # 9. HEAVY COLLISION
            # =====================================================
            collision_penalty = 0.0
            if min_dist < 0.15:
                collision_penalty = -10.0

            # =====================================================
            # 10. ALTURA (ANTI-CRASH)
            # =====================================================
            height_penalty = 0.0
            if pos[2] < 0.2:
                height_penalty = -2.0

            # =====================================================
            # 11. TIME PENALTY
            # =====================================================
            time_penalty = -0.01

            # =====================================================
            # TOTAL
            # =====================================================
            reward = (
                progress_r
                + distance_r
                + goal_shaping
                + direction_bonus
                + movement_penalty
                + speed_penalty
                + bonus
                + obstacle_penalty
                + collision_penalty
                + height_penalty
                + time_penalty
            )

            total_reward += reward
            self.prev_goal_dist[i] = dist

            reward_info.append({
                "drone": i,
                "dist": dist,
                "progress": progress_r,
                "distance_reward": distance_r,
                "goal_shaping": goal_shaping,
                "direction_bonus": direction_bonus,
                "movement_penalty": movement_penalty,
                "speed_penalty": speed_penalty,
                "bonus": bonus,
                "obstacle_penalty": obstacle_penalty,
                "collision_penalty": collision_penalty,
                "height_penalty": height_penalty,
                "total": reward
            })

        self._last_reward_info = reward_info

        return total_reward / self.NUM_DRONES

    def _computeTerminated(self):

        all_drones_on_goal = True

        for i in range(self.NUM_DRONES):

            state = self._getDroneStateVector(i)
            pos = state[0:3]
            roll, pitch = state[7], state[8]

            dist = np.linalg.norm(pos - self.goals[i])

            # -------------------------
            # 1. CHECK GOAL
            # -------------------------
            if dist < 0.3:
                self.reached[i] = True
            else:
                all_drones_on_goal = False

            # -------------------------
            # 2. DEBUG STATE
            # -------------------------
            # (only every 50 steps to avoid saturation)
            if self.step_counter % 50 == 0:
                print(f"[DEBUG] Drone {i} | pos={pos} | dist={dist:.2f} | z={pos[2]:.2f} | roll={roll:.2f} | pitch={pitch:.2f}")

            # -------------------------
            # 3. CRASH: HEIGHT
            # -------------------------
            if pos[2] < 0.05:
                print(f"[TERMINATED] Drone {i} crashed (low altitude): z={pos[2]:.3f}")
                return True

            # -------------------------
            # 4. CRASH: ORIENTATION
            # -------------------------
            if abs(roll) > 1.2 or abs(pitch) > 1.2:
                print(f"[TERMINATED] Drone {i} unstable: roll={roll:.2f}, pitch={pitch:.2f}")
                return True

            # -------------------------
            # 5. OBSTACLES
            # -------------------------
            for obs in self.obstacles:
                obs_pos = np.array(obs[0])
                d_obs = np.linalg.norm(pos - obs_pos)

                if d_obs < 0.2:
                    print(f"[TERMINATED] Drone {i} hit obstacle at distance {d_obs:.2f}")
                    return True

        # -------------------------
        # 6. SUCCESS
        # -------------------------
        if all_drones_on_goal:
            print(f"--- ✅ SUCCESS at step {self.step_counter} ---")
            return True

        # -------------------------
        # 7. DRONE COLLISION
        # -------------------------
        if self.NUM_DRONES > 1:
            d = np.linalg.norm(
                self._getDroneStateVector(0)[0:3] -
                self._getDroneStateVector(1)[0:3]
            )

            if d < 0.15:
                print(f"[TERMINATED] Drone collision: distance={d:.3f}")
                return True

        return False

    def _computeTruncated(self):
        # -------------------------
        # 1. OUT OF BOUNDS
        # -------------------------
        for i in range(self.NUM_DRONES):

            pos = self._getDroneStateVector(i)[0:3]

            if np.any(np.abs(pos) > 15.0):
                print(f"[TRUNCATED] Drone {i} out of bounds: pos={pos}")
                return True

            if self.step_counter % 50 == 0:
                print(f"[DEBUG-TRUNC] Drone {i} pos={pos}")

        # -------------------------
        # 2. TIME LIMIT
        # -------------------------
        elapsed_time = self.step_counter / self.PYB_FREQ

        if elapsed_time > self.EPISODE_LEN_SEC:
            print(f"[TRUNCATED] Time limit reached: {elapsed_time:.2f}s / {self.EPISODE_LEN_SEC}s")
            return True

        if self.step_counter % 100 == 0:
            print(f"[DEBUG-TIME] step={self.step_counter} time={elapsed_time:.2f}s")

        return False
    
    def _computeInfo(self):
        return {
            "is_success": int(all(self.reached)),
            "reward_breakdown": getattr(self, "_last_reward_info", None)
        }

    def _preprocessAction(self, action):
        action = action.reshape(self.NUM_DRONES, 4)
        rpm = np.zeros((self.NUM_DRONES, 4))

        for k in range(self.NUM_DRONES):
            state = self._getDroneStateVector(k)
            pos, vel = state[0:3], state[10:13]
            
            # Direction and climbing speed
            target_v = np.clip(action[k, 0:3], -1, 1)
            speed_factor = (action[k, 3] + 1) / 2 # [0, 1]
            target_vel = (target_v * 0.5) * speed_factor 

            target_pos = pos + target_vel * self.CTRL_TIMESTEP

            rpm_k, _, _ = self.ctrl[k].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=pos,
                cur_quat=state[3:7],
                cur_vel=vel,
                cur_ang_vel=state[13:16],
                target_pos=target_pos,
                target_rpy=np.array([0, 0, state[9]]), # Maintain current yaw
                target_vel=target_vel
            )
            rpm[k, :] = rpm_k
        return rpm

    def step(self, action):
        self.step_counter += 1
        obs, reward, term, trunc, info = super().step(action)
        self.episode_reward += reward
        return obs, reward, term, trunc, info

    def reset(self, seed=None, options=None):
        self.step_counter = 0
        self.episode_reward = 0.0
        noise = np.random.uniform(-0.2, 0.2, size=(self.NUM_DRONES, 3))
        self.INIT_XYZS = np.array([[0, -1.0, 1.2], [0, 1.0, 1.2]]) + noise
        self.reached = [False] * self.NUM_DRONES

        
        new_initial_xyzs = []
    
        for i in range(self.NUM_DRONES):
            found_safe_pos = False
            attempts = 0
            
            while not found_safe_pos and attempts < 100:
                # Generate random position within a range (example: x and y between -3 and 3)
                random_pos = np.array([
                    np.random.uniform(-3, 3),
                    np.random.uniform(-3, 3),
                    1.2 # Fixed takeoff height
                ])
                
                if self._is_pos_safe(random_pos):
                    new_initial_xyzs.append(random_pos)
                    found_safe_pos = True
                attempts += 1
                
            # If it fails 100 times (map is very full), force a known position
            if not found_safe_pos:
                new_initial_xyzs.append(np.array([0, -2.0, 1.2]))

        self.INIT_XYZS = np.array(new_initial_xyzs)

        obs, info = super().reset(seed=seed, options=options)

        for i in range(self.NUM_DRONES):
            pos = self._getDroneStateVector(i)[0:3]
            self.prev_goal_dist[i] = np.linalg.norm(self.goals[i] - pos)

        return obs, info
    
    def _addGoals(self):

        goal_colors = [
            [1.0, 0.55, 0.0, 1],
            [0.65, 0.35, 0.85, 1]
        ]

        labels = ["G1", "G2"]

        for i, goal in enumerate(self.goals):
            visual_sphere = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.15, 
                rgbaColor=goal_colors[i],
                physicsClientId=self.CLIENT
            )
            
            # 2. We created the body in the world WITHOUT collision form
            p.createMultiBody(
                baseMass=0, # Zero mass so that it is static
                baseCollisionShapeIndex=-1, #  -1 means NO COLLISION
                baseVisualShapeIndex=visual_sphere,
                basePosition=goal,
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