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
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class MultiAgentObstacleEnv(BaseRLAviary, MultiAgentEnv):

    def __init__(
        self,
        config=None
    ):  
        #params
        config = config or {}
        gui = config.get("gui", False)
        with_obstacles = config.get("with_obstacles", False)
        debug = config.get("debug", False)
        obs = ObservationType.KIN
        act = ActionType.VEL
        
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

        self.NUM_DRONES = 2
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

        MultiAgentEnv.__init__(self)
        
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

        self.agents = [f"drone_{i}" for i in range(self.NUM_DRONES)]
        self.possible_agents = self.agents.copy()

        # Setting limits and states
        self.EPISODE_LEN_SEC = 120
        self.step_counter = 0
        self.episode_reward = 0.0
        self.reached = [False] * self.NUM_DRONES
        #self.prev_goal_dist = np.zeros(self.NUM_DRONES)
        
        # Observaions spaces (Gymnasium)

        #3 (goal)
        #+ 3 (vel)
        #+ 2 (rp)
        #+ 2 (yaw)
        #+ 3 (ang_vel)
        #+ 3 (closest_vec)
        #+ 1 (dist)
        #+ 15 (obstacles)
        #= 33
        # 2 drones × (pos+vel=6) = 12
        # 2 drones × (pos+vel=6) = 12
        self.observation_space = {
            f"drone_{i}": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(45,),
                dtype=np.float32
            )
            for i in range(self.NUM_DRONES)
        }

        self.action_space = {
            f"drone_{i}": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32
            )
            for i in range(self.NUM_DRONES)
        }

        self._last_reward_info = None

        # -------------------------------
        # CONSTANTES FÍSICAS CALCULADAS
        # -------------------------------
        max_goal_dist = np.max(np.linalg.norm(self.goals, axis=1))
        self.MAX_DIST = max_goal_dist * 1.2

        self.MAX_SPEED = 2.5     # Velocidad máxima del CF2X
        
        # MAX_SPEED (2.5) * (1 / ctrl_freq (48)) = 0.052
        dt = 1.0 / self.CTRL_FREQ 
        self.MAX_PROGRESS = self.MAX_SPEED * dt

        print(f"🛠️ Env Configurado: MAX_DIST={self.MAX_DIST:.2f}, MAX_PROGRESS={self.MAX_PROGRESS:.4f}")

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

    def _computeReward(self):
        # For RLIB is used _computeRewardPerDrone
        return 0.0

    
    def _computeObs(self):
        all_states = [self._getDroneStateVector(i) for i in range(self.NUM_DRONES)]
        obs = {}

        # =========================
        # GLOBAL OBS (SIN GOALS ❗)
        # =========================
        global_obs = []

        for i in range(self.NUM_DRONES):
            state = all_states[i]

            pos = state[0:3] / 5.0
            vel = state[10:13] / 3.0

            global_obs.extend(np.clip(pos, -1, 1))
            global_obs.extend(np.clip(vel, -1, 1))

        global_obs = np.array(global_obs, dtype=np.float32)
        global_obs = np.clip(global_obs, -1.0, 1.0)

        if not np.isfinite(global_obs).all():
            print("🚨 NaN/Inf in global_obs")
            global_obs = np.nan_to_num(global_obs, nan=0.0, posinf=1.0, neginf=-1.0)

        # =========================
        # PER AGENT
        # =========================

        for i in range(self.NUM_DRONES):

            state = all_states[i]
            pos = state[0:3]
            rpy = state[7:10]
            vel = state[10:13]
            ang_vel = state[13:16]

            # -------------------------
            # GOAL RELATIVE
            # -------------------------
            goal_vec = self.goals[i] - pos
            goal_rel = np.clip(goal_vec / 5.0, -1, 1)

            # -------------------------
            # VELOCITY
            # -------------------------
            vel_norm = np.clip(vel / 3.0, -1, 1)

            # -------------------------
            # ATTITUDE
            # -------------------------
            rp = np.clip(rpy[0:2] / 0.5, -1, 1)

            # -------------------------
            # YAW (sin/cos)
            # -------------------------
            yaw = np.array([
                np.sin(rpy[2]),
                np.cos(rpy[2])
            ], dtype=np.float32)

            # -------------------------
            # ANGULAR VELOCITY
            # -------------------------
            ang_vel_norm = np.clip(ang_vel / 10.0, -1, 1)

            # -------------------------
            # ALIGNMENT (🔥 CLAVE)
            # -------------------------
            goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)
            vel_dir = vel / (np.linalg.norm(vel) + 1e-6)

            alignment = np.dot(goal_dir, vel_dir)
            alignment = np.clip(alignment, -1.0, 1.0)

            # -------------------------
            # CLOSEST DRONE
            # -------------------------
            min_dist = 1e6
            closest_vec = np.zeros(3, dtype=np.float32)

            for j in range(self.NUM_DRONES):
                if j == i:
                    continue

                other_pos = all_states[j][0:3]
                rel_vec = other_pos - pos

                dist = np.linalg.norm(rel_vec) + 1e-6

                if dist < min_dist:
                    min_dist = dist
                    closest_vec = rel_vec

            closest_vec = np.clip(closest_vec / self.MAX_DIST, -1, 1)
            dist_norm = np.clip(min_dist / self.MAX_DIST, 0, 1)

            # -------------------------
            # OBSTACLES (TOP 3 + DIST)
            # -------------------------
            obs_flat = []

            if len(self.obstacles) > 0:
                obs_temp = []

                for obs_data in self.obstacles:
                    obs_pos = np.array(obs_data[0])
                    size = obs_data[1]

                    rel_vec = obs_pos - pos
                    dist = np.linalg.norm(rel_vec) + 1e-6

                    obs_temp.append({
                        "rel_vec": np.clip(rel_vec / self.MAX_DIST, -1, 1),
                        "size": np.clip(size / 5.0, 0, 1),
                        "dist": dist
                    })

                obs_temp.sort(key=lambda x: x["dist"])

                for j in range(3):
                    if j < len(obs_temp):
                        item = obs_temp[j]

                        obs_flat.extend(item["rel_vec"])              # 3
                        obs_flat.append(item["size"])                 # 1
                        obs_flat.append(np.clip(item["dist"]/5.0,0,1))# 1
                    else:
                        obs_flat.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            else:
                obs_flat = [0.0] * (3 * 5)

            obs_flat = np.array(obs_flat, dtype=np.float32)

            # -------------------------
            # FINAL LOCAL OBS
            # -------------------------
            local_obs = np.concatenate([
                goal_rel,          # 3
                vel_norm,          # 3mi 
                rp,                # 2
                yaw,               # 2
                ang_vel_norm,      # 3
                closest_vec,       # 3
                [dist_norm],       # 1
                [alignment],       # 1
                obs_flat           # 15 (3 obstáculos * 5)
            ]).astype(np.float32)

            local_obs = np.clip(local_obs, -1.0, 1.0)

            if not np.isfinite(local_obs).all():
                print(f"🚨 NaN in drone {i}")
                local_obs = np.nan_to_num(local_obs, nan=0.0, posinf=1.0, neginf=-1.0)

            # -------------------------
            # FINAL OBS (FLAT)
            # -------------------------
            full_obs = np.concatenate([local_obs, global_obs]).astype(np.float32)

            obs[f"drone_{i}"] = full_obs

        return obs


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
            # DEBUG
            # -------------------------
            if self.step_counter % 50 == 0 and self.debug:
                print(f"[DEBUG] Drone {i} | pos={pos} | dist={dist:.2f} | z={pos[2]:.2f} | roll={roll:.2f} | pitch={pitch:.2f}")

            # -------------------------
            # 3. CRASH: HEIGHT (SOLO REAL)
            # -------------------------
            if pos[2] < 0.03:
                if self.debug:
                    print(f"[TERMINATED] Drone {i} crashed (ground): z={pos[2]:.3f}")
                return True

            # -------------------------
            # 4. CRASH: ORIENTATION (SOLO IRRECUPERABLE)
            # -------------------------
            if abs(roll) > 1.5 or abs(pitch) > 1.5:
                if self.debug:
                    print(f"[TERMINATED] Drone {i} flipped: roll={roll:.2f}, pitch={pitch:.2f}")
                return True

            # -------------------------
            # 5. OBSTACLES (CONTACTO REAL)
            # -------------------------
            for obs in self.obstacles:
                obs_pos = np.array(obs[0])
                d_obs = np.linalg.norm(pos - obs_pos)

                if d_obs < 0.12:
                    if self.debug:
                        print(f"[TERMINATED] Drone {i} hit obstacle HARD: {d_obs:.2f}")
                    return True

        # -------------------------
        # 6. SUCCESS
        # -------------------------
        if all_drones_on_goal:
            if self.debug:
                print(f"--- ✅ SUCCESS at step {self.step_counter} ---")
            return True

        # -------------------------
        # 7. DRONE COLLISION (REAL)
        # -------------------------
        if self.NUM_DRONES > 1:
            d = np.linalg.norm(
                self._getDroneStateVector(0)[0:3] -
                self._getDroneStateVector(1)[0:3]
            )

            if d < 0.10:
                if self.debug:
                    print(f"[TERMINATED] Drone collision HARD: {d:.3f}")
                return True

        return False

    def _computeTruncated(self):
        # -------------------------
        # 1. OUT OF BOUNDS
        # -------------------------
        for i in range(self.NUM_DRONES):

            pos = self._getDroneStateVector(i)[0:3]

            if np.any(np.abs(pos) > 15.0):
                if self.debug:
                    print(f"[TRUNCATED] Drone {i} out of bounds: pos={pos}")
                return True

            if self.step_counter % 50 == 0 and self.debug:
                print(f"[DEBUG-TRUNC] Drone {i} pos={pos}")

        # -------------------------
        # 2. TIME LIMIT
        # -------------------------
        elapsed_time = self.step_counter / self.PYB_FREQ

        if elapsed_time > self.EPISODE_LEN_SEC:
            if self.debug:
                print(f"[TRUNCATED] Time limit reached: {elapsed_time:.2f}s / {self.EPISODE_LEN_SEC}s")
            return True

        if self.step_counter % 100 == 0 and self.debug:
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
            pos = state[0:3]
            vel = state[10:13]

            # =========================
            # 1. DIRECCIÓN
            # =========================
            direction = np.clip(action[k, 0:3], -1, 1)

            # =========================
            # 2. VELOCIDAD (MEJOR ESCALA)
            # =========================
            speed = (action[k, 3] + 1)

            max_speed = 1.5
            target_vel = direction * speed * max_speed

            # =========================
            # 3. TARGET POS (MEJOR ESCALA)
            # =========================
            target_pos = pos + target_vel * 0.5

            # =========================
            # 4. CONTROL PID
            # =========================
            rpm_k, _, _ = self.ctrl[k].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=pos,
                cur_quat=state[3:7],
                cur_vel=vel,
                cur_ang_vel=state[13:16],
                target_pos=target_pos,
                target_rpy=np.array([0, 0, state[9]]),
                target_vel=target_vel
            )

            rpm[k, :] = rpm_k

        return rpm

    def step(self, action_dict):
        self.step_counter += 1

        # =========================
        # 1. Convertir acciones (ROBUSTO)
        # =========================
        actions = []

        for i in range(self.NUM_DRONES):
            agent_id = f"drone_{i}"

            if agent_id in action_dict:
                a = action_dict[agent_id]
            else:
                # 🔥 fallback (muy importante en RLlib nuevo)
                a = np.zeros(4, dtype=np.float32)

            actions.append(a)

        actions = np.array(actions, dtype=np.float32)

        # =========================
        # 2. Step base
        # =========================
        _, _, terminated, truncated, _ = super().step(actions)

        # =========================
        # 3. Reward
        # =========================
        rewards_per_drone = self._computeRewardPerDrone()

        # =========================
        # 4. Observaciones
        # =========================
        obs = self._computeObs()

        # =========================
        # 5. Formato Multi-Agent RLlib
        # =========================
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}

        for i in range(self.NUM_DRONES):
            agent_id = f"drone_{i}"

            rewards[agent_id] = float(rewards_per_drone[i])
            terminateds[agent_id] = bool(terminated)
            truncateds[agent_id] = bool(truncated)
            infos[agent_id] = {}

        # 🔥 obligatorio para MultiAgentEnv
        terminateds["__all__"] = bool(terminated)
        truncateds["__all__"] = bool(truncated)

        # =========================
        # 6. Tracking
        # =========================
        self.episode_reward += sum(rewards_per_drone)

        return obs, rewards, terminateds, truncateds, infos

    def _observationSpace(self):
        return self.observation_space

    def _computeRewardPerDrone(self):

        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

        rewards = []

        # =========================
        # GLOBAL SIGNAL
        # =========================
        goal_dists = [
            np.linalg.norm(self.goals[i] - states[i][0:3])
            for i in range(self.NUM_DRONES)
        ]

        team_progress = np.mean([
            np.dot(
                states[i][10:13],  # vel
                (self.goals[i] - states[i][0:3]) /
                (np.linalg.norm(self.goals[i] - states[i][0:3]) + 1e-6)
            )
            for i in range(self.NUM_DRONES)
        ])

        # =========================
        # PER AGENT
        # =========================
        for i in range(self.NUM_DRONES):

            pos = states[i][0:3]
            vel = states[i][10:13]

            dist = goal_dists[i]
            speed = np.linalg.norm(vel)

            # -------------------------
            # 1. PROGRESS
            # -------------------------
            goal_vec = self.goals[i] - pos
            goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)

            progress = np.dot(vel, goal_dir)

            # -------------------------
            # 4. DIRECTION
            # -------------------------
            goal_vec = self.goals[i] - pos
            goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)

            vel_dir = vel / (np.linalg.norm(vel) + 1e-6)

            alignment = np.dot(goal_dir, vel_dir)

            # -------------------------
            # 5. MOVEMENT
            # -------------------------
            movement_penalty = -0.5 if speed < 0.05 else 0.0

            # -------------------------
            # 6. GOAL BONUS
            # -------------------------
            goal_bonus = 0.0
            if dist < 0.3:
                goal_bonus = 10.0
                self.reached[i] = True

            # -------------------------
            # 7. OBSTACLES
            # -------------------------
            obstacle_penalty = 0.0
            min_dist = float("inf")

            for obs in self.obstacles:
                obs_pos = np.array(obs[0])
                obs_radius = obs[1]

                d = np.linalg.norm(pos - obs_pos) - obs_radius
                min_dist = min(min_dist, d)

            if self.obstacles and min_dist < 0.5:
                obstacle_penalty = -2.0 * (1.0 - (np.clip(min_dist, 0, 0.5) / 0.5))**2

            if self.obstacles and min_dist < 0.15:
                obstacle_penalty += -10.0

            # -------------------------
            # 8. HEIGHT
            # -------------------------
            height_penalty = -2.0 if pos[2] < 0.2 else 0.0

            # -------------------------
            # TOTAL (GLOBAL + LOCAL)
            # -------------------------


            # 1. PROGRESS → [-1, 1]
            progress_norm = np.clip(progress / self.MAX_PROGRESS, -1.0, 1.0)

            # 2. TEAM PROGRESS → [-1, 1]
            team_progress_norm = np.clip(team_progress / self.MAX_PROGRESS, -1.0, 1.0)

            # 3. DIST → [0, 1] (invertido)
            dist_norm = np.clip(dist / self.MAX_DIST, 0.0, 1.0)
            dist_reward = 1.0 - dist_norm

            # 4. SPEED → [0, 1]
            speed_norm = np.clip(speed / self.MAX_SPEED, 0.0, 1.0)

            # 5. ALIGNMENT ya está en [-1, 1]
            alignment_norm = alignment

            # 6. MOVEMENT → [-1, 0]
            movement_norm = movement_penalty  # ya está bien

            # 7. GOAL BONUS → {0,1}
            goal_norm = 1.0 if dist < 0.3 else 0.0

            # 8. OBSTACLES → [-1, 0]
            obstacle_norm = 0.0
            if self.obstacles:
                if min_dist < 0.5:
                    obstacle_norm = - (1.0 - (np.clip(min_dist, 0, 0.5) / 0.5))**2
                if min_dist < 0.15:
                    obstacle_norm -= 1.0

            # 9. HEIGHT → [-1, 0]
            height_norm = -1.0 if pos[2] < 0.2 else 0.0

            time_penalty = -0.01 

            # 2. Re-balanceo de pesos
            reward = (
                0.10 * team_progress_norm
                + 0.40 * progress_norm      # Sube el peso del progreso (era 0.25)
                # + 0.05 * dist_reward       # Baja la distancia estática (era 0.10)
                + 0.10 * alignment_norm    # Baja alineación (era 0.20)
                + 0.05 * movement_norm
                + 0.05 * speed_norm        # Baja velocidad pura
                + 0.25 * goal_norm         # Sube mucho el bonus de llegar (era 0.10)
                + 0.10 * obstacle_norm
                + 0.05 * height_norm
                + time_penalty             # Añade esto
            )

            # reward = np.clip(reward, -10.0, 10.0)

            if not np.isfinite(reward):
                reward = 0.0

            rewards.append(reward)
            # self.prev_goal_dist[i] = dist

        return rewards

    def get_action_space(self, agent_id):
        return self.action_space[agent_id]

    def get_observation_space(self, agent_id):
        return self.observation_space[agent_id]

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.agents = [f"drone_{i}" for i in range(self.NUM_DRONES)]

        # =========================
        self.step_counter = 0
        self.episode_reward = 0.0
        self.reached = [False] * self.NUM_DRONES

        #if not hasattr(self, "prev_goal_dist"):
        #    self.prev_goal_dist = np.zeros(self.NUM_DRONES, dtype=np.float32)

        #for i in range(self.NUM_DRONES):
            #pos = self._getDroneStateVector(i)[0:3]
            #self.prev_goal_dist[i] = np.linalg.norm(self.goals[i] - pos)

        obs = self._computeObs()

        infos = {agent_id: {} for agent_id in self.agents}

        return obs, infos

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