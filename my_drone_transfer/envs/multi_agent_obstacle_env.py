import numpy as np
import csv
import os
import pybullet as p

from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, Physics, DroneModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class MultiAgentObstacleEnv(BaseRLAviary):

    def __init__(self, obs=ObservationType.KIN, act=ActionType.VEL, gui=False):
        self.colors = [
            [0.86, 0.37, 0.34, 1], [0.35, 0.70, 0.90, 1],
            [0.50, 0.78, 0.50, 1], [0.95, 0.77, 0.35, 1],
            [0.72, 0.56, 0.87, 1], [0.60, 0.60, 0.60, 1]
        ]

        # Definición de metas (Goals)
        self.goals = np.array([
            [2.5, 2.5, 1.8],
            [-2.5, -2.5, 1.8]
        ])

        self.obstacles = []
        self.NUM_DRONES = 1
        self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(self.NUM_DRONES)]

        self.reward_log_path = "reward_debug.csv"

        # crear archivo con header
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

        # Configuración de límites y estados
        self.EPISODE_LEN_SEC = 120
        self.step_counter = 0
        self.episode_reward = 0.0
        self.reached = [False] * self.NUM_DRONES
        self.prev_goal_dist = np.zeros(self.NUM_DRONES)
        
        # Espacios de búsqueda (Gymnasium)
        self.observation_space = spaces.Dict({
            "goal": spaces.Box(-1, 1, shape=(self.NUM_DRONES, 3), dtype=np.float32),
            "velocity": spaces.Box(-1, 1, shape=(self.NUM_DRONES, 3), dtype=np.float32),
            "attitude": spaces.Box(-1, 1, shape=(self.NUM_DRONES, 2), dtype=np.float32),
            "yaw": spaces.Box(-1, 1, shape=(self.NUM_DRONES, 2), dtype=np.float32),
            "angular_velocity": spaces.Box(-1, 1, shape=(self.NUM_DRONES, 3), dtype=np.float32),
            "other": spaces.Box(-1, 1, shape=(self.NUM_DRONES, 3), dtype=np.float32),
            "obstacles": spaces.Box(-1, 1, shape=(self.NUM_DRONES, 9), dtype=np.float32),
        })

        radius = 1.5  # máximo 2 al inicio

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

            # 1. Goal Relativo (Normalizado)
            goal_rel = (self.goals[i] - pos) / 5.0
            goal_list.append(np.clip(goal_rel, -1, 1))

            # 2. Velocidad Lineal
            vel_list.append(np.clip(vel / 3.0, -1, 1))

            # 3. Actitud (Roll y Pitch)
            rp_list.append(np.clip(rpy[0:2] / 0.5, -1, 1))

            # 4. Yaw (Representación Sin/Cos para evitar discontinuidad en 2pi)
            yaw_list.append([np.sin(rpy[2]), np.cos(rpy[2])])

            # 5. Velocidad Angular
            ang_vel_list.append(np.clip(ang_vel / 10.0, -1, 1))

            # 6. Relación con el otro dron
            # Ajustado para ser dinámico si añades más drones después

            if self.NUM_DRONES > 1:
                other_idx = 1 - i 
                other_pos = all_states[other_idx][0:3]
                other_rel = (other_pos - pos) / 5.0
                other_list.append(np.clip(other_rel, -1, 1))
            else:
                # 🔥 IMPORTANTE: dummy value consistente
                other_list.append(np.zeros(3))

            # 7. Obstáculos (Los 3 más cercanos)
            rel_obstacles_flat = []
            if len(self.obstacles) > 0:
                rel_obs = []
                for obs_data in self.obstacles:
                    obs_pos = np.array(obs_data[0])
                    r_pos = (obs_pos - pos) / 5.0
                    dist = np.linalg.norm(r_pos)
                    rel_obs.append(r_pos) # Guardamos el vector relativo

                # Ordenar por distancia (norma del vector)
                rel_obs.sort(key=lambda x: np.linalg.norm(x))

                for j in range(3):
                    if j < len(rel_obs):
                        rel_obstacles_flat.extend(rel_obs[j])
                    else:
                        rel_obstacles_flat.extend([1.0, 1.0, 1.0]) # Padding
            else:
                rel_obstacles_flat = [1.0] * 9 # Padding si no hay obstáculos

            obs_list.append(rel_obstacles_flat)

        # Retornar el diccionario con shapes (NUM_DRONES, N)
        return {
            "goal": np.array(goal_list, dtype=np.float32),
            "velocity": np.array(vel_list, dtype=np.float32),
            "attitude": np.array(rp_list, dtype=np.float32),
            "yaw": np.array(yaw_list, dtype=np.float32),
            "angular_velocity": np.array(ang_vel_list, dtype=np.float32),
            "other": np.array(other_list, dtype=np.float32),
            "obstacles": np.array(obs_list, dtype=np.float32),
        }

    def _computeReward(self):

        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

        total_reward = 0
        reward_info = []

        for i in range(self.NUM_DRONES):

            pos = states[i][0:3]
            vel = states[i][10:13]

            dist = np.linalg.norm(self.goals[i] - pos)
            speed = np.linalg.norm(vel)

            # componentes
            progress = self.prev_goal_dist[i] - dist
            progress_r = 3.0 * progress

            distance_r = 1.0 / (1.0 + dist)

            speed_penalty = 0.0
            if dist < 1.0:
                speed_penalty = -0.3 * speed

            bonus = 0.0
            if dist < 0.25 and speed < 0.1:
                bonus = 5.0
                self.reached[i] = True

            time_penalty = -0.01

            reward = progress_r + distance_r + speed_penalty + bonus + time_penalty

            total_reward += reward

            self.prev_goal_dist[i] = dist

            reward_info.append({
                "drone": i,
                "dist": dist,
                "progress": progress_r,
                "distance_reward": distance_r,
                "speed_penalty": speed_penalty,
                "bonus": bonus,
                "time_penalty": time_penalty,
                "total": reward
            })

        # 🔥 GUARDAR para info()
        self._last_reward_info = reward_info

        with open(self.reward_log_path, mode="a", newline="") as f:
            writer = csv.writer(f)

            for r in reward_info:
                writer.writerow([
                    self.step_counter,
                    r["drone"],
                    r["dist"],
                    r["progress"],
                    r["distance_reward"],
                    r["speed_penalty"],
                    r["bonus"],
                    r["time_penalty"],
                    r["total"]
                ])


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
            # (solo cada 50 pasos para no saturar)
            if self.step_counter % 50 == 0:
                print(f"[DEBUG] Drone {i} | pos={pos} | dist={dist:.2f} | z={pos[2]:.2f} | roll={roll:.2f} | pitch={pitch:.2f}")

            # -------------------------
            # 3. CRASH: ALTURA
            # -------------------------
            if pos[2] < 0.05:
                print(f"[TERMINATED] Drone {i} crashed (low altitude): z={pos[2]:.3f}")
                return True

            # -------------------------
            # 4. CRASH: ORIENTACIÓN
            # -------------------------
            if abs(roll) > 1.2 or abs(pitch) > 1.2:
                print(f"[TERMINATED] Drone {i} unstable: roll={roll:.2f}, pitch={pitch:.2f}")
                return True

            # -------------------------
            # 5. OBSTÁCULOS (opcional debug)
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

            # debug periódico
            if self.step_counter % 50 == 0:
                print(f"[DEBUG-TRUNC] Drone {i} pos={pos}")

        # -------------------------
        # 2. TIME LIMIT
        # -------------------------
        elapsed_time = self.step_counter / self.PYB_FREQ

        if elapsed_time > self.EPISODE_LEN_SEC:
            print(f"[TRUNCATED] Time limit reached: {elapsed_time:.2f}s / {self.EPISODE_LEN_SEC}s")
            return True

        # debug tiempo
        if self.step_counter % 100 == 0:
            print(f"[DEBUG-TIME] step={self.step_counter} time={elapsed_time:.2f}s")

        return False
    
    def _computeInfo(self):
        return {
            "is_success": int(all(self.reached)),
            "reward_breakdown": getattr(self, "_last_reward_info", None)
        }

    def _distance_to_wall(self, pos, center, half_extents):

        dx = max(abs(pos[0] - center[0]) - half_extents[0], 0)
        dy = max(abs(pos[1] - center[1]) - half_extents[1], 0)
        dz = max(abs(pos[2] - center[2]) - half_extents[2], 0)

        return np.linalg.norm([dx, dy, dz])

    def _preprocessAction(self, action):
        # Re-ajuste de tu lógica de control
        action = action.reshape(self.NUM_DRONES, 4)
        rpm = np.zeros((self.NUM_DRONES, 4))

        for k in range(self.NUM_DRONES):
            state = self._getDroneStateVector(k)
            pos, vel = state[0:3], state[10:13]
            
            # Dirección y velocidad escalada
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
                target_rpy=np.array([0, 0, state[9]]), # Mantener yaw actual
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
        # Reset de posiciones con el ruido que definiste
        noise = np.random.uniform(-0.2, 0.2, size=(self.NUM_DRONES, 3))
        self.INIT_XYZS = np.array([[0, -1.0, 1.2], [0, 1.0, 1.2]]) + noise
        self.reached = [False] * self.NUM_DRONES
        return super().reset(seed=seed, options=options)

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
            
            # 2. Creamos el cuerpo en el mundo SIN forma de colisión
            p.createMultiBody(
                baseMass=0, # Masa 0 para que sea estático
                baseCollisionShapeIndex=-1, # <--- IMPORTANTE: -1 significa SIN COLISIÓN
                baseVisualShapeIndex=visual_sphere,
                basePosition=goal,
                physicsClientId=self.CLIENT
            )

            #p.addUserDebugText(
            #    text=labels[i],
            #    textPosition=[goal[0], goal[1], goal[2] + 0.4],
            #    textColorRGB=goal_colors[i][:3],
            #    textSize=1.5,
            #    physicsClientId=self.CLIENT
            #)

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