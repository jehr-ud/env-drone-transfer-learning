import numpy as np
import pybullet as p

from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, Physics, DroneModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class MultiAgentObstacleEnv(BaseRLAviary):

    def __init__(
        self,
        obs=ObservationType.KIN,
        act=ActionType.VEL,
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
           # ([3, 0, 1.5], [0.2, 3, 1.5], 3, "wall"),
           # ([-3, 0, 1.5], [0.2, 3, 1.5], 4, "wall"),

           # ([0, 0, 1.5], 3, 0, "cube"),
           # ([0, 1.5, 1.0], 2, 1, "cube"),
           # ([0, -1.5, 1.0], 2, 2, "cube"),

           # ([1.5, 0.8, 1.0], 2, 5, "cube"),
           # ([-1.5, -0.8, 1.0], 2, 0, "cube"),

           # ([1.2, -2.0, 1.0], 2, 2, "cube"),
           # ([-1.2, 2.0, 1.0], 2, 1, "cube"),

           # ([2.2, 1.5, 2.0], 4, 1, "cylinder"),
           # ([-2.2, -1.5, 2.0], 2, 2, "cylinder"),

           # ([2.0, -1.5, 2.0], 4, 4, "cylinder"),
           # ([-2.0, 1.5, 2.0], 2, 5, "cylinder"),

           # ([0.8, 2.5, 2.0], 2, 1, "cylinder"),
           # ([-0.8, -2.5, 2.0], 3, 2, "cylinder")
        ]

        self.goals = np.array([
            [2.5, 2.5, 1.8],
            [-2.5, -2.5, 1.8]
        ])

        self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(2)]

        self.EPISODE_LEN_SEC = 50

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=2,
            neighbourhood_radius=10,
            initial_xyzs=np.array([
                [0, -0.5, 1.2],
                [0, 0.5, 1.2]
            ]),
            initial_rpys=np.zeros((2, 3)),
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=48,
            gui=gui,
            obs=obs,
            act=act
        )

        self.EPISODE_LEN_SEC = 20
        self.reached = [False]*self.NUM_DRONES

        self.SPEED_LIMIT = 0.6

        self.prev_goal_dist = np.zeros(self.NUM_DRONES)
        self.prev_action = np.zeros(self.NUM_DRONES * 4)
        self.last_action = np.zeros(self.NUM_DRONES * 4)
        
        self.max_steps = 5000

        #goal_rel           # 3
        #norm_vel           # 3
        #norm_rp            # 2
        #[yaw_sin, yaw_cos] # 2
        #norm_ang_vel       # 3
        #other_rel          # 3
        #rel_obstacles_flat # 9
        # 25 valores 

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(50,), # 2 drones × 25 valores
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.NUM_DRONES * 4,), # 4 comandos: vx, vy, vz, yaw
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
        all_states = [self._getDroneStateVector(i) for i in range(self.NUM_DRONES)]
        
        for i in range(self.NUM_DRONES):
            state = all_states[i]
            pos = state[0:3]
            quat = state[3:7]     # Quaternion para cálculos más precisos
            rpy = state[7:10]      # Roll, Pitch, Yaw
            vel = state[10:13]     # Velocidad lineal
            ang_vel = state[13:16] # Velocidad angular

            # 1. Meta relativa (Escalada a 5 metros)
            # Si la meta está a 5m, el valor será 1.0. 
            goal_rel = (self.goals[i] - pos) / 5.0

            # 2. Otros drones (Relativo)
            other_idx = 1 - i
            other_pos = all_states[other_idx][0:3]
            other_rel = (other_pos - pos) / 5.0

            # 3. Obstáculos (Los 3 más cercanos)
            # IMPORTANTE: Si la lista está vacía, rellenamos con valores neutros
            rel_obstacles_flat = []
            if len(self.obstacles) > 0:
                rel_obs = []
                for obs_pos, _, _, _ in self.obstacles:
                    r_pos = (np.array(obs_pos) - pos) / 5.0
                    dist = np.linalg.norm(r_pos)
                    rel_obs.append((dist, r_pos))
                
                rel_obs.sort(key=lambda x: x[0])
                
                for j in range(3):
                    if j < len(rel_obs):
                        rel_obstacles_flat.extend(rel_obs[j][1])
                    else:
                        rel_obstacles_flat.extend([1.0, 1.0, 1.0]) # Representa "lejos"
            else:
                # Si no hay obstáculos, enviamos 9 valores de "lejos"
                rel_obstacles_flat = [1.0] * 9

            # 4. Estados propios normalizados
            # Velocidad: limitada a 3m/s (rango -1 a 1)
            norm_vel = np.clip(vel / 3.0, -1, 1)
            
            # Actitud: Roll y Pitch son cruciales para la estabilidad
            # Dividir por 0.5 (~30 grados) ayuda a que la red sea sensible a inclinaciones
            norm_rp = np.clip(rpy[0:2] / 0.5, -1, 1)
            
            # Yaw relativo: ¡ESTO ES CLAVE! 
            # En lugar de usar el Yaw global, usamos el seno y coseno.
            # Esto evita el salto de pi a -pi que confunde a la IA.
            yaw_sin = np.sin(rpy[2])
            yaw_cos = np.cos(rpy[2])
            
            # Velocidad angular: Normalizada a 10 rad/s
            norm_ang_vel = np.clip(ang_vel / 10.0, -1, 1)

            # 5. Concatenación Final (24 valores por dron)
            # 3(goal) + 3(vel) + 2(rp) + 2(yaw_sin_cos) + 3(ang_vel) + 3(other) + 8(obs... ajustado)
            # Para mantener tus 46 totales (23 por dron), usaremos esta estructura:
            drone_obs = np.concatenate([
                goal_rel,          # 3
                norm_vel,          # 3
                norm_rp,           # 2
                [yaw_sin, yaw_cos],# 2
                norm_ang_vel,      # 3
                other_rel,         # 3
                rel_obstacles_flat[0:9]
            ]).astype(np.float32)

            obs.append(drone_obs)

        return np.array(obs).flatten()


    def _computeReward(self):
        rewards = []

        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)

            pos = state[0:3]
            vel = state[10:13]
            goal = self.goals[i]

            # -------------------------------
            # DISTANCIAS
            # -------------------------------
            xy_dist = np.linalg.norm(pos[0:2] - goal[0:2])
            z_dist = abs(pos[2] - goal[2])
            dist_goal = xy_dist + 0.5 * z_dist

            # -------------------------------
            # 1. PROGRESO (FIX REAL)
            # -------------------------------
            progress = self.prev_goal_dist[i] - dist_goal
            reward = 8.0 * progress

            # -------------------------------
            # 2. ATRACCIÓN SUAVE
            # -------------------------------
            reward += 0.3 * np.exp(-xy_dist)
            reward += 0.6 * np.exp(-z_dist)

            # -------------------------------
            # 3. DIRECCIÓN (MUY IMPORTANTE)
            # -------------------------------
            goal_vec = goal - pos
            goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)
            approach_speed = np.dot(vel, goal_dir)

            reward += 0.4 * approach_speed

            # -------------------------------
            # 4. PENALIZAR QUIETUD (CLAVE)
            # -------------------------------
            speed = np.linalg.norm(vel)
            if speed < 0.05:
                reward -= 0.2

            # -------------------------------
            # 5. ESTABILIDAD
            # -------------------------------
            reward -= 0.01 * (abs(state[7]) + abs(state[8]))

            # -------------------------------
            # 6. ZONA DE META
            # -------------------------------
            if xy_dist < 0.4 and z_dist < 0.25:
                reward += 10.0
                reward += (0.4 - xy_dist) * 10.0
                reward += (0.25 - z_dist) * 12.0

                # frenar
                reward -= 0.3 * speed

            # -------------------------------
            # 7. TIEMPO
            # -------------------------------
            reward -= 0.01

            # -------------------------------
            # UPDATE CORRECTO
            # -------------------------------
            self.prev_goal_dist[i] = dist_goal

            rewards.append(reward)

        return np.min(rewards)
    

    def _computeTerminated(self):
        # -------------------------------
        # GOAL CHECK
        # -------------------------------
        new_reached = self.reached.copy()

        for i in range(self.NUM_DRONES):
            dist = np.linalg.norm(
                self._getDroneStateVector(i)[0:3] - self.goals[i]
            )

            if not self.reached[i]:
                if dist < 0.3:
                    new_reached[i] = True
            else:
                if dist > 0.4:
                    new_reached[i] = False

        # actualizar TODO al final
        self.reached = new_reached

        if all(self.reached):
            print("[DEBUG] finish for reached")
            print(self.reached)
            return True

        # -------------------------------
        # SAFETY
        # -------------------------------
        for i in range(self.NUM_DRONES):

            state = self._getDroneStateVector(i)
            pos = state[0:3]

            roll = abs(state[7])
            pitch = abs(state[8])

            # SOLO si está cerca del suelo
            if (roll > 1.5 or pitch > 1.5) and pos[2] < 0.3:
                print("[DEBUG] finish por estar cerca del suelo")
                return True

            if pos[2] < 0.05:
                print("[DEBUG] finish pos[2] < 0.05")
                return True

            for obstacle, size, _, obstacle_type in self.obstacles:

                if obstacle_type == "wall":
                    d = self._distance_to_wall(pos, obstacle, size)
                else:
                    d = np.linalg.norm(pos - np.array(obstacle))

                if d < 0.15:
                    print("[DEBUG] finish d < 0.15")
                    return True

        # -------------------------------
        # DRONE COLLISION
        # -------------------------------
        d_drone = np.linalg.norm(
            self._getDroneStateVector(0)[0:3] -
            self._getDroneStateVector(1)[0:3]
        )

        if d_drone < 0.08:
            print("[DEBUG] finish d_drone collision")
            return True

        return False

    def _computeTruncated(self):

        states = np.array([
            self._getDroneStateVector(i) for i in range(self.NUM_DRONES)
        ])

        for i in range(self.NUM_DRONES):
            pos = states[i][0:3]
            roll = abs(states[i][7])
            pitch = abs(states[i][8])

            # -------------------------------
            # 1. OUT OF BOUNDS
            # -------------------------------
            if abs(pos[0]) > 6 or abs(pos[1]) > 6 or pos[2] > 3:
                print("[DEBUG] truncated: out of bounds")
                return True

            # -------------------------------
            # 2. EXTREME TILT
            # -------------------------------
            if roll > 3.2 or pitch > 3.2:
                print("[DEBUG] truncated: extreme tilt")
                print("roll:", roll, "pitch:", pitch)
                return True

            # -------------------------------
            # 3. MOVING AWAY FROM GOAL
            # -------------------------------
            dist = np.linalg.norm(pos - self.goals[i])

            if dist > self.prev_goal_dist[i] * 1.8:
                print("[DEBUG] truncated: moving away from goal")
                return True

        # -------------------------------
        # 4. TIME
        # -------------------------------
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            print("[DEBUG] truncated: time")
            return True

        return False

    def _computeInfo(self):
        return {
            "is_success": int(all(self.reached))
        }

    def _distance_to_wall(self, pos, center, half_extents):

        dx = max(abs(pos[0] - center[0]) - half_extents[0], 0)
        dy = max(abs(pos[1] - center[1]) - half_extents[1], 0)
        dz = max(abs(pos[2] - center[2]) - half_extents[2], 0)

        return np.linalg.norm([dx, dy, dz])

    def _preprocessAction(self, action):
        action = action.reshape(self.NUM_DRONES, 4)
        rpm = np.zeros((self.NUM_DRONES, 4))
        
        for k in range(self.NUM_DRONES):
            state = self._getDroneStateVector(k)

            pos = state[0:3]
            vel = state[10:13]

            target_v = np.clip(action[k, 0:3], -1, 1)
            speed_factor = (action[k, 3] + 1) / 2

            # -------------------------------
            # Dirección normalizada
            # -------------------------------
            norm_v = np.linalg.norm(target_v)
            v_unit = target_v / norm_v if norm_v > 1e-6 else np.zeros(3)

            # -------------------------------
            # Velocidad controlada
            # -------------------------------
            speed = 0.6 * speed_factor   # 🔥 más bajo
            target_vel = speed * v_unit

            # -------------------------------
            # SUAVIZADO (CRÍTICO)
            # -------------------------------
            alpha = 0.6
            target_vel = alpha * target_vel + (1 - alpha) * vel

            # -------------------------------
            # CONVERTIR A TARGET POSITION
            # -------------------------------
            dt = self.CTRL_TIMESTEP
            target_pos = pos + target_vel * dt * 3.0

            # -------------------------------
            # CONTROL PID
            # -------------------------------
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
        self.prev_action = np.zeros(self.NUM_DRONES * 4)
        self.last_action = np.zeros(self.NUM_DRONES * 4)

        self.episode_reward = 0.0

        self.INIT_Z = 1.2

        self.reached = [False]*self.NUM_DRONES

        obs, info = super().reset(seed=seed, options=options)

        self.prev_goal_dist = np.zeros(self.NUM_DRONES)

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