import numpy as np
import time
from my_drone_transfer.envs.multi_agent_obstacle_env import MultiAgentObstacleEnv

env = MultiAgentObstacleEnv(gui=True)
obs, info = env.reset()

print("--- Test: Navegación hacia la meta ---")

reached_once = [False] * env.NUM_DRONES

for step in range(2000):

    action = np.zeros((env.NUM_DRONES, 4), dtype=np.float32)

    for i in range(env.NUM_DRONES):
        state = env._getDroneStateVector(i)
        pos = state[0:3]

        goal = env.goals[i]

        # -------------------------------
        # DIRECCIÓN HACIA LA META
        # -------------------------------
        direction = goal - pos
        dist = np.linalg.norm(direction)

        if dist > 1e-6:
            direction = direction / dist

        # -------------------------------
        # CONTROL SIMPLE (clave)
        # -------------------------------
        action[i, 0:3] = direction   # vx, vy, vz hacia meta
        action[i, 3] = 0.8           # potencia

        # -------------------------------
        # DETECTAR LLEGADA (DEBUG)
        # -------------------------------
        if dist < 0.3 and not reached_once[i]:
            reached_once[i] = True
            print(f"✅ Drone {i} LLEGÓ a la meta en step {step}")

    obs, reward, terminated, truncated, info = env.step(action.flatten())

    if step % 20 == 0:
        print(f"\nSTEP {step}")
        for i in range(env.NUM_DRONES):
            pos = env._getDroneStateVector(i)[0:3]
            dist = np.linalg.norm(pos - env.goals[i])
            print(f"Drone {i} -> dist_to_goal: {dist:.3f} | reached: {env.reached[i]}")

    env.render()
    time.sleep(1 / env.CTRL_FREQ)

    # -------------------------------
    # FIN DEL EPISODIO
    # -------------------------------
    if terminated:
        print("\n🎯 TODOS los drones llegaron a la meta")
        break

    if truncated:
        print("\n⚠️ Episodio truncado")
        break

env.close()