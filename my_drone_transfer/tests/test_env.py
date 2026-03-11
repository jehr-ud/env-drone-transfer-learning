from my_drone_transfer.envs.multi_agent_obstacle_env import MultiAgentObstacleEnv
import numpy as np
import time

env = MultiAgentObstacleEnv(gui=True)

obs, _ = env.reset()

for step in range(1000):

    # acción neutra (hover esperado)
    action = np.zeros(env.NUM_DRONES * 3, dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\nSTEP {step}")

    for i in range(env.NUM_DRONES):
        state = env._getDroneStateVector(i)

        z = state[2]
        vz = state[12]

        print(f"Drone {i}: z={z:.3f}, vz={vz:.3f}")

    env.render()

    time.sleep(1/240)   # velocidad realista

    if terminated or truncated:
        print("Episode finished")
        break

env.close()