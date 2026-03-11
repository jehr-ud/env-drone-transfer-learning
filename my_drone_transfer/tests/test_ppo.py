from stable_baselines3 import PPO
from my_drone_transfer.envs.multi_agent_obstacle_env import MultiAgentObstacleEnv

env = MultiAgentObstacleEnv(gui=True)

model = PPO.load("models/ppo_multiagent")

obs, _ = env.reset()

for step in range(1000):

    action, _ = model.predict(obs, deterministic=True)

    print(f"\nSTEP {step}")
    print("ACTION:", action)

    obs, reward, terminated, truncated, info = env.step(action)

    for i in range(env.NUM_DRONES):
        state = env._getDroneStateVector(i)

        print(
            f"Drone {i}: "
            f"x={state[0]:.2f} "
            f"y={state[1]:.2f} "
            f"z={state[2]:.2f} "
            f"vz={state[12]:.2f}"
        )

    env.render()

    if terminated or truncated:
        break

env.close()