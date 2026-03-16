from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from my_drone_transfer.envs.multi_agent_obstacle_env import MultiAgentObstacleEnv

import numpy as np
import time

# -------------------------------
# Configuration
# -------------------------------
NUM_EPISODES = 5

# -------------------------------
# Environment
# -------------------------------
env = DummyVecEnv([lambda: MultiAgentObstacleEnv(gui=True)])

env = VecNormalize.load("models/vec_normalize.pkl", env)

env.training = False
env.norm_reward = False

# -------------------------------
# Load model
# -------------------------------
model = PPO.load("models/ppo_multiagent")

# -------------------------------
# Metrics
# -------------------------------
success = 0
collisions = 0

all_rewards = []
all_distances = []
all_lengths = []

# -------------------------------
# Evaluation loop
# -------------------------------
for ep in range(NUM_EPISODES):

    print(f"\n==============================")
    print(f"EPISODE {ep+1}")
    print(f"==============================")

    obs = env.reset()

    episode_reward = 0

    for step in range(1000):

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)

        episode_reward += reward[0]

        real_env = env.envs[0]

        print(f"\nSTEP {step}")
        print("ACTION:", action)
        print("REWARD:", reward)

        goal_reached = True
        final_distances = []

        for i in range(real_env.NUM_DRONES):

            state = real_env._getDroneStateVector(i)

            pos = state[0:3]
            vz = state[12]

            dist = np.linalg.norm(pos - real_env.goals[i])

            final_distances.append(dist)

            print(
                f"Drone {i}: "
                f"x={pos[0]:.2f} "
                f"y={pos[1]:.2f} "
                f"z={pos[2]:.2f} "
                f"vz={vz:.2f} "
                f"dist_goal={dist:.2f}"
            )

            if dist >= 0.2:
                goal_reached = False

        # -------------------------------
        # Goal reached
        # -------------------------------
        if goal_reached:

            print("GOAL REACHED ✅")

            success += 1

            all_rewards.append(episode_reward)
            all_distances.append(np.mean(final_distances))
            all_lengths.append(step)

            break

        # -------------------------------
        # Episode finished
        # -------------------------------
        if done[0]:

            print("EPISODE FINISHED ⚠️")

            crashed = False

            for i in range(real_env.NUM_DRONES):

                state = real_env._getDroneStateVector(i)

                if state[2] < 0.1:
                    print(f"Drone {i} crashed on floor")
                    crashed = True

            if crashed:
                collisions += 1

            all_rewards.append(episode_reward)
            all_distances.append(np.mean(final_distances))
            all_lengths.append(step)

            break

        real_env.render()

        time.sleep(0.03)

# -------------------------------
# Final metrics
# -------------------------------
print("\n==============================")
print("FINAL EVALUATION RESULTS")
print("==============================")

print(f"Success rate: {success / NUM_EPISODES:.2f}")
print(f"Collision rate: {collisions / NUM_EPISODES:.2f}")
print(f"Mean reward: {np.mean(all_rewards):.2f}")
print(f"Mean final distance: {np.mean(all_distances):.2f}")
print(f"Mean episode length: {np.mean(all_lengths):.2f}")