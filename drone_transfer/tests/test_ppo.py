import time
import csv
import os
from datetime import datetime

import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from drone_transfer.envs.single_agent_obstacle_env import SingleDroneEnv
from drone_transfer.config.vars import NUM_EPISODES_TEST

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "models/ppo_drone_simple"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILE = f"evaluation_results_PPO_{timestamp}.csv"

# -------------------------------
# ENV
# -------------------------------
env = SingleDroneEnv(gui=False, with_obstacles=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = PPO.load(MODEL_PATH, env=env)

# -------------------------------
# CSV SETUP
# -------------------------------
file_exists = os.path.isfile(CSV_FILE)

csv_file = open(CSV_FILE, mode="a", newline="")
writer = csv.writer(csv_file)

# Header
if not file_exists:
    writer.writerow([
        "episode",
        "success",
        "episode_reward",
        "steps",
        "final_distance"
    ])

# -------------------------------
# METRICS
# -------------------------------
success_count = 0
all_rewards = []
episode_lengths = []

# -------------------------------
# EVALUATION LOOP
# -------------------------------
for ep in range(NUM_EPISODES_TEST):
    print(f"\n--- EPISODE {ep+1} ---")

    obs, _ = env.reset()
    done = False

    step = 0
    episode_reward = 0
    final_dist = None

    while not done:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += reward

        # -------------------------------
        # DISTANCE
        # -------------------------------
        pos = env._getDroneStateVector(0)[0:3]
        dist = np.linalg.norm(pos - env.goal)
        final_dist = dist

        # Debug visual
        p.addUserDebugLine(
            pos,
            env.goal,
            [1, 0, 0],
            lineWidth=2,
            lifeTime=0.1,
            physicsClientId=env.CLIENT
        )

        if step % 50 == 0:
            print(f"Step {step} | reward {reward:.3f} | dist {dist:.2f}")

        # TERMINATION
        if terminated:
            success = 0

            final_state = env._getDroneStateVector(0)
            dist_to_goal = np.linalg.norm(final_state[0:3] - env.goal)

            if dist_to_goal < 0.32:
                print(f"✅ SUCCESS (Dist: {dist_to_goal:.3f})")
                success = 1
                success_count += 1
            else:
                print(f"💥 CRASH (Dist to Goal: {dist_to_goal:.3f})")
            break

        if truncated:
            print(f"TRUNCATED ❌ dist={dist:.3f}")
            success = 0
            break

        step += 1
        time.sleep(1/48)

    # -------------------------------
    # STORE METRICS
    # -------------------------------
    all_rewards.append(episode_reward)
    episode_lengths.append(step)

    writer.writerow([
        ep,
        success,
        episode_reward,
        step,
        final_dist
    ])
    csv_file.flush()
# -------------------------------
# FINAL METRICS
# -------------------------------
success_rate = success_count / NUM_EPISODES_TEST
final_reward = np.mean(all_rewards)
cumulative_reward = np.sum(all_rewards)
avg_steps = np.mean(episode_lengths)
reward_std = np.std(all_rewards)

print("\n" + "="*50)
print(f"RESULTS OVER {NUM_EPISODES_TEST} EPISODES")
print(f"Success Rate: {success_rate * 100:.2f}%")
print(f"Final Reward (mean): {final_reward:.2f}")
print(f"Cumulative Reward: {cumulative_reward:.2f}")
print(f"Avg Steps: {avg_steps:.2f}")
print(f"Reward Std (stability): {reward_std:.2f}")
print("="*50)

# -------------------------------
# CLEAN
# -------------------------------
csv_file.close()
env.close()