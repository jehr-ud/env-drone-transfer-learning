import time
import csv
import os
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO

from drone_transfer.envs.single_agent_obstacle_env import SingleDroneEnv
from drone_transfer.config.vars import NUM_EPISODES_TEST, ESCENARIOS_PPO

# -------------------------------
# CONFIG
# -------------------------------
os.makedirs("logs", exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# -------------------------------
# LOOP OVER SCENARIOS
# -------------------------------
for escenario in ESCENARIOS_PPO:

    CSV_FILE = f"logs/evaluation_results_{escenario.get('name_model')}_{timestamp}.csv"

    # -------------------------------
    # CSV SETUP
    # -------------------------------
    file_exists = os.path.isfile(CSV_FILE)

    csv_file = open(CSV_FILE, mode="a", newline="")
    writer = csv.writer(csv_file)

    if not file_exists:
        writer.writerow([
            "method",
            "scenario",
            "episode",
            "success",
            "episode_reward",
            "steps",
            "final_distance"
        ])

    print("\n" + "="*60)
    print(f"EVALUATING PPO: {escenario['name_model']}")
    print("="*60)

    # -------------------------------
    # ENV (nuevo por escenario)
    # -------------------------------
    env = SingleDroneEnv(
        gui=False,
        obstacles=escenario.get("obstacles"),
        goal=escenario.get("goal")
    )

    model_path = f"models/{escenario['name_model']}"

    # -------------------------------
    # LOAD MODEL
    # -------------------------------
    model = PPO.load(model_path, env=env)

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

        obs, _ = env.reset()
        done = False

        step = 0
        episode_reward = 0
        final_dist = None
        success = 0

        while not done:

            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            step += 1

            # -------------------------------
            # DISTANCE
            # -------------------------------
            pos = env._getDroneStateVector(0)[0:3]
            dist = np.linalg.norm(pos - env.goal)
            final_dist = dist

            # -------------------------------
            # SUCCESS / TERMINATION
            # -------------------------------
            if done:
                if dist < 0.32:
                    success = 1
                    success_count += 1
                break

        # -------------------------------
        # STORE
        # -------------------------------
        all_rewards.append(episode_reward)
        episode_lengths.append(step)

        writer.writerow([
            "ppo",
            escenario["type"],
            ep,
            success,
            episode_reward,
            step,
            final_dist
        ])

    csv_file.flush()

    # -------------------------------
    # FINAL METRICS (por escenario)
    # -------------------------------
    success_rate = success_count / NUM_EPISODES_TEST
    final_reward = np.mean(all_rewards)
    cumulative_reward = np.sum(all_rewards)
    avg_steps = np.mean(episode_lengths)
    reward_std = np.std(all_rewards)

    print("\n--- SUMMARY ---")
    print(f"Scenario: {escenario['type']}")
    print(f"Success Rate: {success_rate * 100:.2f}%")
    print(f"Final Reward (mean): {final_reward:.2f}")
    print(f"Cumulative Reward: {cumulative_reward:.2f}")
    print(f"Avg Steps: {avg_steps:.2f}")
    print(f"Reward Std: {reward_std:.2f}")

    # -------------------------------
    # CLEAN
    # -------------------------------
    env.close()

# -------------------------------
# CLOSE CSV
# -------------------------------
csv_file.close()

print("\n🚀 ALL PPO EVALUATIONS COMPLETE!")