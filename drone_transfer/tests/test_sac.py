import time
import csv
import os
from datetime import datetime

import numpy as np
from stable_baselines3 import SAC

from drone_transfer.envs.single_agent_obstacle_env import SingleDroneEnv
from drone_transfer.config.vars import NUM_EPISODES_TEST, ESCENARIOS_SAC

os.makedirs("logs", exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for escenario in ESCENARIOS_SAC:

    CSV_FILE = f"logs/evaluation_results_{escenario['name_model']}_{timestamp}.csv"

    print("\n" + "="*60)
    print(f"EVALUATING SAC: {escenario['name_model']}")
    print("="*60)

    env = SingleDroneEnv(
        gui=False,
        obstacles=escenario.get("obstacles"),
        goal=escenario.get("goal")
    )

    model_path = f"models/{escenario['name_model']}"
    model = SAC.load(model_path)
    model.set_env(env)

    success_count = 0
    all_rewards = []
    episode_lengths = []

    with open(CSV_FILE, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow([
            "method",
            "scenario",
            "episode",
            "success",
            "episode_reward",
            "steps",
            "final_distance"
        ])

        for ep in range(NUM_EPISODES_TEST):

            obs, _ = env.reset()
            done = False

            step = 0
            episode_reward = 0
            success = 0
            final_dist = None

            while not done:

                action, _ = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                step += 1

                pos = env._getDroneStateVector(0)[0:3]
                dist = np.linalg.norm(pos - env.goal)
                final_dist = dist

                if done:
                    if dist < 0.32:
                        success = 1
                        success_count += 1
                    break

            all_rewards.append(episode_reward)
            episode_lengths.append(step)

            writer.writerow([
                escenario.get("method", "SAC"),
                escenario["type"],
                ep,
                success,
                episode_reward,
                step,
                final_dist
            ])

        csv_file.flush()

    # -------------------------------
    # SUMMARY
    # -------------------------------
    success_rate = success_count / NUM_EPISODES_TEST

    print("\n--- SUMMARY ---")
    print(f"Scenario: {escenario['type']}")
    print(f"Success Rate: {success_rate * 100:.2f}%")
    print(f"Reward Mean: {np.mean(all_rewards):.2f}")
    print(f"Reward Std: {np.std(all_rewards):.2f}")
    print(f"Avg Steps: {np.mean(episode_lengths):.2f}")

    env.close()

print("\n🚀 ALL SAC EVALUATIONS COMPLETE!")