import time
import csv
import os
import json
from datetime import datetime

import numpy as np
import pybullet as p

from plastic_transfer import PlasticTransfer
from drone_transfer.envs.single_agent_obstacle_env import SingleDroneEnv
from drone_transfer.agents.ppo_agent import build_agent
from drone_transfer.config.vars import NUM_EPISODES_TEST, ESCENARIOS_PLASTIC
from drone_transfer.utils import build_obs_dict

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# -------------------------------
# LOAD DEFINITIONS
# -------------------------------
with open("drone_transfer/train/doc/plastic/learning_definitions.json", "r") as f:
    learning_definitions = json.load(f)


for escenario in ESCENARIOS_PLASTIC:

    CSV_FILE = f"logs/evaluation_results_{escenario.get('name_model')}_{timestamp}.csv"
    MODEL_PATH = f"models/{escenario.get('name_model')}"

    print(f"TEST: {escenario['name_model']}")
    path_model = f"models/{escenario.get('name_model')}"

    env = SingleDroneEnv(
        gui=False,
        obstacles=escenario.get("obstacles"),
        goal=escenario.get("goal")
    )

    # -------------------------------
    # INIT MODEL
    # -------------------------------
    pt = PlasticTransfer(
        env=env,
        model_builder=build_agent,
        hidden_size=escenario.get("meta").get("plastic").get("hidden_size"),
        latent_size=escenario.get("meta").get("plastic").get("latent_size"),
        learning_definitions=learning_definitions,
        obs_to_dict_fn=build_obs_dict
    )

    pt.load(MODEL_PATH)

    # -------------------------------
    # CSV SETUP
    # -------------------------------
    file_exists = os.path.isfile(CSV_FILE)

    csv_file = open(CSV_FILE, mode="a", newline="")
    writer = csv.writer(csv_file)

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

            action = pt.predict(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward

            # -------------------------------
            # DISTANCE
            # -------------------------------
            pos = env._getDroneStateVector(0)[0:3]
            dist = np.linalg.norm(pos - env.goal)
            final_dist = dist

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

            # -------------------------------
            # TERMINATION
            # -------------------------------
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
        # STORE METRICS (IDENTICAL TO PPO)
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