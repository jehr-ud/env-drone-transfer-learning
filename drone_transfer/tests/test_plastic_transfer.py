import numpy as np
import pybullet as p
import time

from plastic_transfer import PlasticTransfer
from drone_transfer.envs.multi_agent_obstacle_env import MultiAgentObstacleEnv
from drone_transfer.agents.ppo_agent import build_agent

# -------------------------------
# CONFIG
# -------------------------------
NUM_EPISODES = 5
MODEL_PATH = "models/plastic_transfer_model"


# -------------------------------
# ENV
# -------------------------------
env = MultiAgentObstacleEnv(gui=True, with_obstacles=True)


# -------------------------------
# INIT MODEL (IMPORTANTE)
# -------------------------------
obs_sample, _ = env.reset()

pt = PlasticTransfer(
    env=env,
    ppo_builder=build_agent,
    hidden_size=128,
    latent_size=16,
)

pt.load(MODEL_PATH)

# -------------------------------
# EVALUATION
# -------------------------------
success_count = 0
all_rewards = []

for ep in range(NUM_EPISODES):
    print(f"\n--- EPISODE {ep+1} ---")

    obs, _ = env.reset()
    done = False
    step = 0
    episode_reward = 0

    while not done:

        # -------------------------------
        # ACTION (IMPORTANTE)
        # -------------------------------
        action = pt.predict(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += reward

        # -------------------------------
        # DEBUG DISTANCES
        # -------------------------------
        current_dists = []
        current_positions = []

        for i in range(env.NUM_DRONES):
            pos = env._getDroneStateVector(i)[0:3]
            dist = np.linalg.norm(pos - env.goals[i])

            current_dists.append(dist)
            current_positions.append(pos)

            color = [1, 0, 0] if i == 0 else [0, 0, 1]

            p.addUserDebugLine(
                current_positions[i],
                env.goals[i],
                color,
                lineWidth=2,
                lifeTime=0.1,
                physicsClientId=env.CLIENT
            )

        # -------------------------------
        # LOG
        # -------------------------------
        if step % 50 == 0:
            print(f"Step {step} | reward {reward:.3f} | dist {np.round(current_dists,2)}")

        # -------------------------------
        # SUCCESS
        # -------------------------------
        if info.get("is_success", 0) == 1:
            print(f"SUCCESS ✅ Distancias finales: {np.round(current_dists, 3)}")
            success_count += 1
            break

        if done:
            print(f"END ❌ Step {step}")
            print(f"Distancias: {np.round(current_dists, 3)}")
            break

        step += 1
        time.sleep(1/48)

    all_rewards.append(episode_reward)


# -------------------------------
# RESULTS
# -------------------------------
print("\n" + "="*30)
print(f"RESULTS OVER {NUM_EPISODES} EPISODES")
print(f"Success Rate: {success_count/NUM_EPISODES * 100:.1f}%")
print(f"Average Reward: {np.mean(all_rewards):.2f}")
print("="*30)

env.close()