import numpy as np
import pybullet as p
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from my_drone_transfer.envs.multi_agent_obstacle_env import MultiAgentObstacleEnv

# -------------------------------
# Configuration
# -------------------------------
NUM_EPISODES = 5
MODEL_PATH = "models/ppo_drone_final"

# -------------------------------
# Environment Setup
# -------------------------------
env = MultiAgentObstacleEnv(gui=True, with_obstacles=True)

# -------------------------------
# Load Model
# -------------------------------
model = PPO.load(MODEL_PATH, env=env)

# -------------------------------
# Evaluation Loop
# -------------------------------
success_count = 0
all_rewards = []

for ep in range(NUM_EPISODES):
    print(f"\n--- EPISODE {ep+1} ---")

    obs, _ = env.reset()
    done = False
    step = 0
    episode_reward = 0

    raw_env = env

    while not done:

        # -------------------------------
        # Acción
        # -------------------------------
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += reward

        current_dists = []
        current_positions = []

        for i in range(raw_env.NUM_DRONES):
            pos = raw_env._getDroneStateVector(i)[0:3]
            dist = np.linalg.norm(pos - raw_env.goals[i])
            current_dists.append(dist)
            current_positions.append(pos)

            # Dibujar líneas
            color = [1, 0, 0] if i == 0 else [0, 0, 1]
            p.addUserDebugLine(
                current_positions[i],
                raw_env.goals[i],
                color,
                lineWidth=2,
                lifeTime=0.1,
                physicsClientId=raw_env.CLIENT
            )

        # DEBUG ligero
        if step % 50 == 0:
            print(f"Step {step} | reward {reward:.3f} | dist {np.round(current_dists,2)}")

        # SUCCESS REAL
        if info.get("is_success", 0) == 1:
            print(f"SUCCESS ✅ Distancias finales: {np.round(current_dists, 3)}")
            success_count += 1
            break

        # TERMINATED (pero no éxito)
        if terminated:
            print(f"TERMINATED ⚠️ Step {step}")
            print(f"Distancias: {np.round(current_dists, 3)}")
            break

        # TRUNCATED
        if truncated:
            print(f"TRUNCATED ❌ Step {step}")
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