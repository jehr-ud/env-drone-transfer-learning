import numpy as np
import pybullet as p
import time
import json

from plastic_transfer import PlasticTransfer
from drone_transfer.envs.single_agent_obstacle_env import SingleDroneEnv
from drone_transfer.agents.ppo_agent import build_agent

# -------------------------------
# CONFIG
# -------------------------------
NUM_EPISODES = 5
MODEL_PATH = "models/plastic_transfer_model"


# -------------------------------
# ENV
# -------------------------------
env = SingleDroneEnv(gui=True, with_obstacles=True)


# -------------------------------
# INIT MODEL (IMPORTANTE)
# -------------------------------
obs_sample, _ = env.reset()

with open("drone_transfer/train/doc/plastic/learning_definitions.json", "r") as f:
    learning_definitions = json.load(f)


def build_obs_dict(obs):
    return {
        "goal_rel": obs[0:3],
        "vel_norm": obs[3:6],
        "rp": obs[6:8],
        "yaw": obs[8:10],
        "ang_vel_norm": obs[10:13],
        "alignment": obs[13],
        "dist": obs[14],
        "rel_vec": obs[15:18],
        "dist_norm": obs[18],
        "size_norm": obs[19],
    }


def base_policy(obs):
    goal = obs[0:3]
    vel = obs[3:6]

    direction = goal - vel

    speed = np.linalg.norm(direction)
    speed = np.clip(speed, 0.0, 1.0)

    return np.array([
        direction[0],
        direction[1],
        direction[2],
        speed
    ])


pt = PlasticTransfer(
    env=env,
    model_builder=build_agent,
    hidden_size=128,
    latent_size=16,
    novelty_threshold=0.2,
    logger_path_file="plastic",
    learning_definitions=learning_definitions,
    obs_to_dict_fn=build_obs_dict,
    base_policy_fn=base_policy,
    observations_keys=[
        "goal",
        "velocity",
        "attitude",
        "yaw",
        "angular_velocity",
        "other",
        "obstacles"
    ]
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
            dist = np.linalg.norm(pos - env.goal)

            current_dists.append(dist)
            current_positions.append(pos)

            color = [1, 0, 0] if i == 0 else [0, 0, 1]

            p.addUserDebugLine(
                current_positions[i],
                env.goal,
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