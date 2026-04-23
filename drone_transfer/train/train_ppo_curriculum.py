import os

from drone_transfer.envs.single_agent_obstacle_env import SingleDroneEnv
from drone_transfer.agents.ppo_agent import build_agent
from drone_transfer.train.training_logger import TrainingLoggerCallback
from ..config.vars import TOTAL_STEPS

# -------------------------------
# CONFIG
# -------------------------------
os.makedirs("models", exist_ok=True)

# -------------------------------
# STAGES
# -------------------------------
stages = [
    {"name": "Easy", "difficulty": 0},
    {"name": "Medium", "difficulty": 1},
    {"name": "Hard", "difficulty": 2},
    {"name": "Final", "difficulty": 3}
]


TOTAL_STEPS_PER_STAGE = TOTAL_STEPS // len(stages)

# -------------------------------
# CREATE INITIAL ENV
# -------------------------------
env = SingleDroneEnv(gui=False, with_obstacles=False)

model = build_agent(env)

print("🚀 Starting Curriculum Training")

# -------------------------------
# TRAIN LOOP
# -------------------------------
for i, stage in enumerate(stages):

    print(f"\n==============================")
    print(f"Training stage: {stage['name']}")
    print(f"==============================")

    # Create env
    env = SingleDroneEnv(gui=False, with_obstacles=True)
    env.set_difficulty(stage["difficulty"])

    # Assign env
    model.set_env(env)

    # -------------------------------
    # CALLBACK
    # -------------------------------
    callback = TrainingLoggerCallback(
        save_freq=200000,
        save_path="./models/checkpoints/",
        name_algo=f"PPO-Curriculum-{stage['name']}"
    )

    # -------------------------------
    # TRAIN
    # -------------------------------
    model.learn(
        total_timesteps=TOTAL_STEPS_PER_STAGE,
        reset_num_timesteps=False,
        progress_bar=True,
        callback=callback
    )

    # -------------------------------
    # SAVE
    # -------------------------------
    model.save(f"models/ppo_curriculum_{stage['name']}")

print("✅ Curriculum Training Complete")

model.save("models/ppo_curriculum_final")