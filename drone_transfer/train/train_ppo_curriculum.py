import os

from drone_transfer.envs.single_agent_obstacle_env import SingleDroneEnv
from drone_transfer.agents.ppo_agent import build_agent
from drone_transfer.train.training_logger import TrainingLoggerCallback

# -------------------------------
# CONFIG
# -------------------------------
TOTAL_STEPS_PER_STAGE = 1_000_000

os.makedirs("models", exist_ok=True)

# -------------------------------
# STAGES
# -------------------------------
stages = [
    {"name": "easy", "difficulty": 0},
    {"name": "medium", "difficulty": 1},
    {"name": "hard", "difficulty": 2},
    {"name": "final", "difficulty": 3}
]

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

    # 🔥 CONTROL REAL DE DIFICULTAD
    if stage["difficulty"] == 0:
        env.obstacles = []

    elif stage["difficulty"] == 1:
        env.obstacles = env.obstacles[:2]

    elif stage["difficulty"] == 2:
        env.obstacles = env.obstacles[:5]

    elif stage["difficulty"] == 3:
        pass  # full env

    # Assign env
    model.set_env(env)

    # -------------------------------
    # CALLBACK
    # -------------------------------
    callback = TrainingLoggerCallback(
        save_freq=200000,
        save_path="./models/checkpoints/",
        name_algo=f"PPO_CURRICULUM_{stage['name']}"
    )

    # -------------------------------
    # TRAIN
    # -------------------------------
    model.learn(
        total_timesteps=TOTAL_STEPS_PER_STAGE,
        reset_num_timesteps=False,  # 🔥 clave
        progress_bar=True,
        callback=callback
    )

    # -------------------------------
    # SAVE
    # -------------------------------
    model.save(f"models/ppo_curriculum_{stage['name']}")

print("✅ Curriculum Training Complete")

model.save("models/ppo_curriculum_final")