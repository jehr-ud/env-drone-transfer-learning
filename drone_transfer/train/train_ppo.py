import os

from drone_transfer.envs.single_agent_obstacle_env import SingleDroneEnv
from drone_transfer.agents.ppo_agent import build_agent
from drone_transfer.train.training_logger import TrainingLoggerCallback
from .vars import TOTAL_STEPS

# -------------------------------
# Create folders
# -------------------------------
os.makedirs("models", exist_ok=True)

# -------------------------------
# 1. Setup Env
# -------------------------------
MODEL_PATH = "models/ppo_drone_final"
env = SingleDroneEnv(gui=False, with_obstacles=True)

# -------------------------------
# 2. Setup Model
# -------------------------------
model = build_agent(env)

# -------------------------------
# 3. Callback
# -------------------------------
callback = TrainingLoggerCallback(
    save_freq=200000, 
    save_path="./models/checkpoints/",
    name_algo="PPO"
)

# -------------------------------
# 4. Train
# -------------------------------
print("Starting training...")
model.learn(
    total_timesteps=TOTAL_STEPS,
    progress_bar=True,
    tb_log_name="PPO_run_train",
    callback=callback
)

# -------------------------------
# 5. Save
# -------------------------------
model.save("models/ppo_drone_simple")

print("Training Complete!")