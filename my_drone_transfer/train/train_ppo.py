import os

from my_drone_transfer.envs.multi_agent_obstacle_env import MultiAgentObstacleEnv
from my_drone_transfer.agents.ppo_agent import build_agent
from my_drone_transfer.train.training_logger import TrainingLoggerCallback

# -------------------------------
# Create folders
# -------------------------------
os.makedirs("models", exist_ok=True)

# -------------------------------
# 1. Setup Env
# -------------------------------
env = MultiAgentObstacleEnv(gui=False)

# -------------------------------
# 2. Setup Model
# -------------------------------
model = build_agent(env)

# -------------------------------
# 3. Callback
# -------------------------------
callback = TrainingLoggerCallback(
    save_freq=200000, 
    save_path="./models/checkpoints/"
)

# -------------------------------
# 4. Train
# -------------------------------
print("Starting training...")
model.learn(
    total_timesteps=1_000_000,
    progress_bar=True,
    tb_log_name="PPO_run_train",
    callback=callback
)

# -------------------------------
# 5. Save
# -------------------------------
model.save("models/ppo_drone_final")
# env.save("models/vec_normalize.pkl")

print("Training Complete!")