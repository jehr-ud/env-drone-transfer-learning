import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

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
env = DummyVecEnv([lambda: Monitor(MultiAgentObstacleEnv(gui=False))])

env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=True,
    # clip_obs=10.,
    # clip_reward=10.
)

env.seed(42)

# -------------------------------
# 2. Setup Model
# -------------------------------
model = build_agent(env)
model.set_random_seed(42)

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
    total_timesteps=2_000_000,
    progress_bar=True,
    tb_log_name="PPO_run_train",
    callback=callback
)

# -------------------------------
# 5. Save
# -------------------------------
model.save("models/ppo_drone_final")
env.save("models/vec_normalize.pkl")

print("Training Complete!")