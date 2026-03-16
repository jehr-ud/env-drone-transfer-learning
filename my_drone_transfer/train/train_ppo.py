from my_drone_transfer.agents.ppo_agent import build_agent
from my_drone_transfer.envs.multi_agent_obstacle_env import MultiAgentObstacleEnv

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .training_logger import TrainingLoggerCallback

# -------------------------------
# Environment
# -------------------------------
env = DummyVecEnv([lambda: MultiAgentObstacleEnv()])

env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=5.0
)

# -------------------------------
# Agent
# -------------------------------
model = build_agent(env)

# -------------------------------
# Callback
# -------------------------------
callback = TrainingLoggerCallback()

# -------------------------------
# Training
# -------------------------------
model.learn(
    total_timesteps=100,
    callback=callback
)

# -------------------------------
# Save model
# -------------------------------
model.save("models/ppo_multiagent")

env.save("models/vec_normalize.pkl")