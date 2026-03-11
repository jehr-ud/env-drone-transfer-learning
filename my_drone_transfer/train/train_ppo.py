from my_drone_transfer.agents.ppo_agent import build_agent
from my_drone_transfer.envs.multi_agent_obstacle_env import MultiAgentObstacleEnv

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = MultiAgentObstacleEnv()
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

model = build_agent(env)

model.learn(total_timesteps=1000000)

model.save("models/ppo_multiagent")