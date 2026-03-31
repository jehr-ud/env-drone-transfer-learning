from plastic_transfer import PlasticTransfer
from my_drone_transfer.agents.ppo_agent import build_agent

from my_drone_transfer.envs.multi_agent_obstacle_env import MultiAgentObstacleEnv

env = MultiAgentObstacleEnv(gui=False)


input_size = 59 # 25 obs by dron x 2 = 50 + 8 actions + 1 reward
hidden_size = 128 # max(32, input_size * 2)
latent_size = 16


pt = PlasticTransfer(
    env=env,
    ppo_builder=build_agent,
    input_size=input_size,
    hidden_size=hidden_size,
    latent_size=latent_size,
    novelty_threshold=0.2,
)

pt.learn(2_000_000)

pt.save("models/plastic_transfer_model")