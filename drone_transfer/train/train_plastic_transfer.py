from plastic_transfer import PlasticTransfer
from drone_transfer.agents.ppo_agent import build_agent

from drone_transfer.envs.multi_agent_obstacle_env import MultiAgentObstacleEnv

env = MultiAgentObstacleEnv(config={
    "gui":False, 
    "with_obstacles":True
})

pt = PlasticTransfer(
    env=env,
    model_builder=build_agent,
    hidden_size=128,
    latent_size=16,
    novelty_threshold=0.2,
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

pt.learn(2_000_000)

pt.save("models/plastic_transfer_model")