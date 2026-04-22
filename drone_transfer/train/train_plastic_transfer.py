import json
import numpy as np

from plastic_transfer import PlasticTransfer
from drone_transfer.agents.ppo_agent import build_agent
from drone_transfer.envs.single_agent_obstacle_env import SingleDroneEnv
from .vars import TOTAL_STEPS

env = SingleDroneEnv(gui=False, with_obstacles=True)


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
    skill_train_steps= TOTAL_STEPS / len(learning_definitions.get("skills")),
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

pt.learn(TOTAL_STEPS)
pt.save("models/plastic_transfer_model")