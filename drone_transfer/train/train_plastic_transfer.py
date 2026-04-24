import json
import numpy as np

from plastic_transfer import PlasticTransfer
from drone_transfer.agents.ppo_agent import build_agent
from drone_transfer.envs.single_agent_obstacle_env import SingleDroneEnv
from ..config.vars import TOTAL_STEPS, ESCENARIOS_PLASTIC, N_STEPS

with open("drone_transfer/train/doc/plastic/learning_definitions.json", "r") as f:
    learning_definitions = json.load(f)

with open("drone_transfer/train/doc/plastic/base_policy.json", "r") as f:
    policy_config = json.load(f)


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

for escenario in ESCENARIOS_PLASTIC:
    print(f"Training: {escenario['name_model']}")
    path_model = f"models/{escenario.get('name_model')}"

    env = SingleDroneEnv(
        gui=False,
        obstacles=escenario.get("obstacles"),
        goal=escenario.get("goal")
    )

    pt = PlasticTransfer(
        env=env,
        model_builder=build_agent,
        hidden_size=escenario.get("meta").get("plastic").get("hidden_size"),
        latent_size=escenario.get("meta").get("plastic").get("latent_size"),
        logger_path_file=escenario.get("name_model"),
        learning_definitions=learning_definitions,
        skill_train_steps = N_STEPS,
        obs_to_dict_fn=build_obs_dict,
        policy_config=policy_config,
        debug=True
    )

    if not escenario.get("scratch"):
        source_path = f"models/{escenario.get('source_model')}"
        pt.load(source_path)

    pt.learn(TOTAL_STEPS)

    pt.save(path_model)
    env.close()