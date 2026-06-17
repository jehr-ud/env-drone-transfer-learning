import json
import numpy as np

from plastic_transfer import PlasticTransfer
from drone_transfer.agents.ppo_agent import build_agent
from drone_transfer.envs.single_agent_obstacle_env import SingleDroneEnv
from ..config.vars import (
    TOTAL_STEPS,
    ESCENARIOS_PLASTIC,
    N_STEPS,
    N_STEPS_SKILLS,
    DECAY_PLASTIC_SCALE
)
from drone_transfer.utils import build_obs_dict

with open("drone_transfer/train/doc/plastic/learning_definitions.json", "r") as f:
    learning_definitions = json.load(f)

with open("drone_transfer/train/doc/plastic/base_policy.json", "r") as f:
    policy_config = json.load(f)


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
        skill_train_steps = N_STEPS_SKILLS,
        obs_to_dict_fn=build_obs_dict,
        policy_config=policy_config,
        debug=True
    )

    if not escenario.get("scratch"):
        source_path = f"models/{escenario.get('source_model')}"
        pt.load(source_path)

    pt.learn(
        total_steps=TOTAL_STEPS,
        decay_scale=DECAY_PLASTIC_SCALE,
    )

    pt.save(path_model)
    env.close()