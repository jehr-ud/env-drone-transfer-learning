import copy
import random

import numpy as np

TOTAL_STEPS = 1_000_000

N_STEPS = 2048
N_STEPS_SKILLS = 256
NUM_EPISODES_TEST = 100
DECAY_PLASTIC_SCALE = TOTAL_STEPS * 0.3


TRANSFER_CONFIG = {
    "simple": {
        "goal": np.array([2.0, 2.0, 1.5]),
        "obstacles": [
            ([0.8, 0.0, 1.5], 0.5, 3, "cube"),
            ([1.0, 0.5, 1.5], 0.5, 3, "cube"),

            ([-1.0, -0.5, 2.2], 0.5, 3, "cube"),

            ([0.5, 1.0, 2.6], 0.3, 3, "cylinder"),
            ([-0.5, -1.0, 2.6], 0.3, 3, "cylinder"),

            ([1.6, 2.0, 1.5], 0.45, 3, "cube"),
            ([2.0, 1.6, 1.5], 0.45, 3, "cube"),
        ]
    },

    "medium": {
        "goal": np.array([2.5, -2.5, 1.5]),
        "obstacles": [
            ([0.0, 1.5, 1.5], 0.5, 3, "cube"),
            ([0.0, -1.5, 1.5], 0.5, 3, "cube"),
            ([1.5, 1.5, 1.5], 0.5, 3, "cube"),
            ([1.5, -1.5, 1.5], 0.5, 3, "cube"),

            ([1.0, 0.3, 1.5], 0.45, 3, "cube"),

            ([1.0, 0.0, 2.8], 0.3, 4, "cylinder"),
            ([2.0, -1.0, 3.0], 0.3, 4, "cylinder"),
            ([2.0, 1.0, 3.0], 0.3, 4, "cylinder"),

            ([1.6, -1.2, 2.3], 0.4, 3, "cube"),

            ([2.2, -2.0, 1.5], 0.4, 3, "cube"),
            ([2.8, -2.0, 1.5], 0.4, 3, "cube"),

            ([2.5, -2.5, 3.2], 0.3, 4, "cylinder"),
        ]
    },

    "complex": {
        "goal": np.array([3.0, 3.0, 1.5]),
        "obstacles": [
            ([0.5, 1.0, 1.5], 0.6, 3, "cube"),
            ([0.5, -1.0, 1.5], 0.6, 3, "cube"),

            ([1.5, 0.0, 1.5], 0.6, 3, "cube"),

            ([2.5, 1.0, 1.5], 0.6, 3, "cube"),
            ([2.5, -1.0, 1.5], 0.6, 3, "cube"),

            ([2.0, 2.0, 1.5], 0.6, 3, "cube"),

            ([1.5, 1.5, 2.6], 0.5, 3, "cube"),
            ([2.5, 2.0, 2.6], 0.5, 3, "cube"),

            ([2.0, 0.0, 3.2], 0.35, 4, "cylinder"),
            ([1.5, 2.5, 3.0], 0.35, 4, "cylinder"),

            ([1.0, 2.5, 0.8], 0.5, 3, "cube"),

            ([2.7, 3.0, 1.5], 0.5, 3, "cube"),
            ([3.3, 3.0, 1.5], 0.5, 3, "cube"),
            ([3.0, 2.7, 1.5], 0.5, 3, "cube"),

            ([3.0, 3.0, 3.5], 0.4, 4, "cylinder"),
        ]
    }
}


def build_curriculum_scenarios(base_scenarios, method_name):

    scenarios = []

    curriculum_phases = ["exploration", "consolidation"]

    for base in base_scenarios:

        base_type = base["type"]

        is_scratch = base["scratch"]

        type_train = "scratch" if is_scratch else "transfer"

        exploration_name = (
            f"{method_name}_{base_type}_{type_train}_exploration"
        )

        consolidation_name = (
            f"{method_name}_{base_type}_consolidation"
        )

        for phase_idx, phase_name in enumerate(curriculum_phases):

            new_s = copy.deepcopy(base)

            obstacles = new_s["obstacles"]

            # ====================================
            # EXPLORATION
            # ====================================
            if phase_name == "exploration":

                n = max(1, int(len(obstacles) * 0.4))

                new_s["obstacles"] = random.sample(obstacles, n)

                new_s["name_model"] = exploration_name

                new_s["scratch"] = True

            # ====================================
            # CONSOLIDATION
            # ====================================
            else:

                new_s["obstacles"] = obstacles

                new_s["name_model"] = consolidation_name

                # IMPORTANTE
                new_s["scratch"] = False

                new_s["source_model"] = exploration_name

            new_s["method"] = method_name

            new_s["curriculum"] = {
                "phase": phase_name,
                "phase_idx": phase_idx,
                "base_level": base_type
            }

            scenarios.append(new_s)

    return scenarios


def build_scenarios_for_method(base_scenarios, method_name):
    scenarios = []

    for s in base_scenarios:
        new_s = s.copy()

        # ------------------------
        # rename model
        # ------------------------
        new_s["name_model"] = f"{method_name}_{s['name_model']}"

        # ------------------------
        # rename source if exists
        # ------------------------
        if not s.get("scratch"):
            new_s["source_model"] = f"{method_name}_{s['source_model']}"

        # ------------------------
        # add method tag
        # ------------------------
        new_s["method"] = method_name

        scenarios.append(new_s)

    return scenarios

ESCENARIOS_TRANSFER = [
    {
        "type": "simple",
        "scratch": True,
        "name_model": "simple_scratch",
        "obstacles": TRANSFER_CONFIG.get("simple").get("obstacles"),
        "goal": TRANSFER_CONFIG.get("simple").get("goal"),
        "meta": {
            "plastic": {
                "hidden_size": 64,
                "latent_size": 32,
            }
        }
    },
     {
        "type": "medium",
        "scratch": True,
        "name_model": "medium_scratch",
        "obstacles": TRANSFER_CONFIG.get("medium").get("obstacles"),
        "goal": TRANSFER_CONFIG.get("medium").get("goal"),
        "meta": {
            "plastic": {
                 "hidden_size": 64,
                "latent_size": 32,
            }
        }
    },
    {
        "type": "complex",
        "scratch": True,
        "name_model": "complex_scratch",
        "obstacles": TRANSFER_CONFIG.get("complex").get("obstacles"),
        "goal": TRANSFER_CONFIG.get("complex").get("goal"),
        "meta": {
            "plastic": {
                "hidden_size": 64,
                "latent_size": 32,
            }
        }
    },
    {
        "type": "medium",
        "scratch": False,
        "name_model": "medium_transfer",
        "source_model": "simple_scratch",
        "obstacles": TRANSFER_CONFIG.get("medium").get("obstacles"),
        "goal": TRANSFER_CONFIG.get("medium").get("goal"),
        "meta": {
            "plastic": {
                "hidden_size": 64,
                "latent_size": 32,
            }
        }
    },
    {
        "type": "complex",
        "scratch": False,
        "name_model": "complex_transfer",
        "source_model": "simple_scratch",
        "obstacles": TRANSFER_CONFIG.get("complex").get("obstacles"),
        "goal": TRANSFER_CONFIG.get("complex").get("goal"),
        "meta": {
            "plastic": {
                "hidden_size": 64,
                "latent_size": 32,
            }
        }
    }
]


ESCENARIOS_PLASTIC = build_scenarios_for_method(
    ESCENARIOS_TRANSFER,
    "plastic"
)

ESCENARIOS_PPO = build_scenarios_for_method(
    ESCENARIOS_TRANSFER,
    "ppo"
)

ESCENARIOS_CURRICULUM_PPO = build_curriculum_scenarios(
    ESCENARIOS_TRANSFER,
    "Ppo-curriculum"
)