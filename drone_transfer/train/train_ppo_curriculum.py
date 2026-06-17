import os
from stable_baselines3 import PPO

from drone_transfer.envs.single_agent_obstacle_env import SingleDroneEnv
from drone_transfer.agents.ppo_agent import build_agent
from drone_transfer.train.training_logger import TrainingLoggerCallback
from ..config.vars import TOTAL_STEPS, ESCENARIOS_CURRICULUM_PPO


os.makedirs("models", exist_ok=True)

for escenario in ESCENARIOS_CURRICULUM_PPO:

    print("\n" + "="*50)
    print(f"TRAINING PPO CURRICULUM: {escenario['name_model']}")
    print(f"Phase: {escenario.get('curriculum', {}).get('phase')}")
    print(f"Level: {escenario.get('curriculum', {}).get('base_level')}")
    print(f"Scratch: {escenario.get('scratch')}")
    print("="*50)

    # -------------------------------
    # ENV
    # -------------------------------
    env = SingleDroneEnv(
        gui=False,
        obstacles=escenario.get("obstacles"),
        goal=escenario.get("goal")
    )

    model_path = f"models/{escenario['name_model']}"

    # -------------------------------
    # MODEL
    # -------------------------------
    if escenario.get("scratch"):
        model = build_agent(env)
    else:
        source_path = f"models/{escenario['source_model']}"

        if not os.path.exists(source_path + ".zip"):
            raise ValueError(f"Source model not found: {source_path}")

        model = PPO.load(source_path)
        model.set_env(env)

    # -------------------------------
    # CALLBACK
    # -------------------------------
    callback = TrainingLoggerCallback(
        save_freq=200000,
        save_path="./models/checkpoints/",
        name_algo=escenario["name_model"]
    )

    # -------------------------------
    # TRAIN
    # -------------------------------
    model.learn(
        total_timesteps=TOTAL_STEPS,
        progress_bar=True,
        tb_log_name=escenario["name_model"],
        callback=callback,
        reset_num_timesteps=False
    )

    # -------------------------------
    # SAVE
    # -------------------------------
    model.save(model_path)

    env.close()

print("✅ Curriculum Training Complete")