import os

from stable_baselines3 import SAC
from drone_transfer.envs.single_agent_obstacle_env import SingleDroneEnv
from drone_transfer.agents.sac_agent import build_agent
from drone_transfer.train.training_logger import TrainingLoggerCallback
from ..config.vars import TOTAL_STEPS, ESCENARIOS_SAC

# -------------------------------
# Create folders
# -------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("models/checkpoints", exist_ok=True)

# -------------------------------
# LOOP OVER SCENARIOS
# -------------------------------
for escenario in ESCENARIOS_SAC:

    print("\n" + "="*50)
    print(f"TRAINING SAC: {escenario['name_model']}")
    print("="*50)

    # -------------------------------
    # ENV (nuevo por escenario)
    # -------------------------------
    env = SingleDroneEnv(
        gui=False,
        obstacles=escenario.get("obstacles"),
        goal=escenario.get("goal")
    )

    model_path = f"models/{escenario['name_model']}"

    # -------------------------------
    # MODEL (scratch vs transfer)
    # -------------------------------
    if escenario.get("scratch"):
        model = build_agent(env)
    else:
        source_path = f"models/{escenario['source_model']}"
        model = SAC.load(source_path, env=env)

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
        callback=callback
    )

    # -------------------------------
    # SAVE
    # -------------------------------
    model.save(model_path)

    # -------------------------------
    # CLEAN
    # -------------------------------
    env.close()

print("\n🚀 ALL SAC TRAINING COMPLETE!")