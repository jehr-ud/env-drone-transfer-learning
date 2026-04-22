from stable_baselines3 import PPO

from drone_transfer.train.vars import N_STEPS

def build_agent(env):
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=N_STEPS,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1
    )

    return model