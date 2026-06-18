from stable_baselines3 import SAC

from drone_transfer.config.vars import N_STEPS

def build_agent(env):
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=500000,
        learning_starts=5000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1
    )

    return model