from stable_baselines3 import PPO

def build_agent(env):

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,

        learning_rate=5e-5,
        batch_size=256,
        n_steps=4096,

        gamma=0.995,
        gae_lambda=0.97,

        ent_coef=0.005,
        clip_range=0.15,

        vf_coef=0.5,
        max_grad_norm=0.5,

        policy_kwargs=dict(
            net_arch=[256, 256]
        )
    )

    return model