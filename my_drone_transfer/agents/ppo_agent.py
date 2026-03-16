from stable_baselines3 import PPO
import torch

def build_agent(env):

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,

        # -------------------------------
        # Learning stability
        # -------------------------------
        learning_rate=3e-5,

        # -------------------------------
        # PPO rollout
        # -------------------------------
        n_steps=4096,
        batch_size=256,
        n_epochs=10,

        # -------------------------------
        # Discounting
        # -------------------------------
        gamma=0.995,
        gae_lambda=0.97,

        # -------------------------------
        # Exploration
        # -------------------------------
        ent_coef=0.002,

        # -------------------------------
        # PPO clipping
        # -------------------------------
        clip_range=0.12,

        # -------------------------------
        # Value function
        # -------------------------------
        vf_coef=0.5,
        max_grad_norm=0.5,

        # -------------------------------
        # Policy network
        # -------------------------------
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.Tanh
        )
    )

    return model