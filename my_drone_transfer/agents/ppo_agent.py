from stable_baselines3 import PPO
import torch

def build_agent(env):
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256]
        ),
        activation_fn=torch.nn.ReLU
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
    )
    return model