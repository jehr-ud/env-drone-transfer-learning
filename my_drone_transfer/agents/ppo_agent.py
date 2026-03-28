from stable_baselines3 import PPO
import torch

def build_agent(env):
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        activation_fn=torch.nn.Tanh
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_drone_tensorboard/",
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=256,        # Aumentado para mayor estabilidad en multiaente
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.97,
        ent_coef=0.005,        # Reducido ligeramente para evitar vibraciones excesivas
        clip_range=0.15,
        clip_range_vf=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        device="auto"          # Asegura uso de GPU si está disponible
    )
    return model