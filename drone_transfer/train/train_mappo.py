import os
import ray

from ray.rllib.algorithms.ppo import PPOConfig
from drone_transfer.envs.multi_agent_obstacle_env import MultiAgentObstacleEnv

# -------------------------------
# 1. Init Ray
# -------------------------------
ray.init(ignore_reinit_error=True)

# -------------------------------
# 2. Paths
# -------------------------------
BASE_PATH = os.path.abspath("models/mappo_new")
os.makedirs(BASE_PATH, exist_ok=True)
# -------------------------------
# 3. Config (API NUEVA)
# -------------------------------

config = (
    PPOConfig()
    .environment(
        env=MultiAgentObstacleEnv,
        env_config={
            "gui": False,
            "with_obstacles": False
        }
    )
    .framework("torch")

    .env_runners(num_env_runners=2)

    .training(
        lr=5e-5,
        gamma=0.99,
        lambda_=0.95,
        train_batch_size=16000,
        clip_param=0.2,
        grad_clip=0.5,
        vf_clip_param=10.0,
        entropy_coeff=0.02,
    )

    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
    )
)

# -------------------------------
# 4. Build Algorithm
# -------------------------------
algo = config.build()

# -------------------------------
# 5. Training loop
# -------------------------------
print("🚀 Starting training (RLModule API)...")

NUM_ITERATIONS = 2000
best_reward = float("-inf")

for i in range(NUM_ITERATIONS):

    result = algo.train()

    reward_mean = result["env_runners"]["episode_return_mean"]

    print(f"\nIter {i}")
    print(f"  reward_mean: {reward_mean}")
    print(f"  reward_min: {result['env_runners'].get('episode_return_min', 'N/A')}")
    print(f"  reward_max: {result['env_runners'].get('episode_return_max', 'N/A')}")
    # -------------------------------
    # 🏆 Guardar mejor modelo
    # -------------------------------en 
    if reward_mean > best_reward:
        best_reward = reward_mean

        training_result = algo.save_to_path(BASE_PATH)
        print(f"🏆 Best model saved")

# -------------------------------
# 6. Guardar modelo final
# -------------------------------
final_result = algo.save_to_path(BASE_PATH)
final_result = algo.save(checkpoint_dir=BASE_PATH)
print(f"\n💾 Final model saved")

print("✅ Training Complete!")

ray.shutdown()