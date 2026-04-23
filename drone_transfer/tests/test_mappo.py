import os
import numpy as np
import pybullet as p
import torch

from ray.rllib.core.rl_module.rl_module import RLModule
from drone_transfer.envs.multi_agent_obstacle_env import MultiAgentObstacleEnv
from drone_transfer.config.vars import NUM_EPISODES_TEST

# -------------------------------
# CONFIG
# -------------------------------
CHECKPOINT_PATH = os.path.abspath("models/mappo_new")

# -------------------------------
# ENV
# -------------------------------
env = MultiAgentObstacleEnv({
    "gui": True,
    "with_obstacles": False
})

# -------------------------------
# LOAD RL MODULE
# -------------------------------
rl_module = RLModule.from_checkpoint(
    os.path.join(
        CHECKPOINT_PATH,
        "learner_group",
        "learner",
        "rl_module",
        "shared_policy"
    )
)

rl_module.eval()

device = next(rl_module.parameters()).device

print("✅ Model loaded")


# -------------------------------
# LOOP
# -------------------------------
for ep in range(NUM_EPISODES_TEST):

    print(f"\n--- EPISODE {ep+1} ---")

    obs, _ = env.reset()
    step = 0

    while True:

        actions = {}

        for agent_id, agent_obs in obs.items():

            obs_tensor = torch.tensor(
                agent_obs,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            out = rl_module.forward_inference({
                "obs": obs_tensor
            })

            logits = out["action_dist_inputs"][0].detach().cpu().numpy()

            action_dim = env.action_space[agent_id].shape[0]

            mean = logits[:action_dim]
            action = np.tanh(mean)

            actions[agent_id] = action

        # STEP
        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        done = terminateds["__all__"] or truncateds["__all__"]

        print(rewards)

        # debug
        dists = [
            np.linalg.norm(env._getDroneStateVector(i)[0:3] - env.goals[i])
            for i in range(env.NUM_DRONES)
        ]

        if step % 50 == 0:
            print(f"Step {step} | dist {np.round(dists,2)}")

        if done:
            print("FINAL DIST:", np.round(dists, 3))
            break

        step += 1

env.close()