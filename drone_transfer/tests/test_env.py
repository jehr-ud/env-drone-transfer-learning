import numpy as np
import time
from drone_transfer.envs.single_agent_obstacle_env import SingleDroneEnv


def run_test_episode(env, target_position, description):
    print(f"\n--- Starting: {description} ---")
    obs, _ = env.reset()

    target_pos_arr = np.array(target_position, dtype=np.float32)

    # Detect if goal
    is_env_goal = np.linalg.norm(target_pos_arr - env.goal) < 0.1

    if not is_env_goal:
        time.sleep(0.1)
        env._highlightObstacle(target_pos_arr, color=[1, 1, 0, 1])
        print(f"🟡 Highlighting obstacle at {target_position}")

    for step in range(1000):
        state = env._getDroneStateVector(0)
        pos = state[0:3]

        # =========================
        # TARGET ADJUSTMENT
        # =========================
        target = target_pos_arr.copy()

        if not is_env_goal:
            # 💥 crash mode → prevent hover up
            target[2] = pos[2]          # XY plane
            target[1] += 0.2            # small offset → prevents perfect balance

        # =========================
        # DIRECTION
        # =========================
        direction = target - pos
        dist = np.linalg.norm(direction)

        # =========================
        # CONTROL
        # =========================
        if is_env_goal:
            # 🎯 smooth mode (convergence)
            if dist < 0.1:
                action = np.zeros(4, dtype=np.float32)  # stop
            else:
                direction = direction / (dist + 1e-6)
                action = np.zeros(4, dtype=np.float32)
                action[0:3] = direction
                action[3] = 0.6

        else:
            # 💥 impact mode (aggressive)
            direction[2] = 0
            direction = np.sign(direction)

            action = np.zeros(4, dtype=np.float32)
            action[0:3] = direction
            action[3] = 1.0

        # =========================
        # STEP
        # =========================
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 50 == 0:
            print(f"Step {step} | Dist: {dist:.3f}")

        # =========================
        # END CONDITIONS
        # =========================
        if terminated:
            final_state = env._getDroneStateVector(0)
            dist_to_goal = np.linalg.norm(final_state[0:3] - env.goal)

            if dist_to_goal < 0.32:
                print(f"✅ SUCCESS (Dist: {dist_to_goal:.3f})")
            else:
                print(f"💥 CRASH (Dist to Goal: {dist_to_goal:.3f})")
            break

        if truncated:
            print("🕒 TIME LIMIT")
            break

        time.sleep(1 / env.CTRL_FREQ)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    env = SingleDroneEnv(gui=True, with_obstacles=True)

    # 🎯 TEST 1: GOAL
    run_test_episode(env, env.goal, "FLY TO GOAL")

    # 💥 TEST 2: CRASH
    if len(env.obstacles) > 2:
        obstacle_pos = env.obstacles[2][0]
        run_test_episode(env, obstacle_pos, "INTENTIONAL CRASH TEST")

    env.close()