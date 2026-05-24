import numpy as np
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO
from envs.arm_env import TokamakArmEnv
import imageio
import os

gym.register_envs(gymnasium_robotics)

def record_arm(arm_type, n_episodes=5, fps=30):
    print(f"\nRecording {arm_type.upper()} arm — {n_episodes} episodes...")

    os.makedirs("results/videos", exist_ok=True)

    env   = TokamakArmEnv(arm_type=arm_type)
    model = PPO.load(f"models/{arm_type}/best_model")

    render_env = gym.make(
        "FetchReachDense-v4",
        max_episode_steps=200,
        render_mode="rgb_array"
    )

    for ep in range(n_episodes):
        obs, _  = render_env.reset()
        env_obs, _ = env.reset()
        done    = False
        frames  = []
        t       = 0

        while not done:
            # build obs matching trained policy
            flat_obs = np.concatenate([
                obs["observation"],
                env._get_target(t),
                [t / 200]
            ]).astype(np.float32)

            action, _ = model.predict(flat_obs, deterministic=True)

            obs, _, terminated, truncated, _ = render_env.step(action)
            env_obs, _, term2, trunc2, info  = env.step(action)
            done = terminated or truncated or term2 or trunc2

            frame = render_env.render()
            frames.append(frame)
            t += 1

        out_path = f"results/videos/{arm_type}_ep{ep+1}.mp4"
        imageio.mimsave(out_path, frames, fps=fps)
        print(f"  Saved: {out_path} ({len(frames)} frames)")

    render_env.close()
    env.close()
    print(f"\n{arm_type.upper()} recording complete.")

if __name__ == "__main__":
    record_arm("franka", n_episodes=5)
    record_arm("dtt",    n_episodes=5)
    print("\nAll videos saved to results/videos/")
