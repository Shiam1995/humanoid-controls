import numpy as np
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO
from envs.arm_env import TokamakArmEnv
import matplotlib.pyplot as plt

gym.register_envs(gymnasium_robotics)

def replay(arm_type, n_episodes=200):
    print(f"\nReplaying {arm_type.upper()} for {n_episodes} episodes...")

    env   = TokamakArmEnv(arm_type=arm_type)
    model = PPO.load(f"models/{arm_type}/best_model")

    all_errors  = []
    all_rewards = []

    for ep in range(n_episodes):
        obs, _    = env.reset()
        done      = False
        ep_errors = []
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_errors.append(info["pos_error"])
            ep_reward += reward

        all_errors.append(np.mean(ep_errors))
        all_rewards.append(ep_reward)

        if ep % 50 == 0:
            print(f"  Episode {ep}: mean_error={np.mean(ep_errors):.4f}m reward={ep_reward:.1f}")

    print(f"\n{arm_type.upper()} summary over {n_episodes} episodes:")
    print(f"  Mean error:  {np.mean(all_errors):.4f} +/- {np.std(all_errors):.4f} m")
    print(f"  Mean reward: {np.mean(all_rewards):.1f} +/- {np.std(all_rewards):.1f}")

    return all_errors, all_rewards

if __name__ == "__main__":
    f_errors, f_rewards = replay("franka", n_episodes=200)
    d_errors, d_rewards = replay("dtt",    n_episodes=200)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("200 Episode Replay — Franka vs DTT", fontsize=13)

    axes[0].plot(f_errors, color="orange", alpha=0.6, label="Franka")
    axes[0].plot(d_errors, color="blue",   alpha=0.6, label="DTT")
    axes[0].axhline(np.mean(f_errors), color="orange", linestyle="--", linewidth=1.5)
    axes[0].axhline(np.mean(d_errors), color="blue",   linestyle="--", linewidth=1.5)
    axes[0].set_title("Tracking error per episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Mean position error (m)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(f_rewards, color="orange", alpha=0.6, label="Franka")
    axes[1].plot(d_rewards, color="blue",   alpha=0.6, label="DTT")
    axes[1].axhline(np.mean(f_rewards), color="orange", linestyle="--", linewidth=1.5)
    axes[1].axhline(np.mean(d_rewards), color="blue",   linestyle="--", linewidth=1.5)
    axes[1].set_title("Reward per episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Total reward")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("results/plots/replay_200.png", dpi=150)
    print("\nPlot saved to results/plots/replay_200.png")
    plt.show()
