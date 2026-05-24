import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.arm_env import TokamakArmEnv
import os

def evaluate_arm(arm_type, n_episodes=10):
    print(f"\nEvaluating {arm_type.upper()}...")
    
    env   = TokamakArmEnv(arm_type=arm_type)
    model = PPO.load(f"models/{arm_type}/best_model")
    
    all_errors    = []
    all_positions = []
    all_targets   = []
    episode_rewards = []

    for ep in range(n_episodes):
        obs, _     = env.reset()
        done       = False
        ep_errors  = []
        ep_reward  = 0
        positions  = []
        targets    = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_errors.append(info["pos_error"])
            ep_reward += reward
            positions.append(info["ee_pos"].copy())
            targets.append(env._get_target(env.t).copy())

        all_errors.append(ep_errors)
        episode_rewards.append(ep_reward)
        if ep == 0:
            all_positions = positions
            all_targets   = targets

    # metrics
    mean_error  = np.mean([np.mean(e) for e in all_errors])
    std_error   = np.std([np.mean(e) for e in all_errors])
    mean_reward = np.mean(episode_rewards)

    print(f"{arm_type.upper()} Results:")
    print(f"  Mean tracking error: {mean_error:.4f} +/- {std_error:.4f} m")
    print(f"  Mean episode reward: {mean_reward:.2f}")

    return {
        "arm_type":    arm_type,
        "mean_error":  mean_error,
        "std_error":   std_error,
        "mean_reward": mean_reward,
        "errors":      all_errors,
        "positions":   np.array(all_positions),
        "targets":     np.array(all_targets)
    }

def plot_comparison(franka_results, dtt_results):
    os.makedirs("results/plots", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Franka vs DTT Arm — Tokamak Vessel Tracking", fontsize=14)

    # 1. tracking error over time
    ax = axes[0, 0]
    ax.plot(np.mean(franka_results["errors"], axis=0), label="Franka", color="orange")
    ax.plot(np.mean(dtt_results["errors"],   axis=0), label="DTT",    color="blue")
    ax.set_title("Mean Tracking Error Over Episode")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Position Error (m)")
    ax.legend()
    ax.grid(True)

    # 2. bar chart mean error
    ax = axes[0, 1]
    arms   = ["Franka", "DTT"]
    errors = [franka_results["mean_error"], dtt_results["mean_error"]]
    stds   = [franka_results["std_error"],  dtt_results["std_error"]]
    bars   = ax.bar(arms, errors, yerr=stds, color=["orange", "blue"], alpha=0.7, capsize=5)
    ax.set_title("Mean Tracking Error Comparison")
    ax.set_ylabel("Position Error (m)")
    ax.grid(True, axis="y")

    # 3. trajectory tracking xy
    ax = axes[1, 0]
    ax.plot(franka_results["targets"][:, 0],
            franka_results["targets"][:, 2],   label="Target",       color="green", linewidth=2)
    ax.plot(franka_results["positions"][:, 0],
            franka_results["positions"][:, 2],  label="Franka",       color="orange", linestyle="--")
    ax.plot(dtt_results["positions"][:, 0],
            dtt_results["positions"][:, 2],     label="DTT",          color="blue",   linestyle="--")
    ax.set_title("End-Effector Trajectory (X-Z plane)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.legend()
    ax.grid(True)

    # 4. reward comparison
    ax = axes[1, 1]
    ax.bar(arms,
           [franka_results["mean_reward"], dtt_results["mean_reward"]],
           color=["orange", "blue"], alpha=0.7)
    ax.set_title("Mean Episode Reward")
    ax.set_ylabel("Reward")
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig("results/plots/comparison.png", dpi=150)
    print("\nPlot saved to results/plots/comparison.png")
    plt.show()

if __name__ == "__main__":
    franka_results = evaluate_arm("franka")
    dtt_results    = evaluate_arm("dtt")
    plot_comparison(franka_results, dtt_results)
