import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from envs.arm_env import TokamakArmEnv

def train_arm(arm_type, timesteps=3_000_000):
    print(f"\n{'='*50}")
    print(f"Training {arm_type.upper()} arm")
    print(f"{'='*50}\n")

    # make envs
    env      = make_vec_env(lambda: TokamakArmEnv(arm_type=arm_type), n_envs=4)
    eval_env = make_vec_env(lambda: TokamakArmEnv(arm_type=arm_type), n_envs=1)

    # save paths
    save_dir = f"results/{arm_type}"
    os.makedirs(save_dir, exist_ok=True)

    # eval callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/{arm_type}",
        log_path=save_dir,
        eval_freq=10_000,
        n_eval_episodes=5,
        verbose=1
    )

    # model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=f"results/tensorboard/{arm_type}"
    )

    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback,
        progress_bar=True
    )

    model.save(f"models/{arm_type}/final_model")
    print(f"\n{arm_type.upper()} training complete.")
    return model

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    franka_model = train_arm("franka", timesteps=3_000_000)
    dtt_model    = train_arm("dtt",    timesteps=3_000_000)

    print("\nBoth arms trained. Run evaluate.py to compare.")
