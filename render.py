import numpy as np
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO
from envs.arm_env import TokamakArmEnv
import time

gym.register_envs(gymnasium_robotics)

def render_arm(arm_type):
    print(f"\nRendering {arm_type.upper()} arm...")
    
    env   = TokamakArmEnv(arm_type=arm_type)
    model = PPO.load(f"models/{arm_type}/best_model")
    
    # wrap with human rendering
    render_env = gym.make("FetchReachDense-v4", 
                          max_episode_steps=200,
                          render_mode="human")
    
    obs, _ = render_env.reset()
    done   = False
    t      = 0
    
    while not done:
        # get action from trained policy
        flat_obs = np.concatenate([
            obs["observation"],
            TokamakArmEnv._get_target(env, t),
            [t / 200]
        ]).astype(np.float32)
        
        action, _ = model.predict(flat_obs, deterministic=True)
        obs, _, terminated, truncated, _ = render_env.step(action)
        done = terminated or truncated
        t += 1
        time.sleep(0.01)
    
    render_env.close()
    print(f"{arm_type.upper()} render complete.")

if __name__ == "__main__":
    for arm in ["franka", "dtt"]:
        render_arm(arm)
        time.sleep(1)
