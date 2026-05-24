import numpy as np
import gymnasium as gym
from gymnasium import spaces
import gymnasium_robotics
import os

gym.register_envs(gymnasium_robotics)

class TokamakArmEnv(gym.Env):
    def __init__(self, arm_type="franka", noise_std=0.01, action_delay=2):
        super().__init__()
        
        self.arm_type    = arm_type
        self.noise_std   = noise_std
        self.action_delay = action_delay

        # use built-in FetchReach as base for franka
        # for dtt we use same env but different reward shaping
        self.base_env = gym.make("FetchReachDense-v4", max_episode_steps=200)
        
        self.T = 200
        self.A = 0.15
        self.t = 0

        # action delay buffer
        n_actions = self.base_env.action_space.shape[0]
        self.action_buffer = [np.zeros(n_actions)] * self.action_delay

        # wall limits (port constraints)
        self.wall_x   = 0.325
        self.wall_z   = 1.0
        self.center   = self.base_env.unwrapped.initial_gripper_xpos if hasattr(
            self.base_env.unwrapped, 'initial_gripper_xpos') else np.array([1.3, 0.75, 0.6])

        # spaces
        obs      = self.base_env.observation_space
        obs_size = (obs["observation"].shape[0] +
                    obs["desired_goal"].shape[0] + 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = self.base_env.action_space

    def _get_target(self, t):
        phase = 2 * np.pi * t / self.T
        # figure-eight in base env coordinate frame
        center = np.array([1.3, 0.75, 0.6])
        x = center[0] + self.A * np.sin(phase)
        y = center[1] + self.A * np.sin(2 * phase) * 0.5
        z = center[2]
        return np.array([x, y, z])

    def _get_obs(self, raw_obs):
        obs     = raw_obs["observation"]
        obs    += np.random.normal(0, self.noise_std, obs.shape)
        target  = self._get_target(self.t)
        phase   = np.array([self.t / self.T])
        return np.concatenate([obs, target, phase]).astype(np.float32)

    def _get_reward(self, raw_obs, action):
        ee_pos      = raw_obs["observation"][:3]
        target      = self._get_target(self.t)
        prev_action = self.action_buffer[-1]

        # tracking
        pos_error  = np.linalg.norm(ee_pos - target)
        r_track    = 5.0 * np.exp(-10 * pos_error**2)

        # smoothness
        r_smooth   = -0.05 * np.sum((action - prev_action)**2)

        # collision — distance from port walls
        # dtt arm gets bonus for staying closer to center
        dist_wall  = self.wall_x - abs(ee_pos[0] - 1.3)
        r_collision = -0.1 * max(0, 0.05 - dist_wall)**2

        # dtt bonus — domain specific arm penalises wall proximity more
        if self.arm_type == "dtt":
            r_collision *= 1.5

        return r_track + r_smooth + r_collision, pos_error

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raw_obs, _ = self.base_env.reset(seed=seed)
        self.t = 0
        n_actions = self.action_space.shape[0]
        self.action_buffer = [np.zeros(n_actions)] * self.action_delay
        return self._get_obs(raw_obs), {}

    def step(self, action):
        # action delay
        self.action_buffer.append(action.copy())
        delayed_action = self.action_buffer.pop(0)

        # noise on action
        delayed_action += np.random.normal(0, self.noise_std,
                                           delayed_action.shape)
        delayed_action  = np.clip(delayed_action, -1.0, 1.0)

        raw_obs, _, terminated, truncated, info = self.base_env.step(
            delayed_action)
        self.t += 1

        obs             = self._get_obs(raw_obs)
        reward, pos_error = self._get_reward(raw_obs, action)
        terminated      = self.t >= self.T

        info["pos_error"] = pos_error
        info["ee_pos"]    = raw_obs["observation"][:3].copy()

        return obs, reward, terminated, truncated, info

    def close(self):
        self.base_env.close()
