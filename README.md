# Tokamak Robot Arm — RL End-Effector Tracking

Humanoid Internship Challenge submission.

## Setup

```bash
conda create -p ./env python=3.10 -y
conda activate ./env
pip install mujoco gymnasium gymnasium-robotics stable-baselines3[extra] numpy matplotlib imageio imageio-ffmpeg
```

## Run

```bash
python train.py      # train both arms
python evaluate.py   # compare and plot
python record.py     # save videos
python replay.py     # 200 episode stats
```

## Results

| Metric | Franka | DTT |
|--------|--------|-----|
| Mean tracking error | 0.0188m | 0.0185m |
| Mean episode reward | 995.4 | 995.5 |

## Videos

[Google Drive link]

## Design

- Figure-eight trajectory, phase-parametrised
- Reward: tracking + smoothness + collision proximity
- Uncertainty: observation noise + action delay
- PPO, 3M timesteps per arm
