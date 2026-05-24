# Tokamak Robot Arm — RL End-Effector Tracking

Humanoid.
<img width="1280" height="720" alt="Untitled design" src="https://github.com/user-attachments/assets/fbbc4060-624c-41b9-82da-c6f7f0dac788" />

# Tokamak Robot Arm — RL End-Effector Tracking
link to yuoutube video https://youtu.be/w9sFMGH7Mz4

Humanoid Internship Challenge submission — Controls Internship 2026.

## The Idea

Most submissions use a standard arm in empty space. This one connects the challenge to a real engineering problem — nuclear fusion reactor maintenance.

Real fusion reactors like the DTT (Divertor Tokamak Test) facility in Frascati, Italy need robot arms to perform maintenance inside the vessel. Humans cannot enter due to radiation. The arms must track precisely, handle sensor noise, respond to control delays, and avoid hitting the vessel walls — exactly what this challenge asks for.

We compared two arms:
- **Franka Panda** — standard 7-DOF industrial arm, well known in robotics research
- **DTT arm** — reward-shaped around real tokamak port geometry (650×600mm port, Calvo et al. 2024)

The comparison is an ablation study built into the submission — same base environment, same algorithm, same training budget. The only difference is the reward shaping. This isolates the effect of domain-specific design.

---

## Results

| Metric | Franka | DTT |
|--------|--------|-----|
| Mean tracking error | 0.0188m | 0.0185m |
| Mean episode reward | 995.4 | 995.5 |
| Training timesteps | 3M | 3M |
| Uncertainty sources | 2 | 2 |

Both arms achieve near-maximum reward (~995/1000) under combined observation noise and action delay. DTT consistently outperforms Franka across 200 evaluation episodes — lower tracking error and higher reward.

The small but consistent difference suggests domain-specific reward shaping provides a real advantage even when the base physics are identical.

---


---

## Setup

```bash
git clone https://github.com/Shiam1995/humanoid-controls
cd humanoid-controls
conda create -p ./env python=3.10 -y
conda activate ./env
pip install mujoco gymnasium gymnasium-robotics stable-baselines3[extra] numpy matplotlib imageio imageio-ffmpeg
```

---

## How to Run

### Train both arms
```bash
python train.py
```
Trains Franka then DTT sequentially. ~40 minutes each. Models saved to `models/franka/` and `models/dtt/`.

### Evaluate and plot
```bash
python evaluate.py
```
Loads best models, runs 10 episodes each, saves comparison plots to `results/plots/`.

### Record videos
```bash
python record.py
```
Saves 5 mp4 episodes per arm to `results/videos/`.

### 200 episode replay
```bash
python replay.py
```
Runs 200 episodes per arm, plots tracking error and reward over episodes.

---

## Design Choices

### Why FetchReachDense-v4

I initially built custom MuJoCo XML files for both arms. These caused NaN instability  the links had no mass or inertia defined so the physics equations divided by zero. After trying three fixes (reduced gains, inertiafromgeom flag, smaller timestep) we switched to FetchReachDense-v4 — a pre-validated Franka arm with correct physics already defined. We added our custom reward, trajectory, noise and delay on top.

Lesson: under time pressure, build on validated environments rather than reinventing the physics.

### Trajectory

Phase-parametrised figure-eight:

```python
phase = 2 * pi * t / T
x = center[0] + A * sin(phase)
y = center[1] + A * sin(2 * phase) * 0.5
z = center[2]
```

The agent observes the phase directly — it can anticipate where the target is going rather than just reacting to where it is. This is the key design decision for smooth tracking. Most implementations only give the agent the current target position, which produces jittery reactive behaviour.

### Reward Function

Three terms:

```python
r_track     = 5.0 * exp(-10 * pos_error²)          # tracking accuracy
r_smooth    = -0.05 * sum((action - prev_action)²)  # smoothness
r_collision = -0.1 * max(0, 0.05 - dist_wall)²     # vessel wall proximity
```

**r_track** — exponential decay means small errors are heavily rewarded. The 5.0 multiplier makes this the dominant signal.

**r_smooth** — penalises large changes between consecutive actions. Without this the arm jitters. Directly addresses the challenge requirement for smooth stable motion.

**r_collision** — penalises getting within 5cm of the vessel walls. This is what connects the project to the fusion narrative. No other submission will have this term.

The DTT arm receives a 1.5x collision penalty — more conservative near walls, reflecting real reactor safety requirements where hitting a wall is catastrophic.

### Uncertainty

Two sources introduced simultaneously:

```python
# observation noise
obs += np.random.normal(0, noise_std, obs.shape)

# action delay — queue based
self.action_buffer.append(action.copy())
delayed_action = self.action_buffer.pop(0)
```

Both mimic real conditions inside a fusion reactor — sensors are noisy due to radiation interference, and electronics are radiation-hardened and slow, introducing latency.

### Algorithm

PPO (Proximal Policy Optimization) via stable-baselines3. Chosen for stability, speed, and continuous control performance. MlpPolicy — two hidden layers. 3 million timesteps per arm, 4 parallel environments.

---

## What I Would Do With More Time

**1. Fix the DTT arm visuals**
Rebuild the custom MuJoCo XML with correct link masses and inertia so both arms look physically different in the video. Currently both render as a Franka because we use FetchReachDense-v4 as the base.

**2. Helix trajectory**
Extend the figure-eight to a helical path moving along the vessel axis. More realistic for a diagnostic probe sweeping inside a tokamak. Would better demonstrate the fusion narrative.

**3. Orientation tracking**
Add quaternion tracking to the reward — the optional part of the challenge. The arm would track not just position but the angle of the end-effector. Important for welding and pipe handling tasks in real reactors.

**4. Noise sweep ablation**
Train multiple policies at different noise levels and plot how tracking error degrades. Would quantify robustness more rigorously rather than showing one fixed noise level.

**5. Longer training**
3 million steps was enough to converge but 10 million would likely reduce tracking error further and widen the gap between Franka and DTT.

**6. World model layer**
Add a learned latent dynamics model that predicts future states. Would allow the arm to plan ahead rather than just react — more relevant to fusion where plasma state changes need to be anticipated.

---

## Project Structure
