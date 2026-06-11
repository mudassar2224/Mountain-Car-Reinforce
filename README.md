# 🏔️ Mountain Car — REINFORCE Policy Gradient

> A clean implementation of the REINFORCE (Monte Carlo Policy Gradient) algorithm on the classic **MountainCar-v0** environment, built from scratch using **TensorFlow / Keras** and designed to run in **Google Colab**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1csoiCsoHB1FNllWXcCqqcNs8GVHI7Hnh?usp=sharing)

---

## 📌 Problem Statement

The **MountainCar** problem is a classic reinforcement learning challenge: a car sits in a valley between two hills. The engine alone is too weak to drive straight up the slope — the agent must learn to **build momentum** by rocking back and forth, then use that momentum to reach the goal at the top of the right hill.

This makes it a great testbed for policy gradient methods because:
- The reward signal is **sparse and delayed**
- The agent must discover a **non-obvious strategy** (oscillation)
- Progress is hard to measure step-by-step

---

## 🧠 Algorithm: REINFORCE

REINFORCE is a **Monte Carlo Policy Gradient** method. Instead of learning a value function, it directly optimises the policy network by computing the gradient of expected return.

**Core update rule:**

```
∇J(θ) = E[ ∇ log π_θ(a|s) · G_t ]
```

Where `G_t` is the discounted return from timestep `t`.

### Key Design Choices

| Component | Choice | Reason |
|---|---|---|
| Policy Network | 2 hidden layers × 64 units, ReLU, He init | Stable gradient flow |
| Output | Softmax over 3 actions | Proper probability distribution |
| Optimizer | Adam with `clipnorm=1.0` | Prevents gradient explosion |
| Returns | Normalised (zero mean, unit std) | Reduces variance |
| Entropy Bonus | `β = 0.02` | Encourages exploration |
| Discount Factor | `γ = 0.999` | Values long-horizon strategies |

---

## 🏗️ Project Structure

```
Mountain-Car-Reinforce/
│
├── Mountain_climbing_Using_Reinforce_Algorithm.ipynb   # Main notebook
└── README.md
```

### Notebook Cells at a Glance

| Cell | Purpose |
|---|---|
| 1 | Imports & global setup (TensorFlow, NumPy, Matplotlib) |
| 2 | Custom `MountainCarEnv` — physics, reward shaping, state normalisation |
| 3 | `PolicyNetwork` — Keras stochastic policy π_θ(a\|s) |
| 4 | `REINFORCEAgent` — episode collection, return computation, gradient update |
| 5 | Training loop with early stopping (target: 80% success rate) |
| 6+ | Evaluation, animated visualisation, training curves |

---

## ⚙️ Environment Details

The custom `MountainCarEnv` closely follows the OpenAI Gym spec:

- **State**: `[position, velocity]` (normalised to `[0, 1]`)
- **Actions**: `0` = push left, `1` = no push, `2` = push right
- **Goal**: reach position ≥ 0.45
- **Reward shaping**:
  - `+100` on reaching the goal
  - Small continuous bonuses for height gain, rightward velocity, and forward progress
  - `-0.1` per step to encourage efficiency

---

## 🚀 Quick Start

### Run in Google Colab (Recommended)

Click the badge above, or go directly:  
🔗 [Open Colab Notebook](https://colab.research.google.com/drive/1csoiCsoHB1FNllWXcCqqcNs8GVHI7Hnh?usp=sharing)

No installation needed — just **Run All**.

### Run Locally

```bash
git clone https://github.com/mudassar2224/Mountain-Car-Reinforce.git
cd Mountain-Car-Reinforce
pip install tensorflow numpy matplotlib
jupyter notebook Mountain_climbing_Using_Reinforce_Algorithm.ipynb
```

---

## 📊 Training Configuration

```python
CONFIG = dict(
    num_episodes      = 5000,
    max_steps         = 1000,
    learning_rate     = 0.001,
    gamma             = 0.999,
    entropy_beta      = 0.02,
    early_stop_rate   = 80.0,    # Stop when 80% success rate reached
    early_stop_window = 100,
)
```

Training takes approximately **5–10 minutes on CPU** or **~2 minutes on GPU/TPU** in Colab.

---

## 📈 Results

The agent learns to:
1. Build momentum by oscillating in the valley
2. Consistently reach the goal flag
3. Achieve an **80%+ success rate** well before 5000 episodes

Training curves include:
- Episode reward over time
- 50-episode moving average
- Loss curve
- Success rate (rolling 100 episodes)

---

## 🛠️ Tech Stack

- **Python 3.x**
- **TensorFlow / Keras** — policy network & gradient updates
- **NumPy** — environment physics & return computation
- **Matplotlib** — training curves & animated car visualisation
- **Google Colab** — GPU/TPU runtime

---

## 📚 References

- Williams, R.J. (1992). *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.* Machine Learning.
- Sutton & Barto — *Reinforcement Learning: An Introduction*, Chapter 13 (Policy Gradient Methods)
- [OpenAI Gym MountainCar-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car/)

---

## 👤 Author

**Mudassar**  
[![GitHub](https://img.shields.io/badge/GitHub-mudassar2224-181717?logo=github)](https://github.com/mudassar2224)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
