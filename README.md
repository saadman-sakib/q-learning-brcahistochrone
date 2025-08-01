# Reinforcement Learning (Q-Learning) for Time-Optimal Path Planning (Brachistochrone)

## Overview

This repository implements a simple Q-learning agent that learns to navigate a 100×100 grid from a start position to an end position in a time-optimal manner, subject to gravity-influenced movement dynamics. The agent receives higher rewards for shorter traversal times.

## Features

* **Grid Environment:** 100×100 grid.
* **Dynamics:** Vertical movements are influenced by gravity; horizontal movements are unaffected.
* **Actions:** Move right, up, or down by one cell.
* **Reward:** Inverse of total time taken:

  $$
  R = \frac{C}{T}
  $$

  where $C = 1000$ is a scaling constant.
* **Learning:** Q-learning with epsilon-greedy exploration.

## Equations and Algorithms

### 1. Time to Traverse an Action

$$
\Delta t = \frac{1}{\sqrt{2 g \cdot y_{\text{move}}}}
$$

* $g$: gravitational acceleration (9.8 m/s²)
* $y_{\text{move}}$: vertical coordinate of the **destination** cell

### 2. Q-Value Update Rule

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'}Q(s',a') - Q(s,a) \right]
$$

* Learning rate: $\alpha = 0.5$
* Discount factor: $\gamma = 1$ (future rewards fully weighted)
* Reward $r = \frac{1000}{\text{total\_time}}$ upon reaching the goal; otherwise 0

### 3. Epsilon-Greedy Policy

$$
a = 
\begin{cases}
\text{random action} & \text{with probability } \epsilon \\
\arg\max_{a'} Q(s,a') & \text{with probability } 1 - \epsilon
\end{cases}
$$

* $\epsilon = 0.1$

### 4. State and Action Discretization

* States and actions are rounded to one decimal place:
  $(x, y) \rightarrow (\text{round}(x,1), \text{round}(y,1))$

---

## Code Structure

### `Agent` Class

* **Initialization:** Loads a pre-trained Q-table from `model_test.pickle` if available.
* **Action Checking:** Filters out invalid moves (e.g., out-of-bounds, revisits, downward moves below $y = 50$).
* **Available Actions:**

  * Right: $(x+1, y)$
  * Up: $(x, y-1)$
  * Down: $(x, y+1)$
* **Time Calculation:** `dt()` method computes traversal time.
* **Q-Table Methods:** `get_q_value()`, `update_q_value()`, `best_future_reward()`.
* **Training:** `train(n_episodes)` runs Q-learning for a given number of episodes.
* **Policy Execution:** `walk()` returns a deterministic, shortest-time path using the learned policy.

---

## Installation

```bash
git clone <repo_url>
cd <repo_dir>
```

### Create a virtual environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install dependencies

```bash
pip install matplotlib
```

---

## Usage Example

```python
from agent import Agent

agent = Agent()
agent.train(10000)  # Train for 10,000 episodes
agent.save_model("model_test.pickle")  # Save model

# Visualize learned path
path = agent.walk()

import matplotlib.pyplot as plt
x_coords = [p[0] for p in path]
y_coords = [-p[1] for p in path]  # Invert Y for proper plotting
plt.scatter(x_coords, y_coords)
plt.title("Learned Path from Start to Goal")
plt.xlabel("X")
plt.ylabel("-Y")
plt.show()
```

---

## Notes and Possible Extensions

* **Discount Factor:** Currently $\gamma = 1$. Use $0 < \gamma < 1$ to emphasize shorter paths.
* **Action Space:** Add diagonal or multi-cell movements.
* **State Representation:** Use continuous coordinates instead of rounding.
* **Visualization:** Animate the agent’s learning progress across episodes.

