# Reinforcement Learning (Q-Learning) for Time-Optimal Path Planning (Brachistochrone)

## Overview

This repository implements a simple Q-learning agent that learns to navigate a 100×100 grid from a start position to an end position in a time-optimal manner, subject to gravity-influenced movement dynamics. The agent receives a higher reward for shorter traversal times.

## Features

* **Grid Environment:** 100×100 grid.
* **Dynamics:** Vertical movements influenced by gravity; horizontal movements unaffected.
* **Actions:** Move right, up, or down by one cell.
* **Reward:** Inverse of total time taken: $R = \frac{C}{T}$ (where $C=1000$ is a scaling constant).
* **Learning:** Q-learning with epsilon-greedy exploration.

## Equations and Algorithms

1. **Time to traverse an action**:

   $$
   \Delta t = \frac{1}{\sqrt{2 g \cdot y_{\text{move}}}}
   $$

   where:

   * $g$ is gravitational acceleration (9.8 m/s²).
   * $y_{\text{move}}$ is the vertical coordinate of the destination cell.

2. **Q-value update rule**:

   $$
   Q(s,a) \leftarrow Q(s,a) + \alpha \Bigl[ r + \gamma \max_{a'}Q(s',a') - Q(s,a) \Bigr]
   $$

   In this implementation:

   * Learning rate $\alpha = 0.5$.
   * Discount factor $\gamma = 1$ implicitly, since future rewards carry full weight.
   * Reward $r = \frac{1000}{\text{total_{time}}}$ upon reaching the goal; 0 otherwise.

3. **Epsilon-greedy policy**:

   $$
   a = \begin{cases}
     \text{random action} & \text{with probability } \epsilon \\
     \arg\max_{a'} Q(s,a') & \text{with probability } 1-\epsilon
   \end{cases}
   $$

   where $\epsilon = 0.1$$.

4. **State and Action Discretization**:

   * States and actions are rounded to one decimal place: $(x,y) \to (\text{round}(x,1),\text{round}(y,1))$.

## Code Structure

* `Agent` class implements:

  * **Initialization**: Loads a pre-trained Q-table from `model_test.pickle` if available.
  * **Action Checking**: Filters out invalid moves (out-of-bounds, revisits, downward moves below y=50, etc.).
  * **Available Actions**: Right `(x+1,y)`, Up `(x,y-1)`, Down `(x,y+1)`.
  * **Time Calculation**: `dt()` method for computing traversal time.
  * **Q-table Methods**: `get_q_value()`, `update_q_value()`, and `best_future_reward()`.
  * **Training**: `train(n_episodes)` runs Q-learning for a specified number of episodes.
  * **Policy Execution**: `walk()` returns a deterministic shortest-time path using the learned policy.

* **Main Script**:

  1. Instantiate `Agent()`.
  2. Call `agent.train(10000)` to learn over 10,000 episodes.
  3. Save Q-table to `model_test.pickle`.
  4. Visualize a sample path via `matplotlib`.

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repo_url>
   cd <repo_dir>
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install matplotlib
   ```

## Usage

```python
from agent import Agent

agent = Agent()
# Train for 10,000 episodes
agent.train(10000)
# Save model
agent.save_model("model_test.pickle")
# Visualize learned path
path = agent.walk()

# Plot
import matplotlib.pyplot as plt
x_coords = [p[0] for p in path]
y_coords = [-p[1] for p in path]
plt.scatter(x_coords, y_coords)
plt.title("Learned Path from Start to Goal")
plt.xlabel("X")
plt.ylabel("-Y")
plt.show()
```

## Notes and Extensions

* **Discount Factor:** Currently set to 1. Introduce $0<\gamma<1$ to value shorter paths more strongly.
* **Action Space:** Extend diagonals or multi-step moves.
* **State Representation:** Use continuous coordinates without rounding.
* **Visualization:** Animate the agent’s movement across episodes.

---

*Prepared by the Reinforcement Learning Path Planning Team*
