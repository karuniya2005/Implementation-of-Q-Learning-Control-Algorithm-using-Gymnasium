# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement
To implement the Q-Learning control algorithm using the Gymnasium FrozenLake-v1 environment. The agent must learn the optimal action-value function through repeated interaction with the environment and determine a policy that allows it to reach the goal while avoiding holes.


## Software Requirements
1. Python 3.x
2. Gymnasium
3. NumPy
4. Matplotlib
5. Jupyter Notebook / Google Colab / VS Code


## Environment Description
FrozenLake-v1 is a grid-world reinforcement learning environment provided by Gymnasium. The environment consists of a 4 × 4 grid with 16 states and 4 possible actions:

Action	Meaning
0	      Left
1	      Down
2	      Right
3	      Up


## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm
1.  Create the FrozenLake-v1 environment with is_slippery=False.
2.  Obtain the number of states and actions from the environment.
3.  Initialize the Q-table with zeros.
4.  Set the learning rate, discount factor, epsilon, epsilon decay rate, and number of episodes.
5.  Reset the environment at the beginning of every episode.
6.  elect an action using the epsilon-greedy strategy.
7.  Execute the selected action using env.step().
8.  Update the Q-value using the Q-Learning update equation.
9.  Continue until the episode terminates.
10. Gradually decrease epsilon while maintaining a minimum value of 0.1.
11. Repeat the training process for 20,000 episodes.
12. Obtain the learned policy using argmax(Q, axis=1).
13. Obtain the state-value function using the maximum Q-value for each state.
14. Calculate the average reward over the last 1,000 episodes.
15. Plot the moving-average learning curve.


## Python Program

```python

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

custom_map = [
    "FFFG",
    "FHFH",
    "FFFF",
    "SFFF"
]

env = gym.make(
    "FrozenLake-v1",
    desc=custom_map,
    is_slippery=False
)

start_state = 12
goal_state = 3

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 10000
max_steps_per_episode = 100

alpha = 0.1
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((
    env.observation_space.n,
    env.action_space.n
))

# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):

    if np.random.rand() < epsilon:
        return env.action_space.sample()

    return np.argmax(Q[state])

# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):

    state, _ = env.reset()

    total_reward = 0

    for step in range(max_steps_per_episode):

        action = epsilon_greedy_action(
            state, epsilon
        )

        next_state, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

        # Q-Learning update
        if terminated or truncated:

            Q[state, action] += alpha * (
                reward - Q[state, action]
            )

            break

        Q[state, action] += alpha * (
            reward
            + gamma * np.max(Q[next_state])
            - Q[state, action]
        )

        state = next_state

    episode_rewards.append(total_reward)

    # Epsilon decay
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):

    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)

# -------------------------------------------------
# Output
# -------------------------------------------------

state_values = np.max(Q, axis=1)

learned_policy = np.argmax(Q, axis=1)

print("KARUNIYA M")
print("212223240068")

print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)

print_policy(learned_policy)

average_reward = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    average_reward
)

# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Q-Learning Learning Curve - FrozenLake")
plt.grid(True)
plt.show()

env.close()

```
---

## Output

## Final Q-table:

<img width="330" height="440" alt="image" src="https://github.com/user-attachments/assets/57ef104e-d38c-4ca5-99fa-6c10dfdd60a9" />




## Estimated State-Value Function:

<img width="378" height="122" alt="image" src="https://github.com/user-attachments/assets/5b66569b-0a7a-421e-8b04-7d4d6b894e59" />






## Learned Policy:

<img width="286" height="121" alt="image" src="https://github.com/user-attachments/assets/99f7ef34-ae11-4081-8d49-ede3a55ac33a" />





## Average reward over last 1000 episodes: 
<img width="488" height="28" alt="image" src="https://github.com/user-attachments/assets/fadde726-0ccf-4eb4-b8f5-f8e37ae53a19" />

## Q-Learning Learning Curve - FrozenLake
<img width="701" height="487" alt="image" src="https://github.com/user-attachments/assets/bce7c80d-0db4-4f1a-a305-2c72db69d705" />


---

## Result

The Q-Learning control algorithm was successfully implemented
using the Gymnasium FrozenLake-v1 environment.

The agent learned an action-value function through repeated
interaction with the environment and obtained a learned policy
for selecting actions. The learned Q-table, state-value function,
policy, and average reward were successfully obtained.

---

## Inference
Q-Learning successfully learns an optimal policy by updating the
Q-table based on the rewards obtained from the environment.

The epsilon-greedy strategy allows the agent to explore different
actions initially and gradually exploit the learned Q-values as
epsilon decreases.

After sufficient training, the agent learns suitable actions for
moving from the starting state toward the goal while avoiding
holes. The learning curve indicates the improvement in the agent's
performance during training.

---

