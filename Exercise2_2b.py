"""
Reproduces the 10-armed testbed described in the textbook on reinforcement learning.

The testbed consists of 2000 randomly generated n-armed bandit tasks with n = 10.
For each action a, the rewards are selected from a normal (Gaussian) probability
distribution with mean Q*(a) and variance 1.

The program runs simulations of the softmax method using the Gibbs distribution at
different temperatures. Each simulation consists of 1000 plays of the bandit task.

The results are plotted as two graphs, one showing the increase in expected reward
with experience and the other showing the percentage of tasks in which the optimal
action was found.

The figure is saved as a PNG file.
"""

import numpy as np
import matplotlib.pyplot as plt

NUM_TASKS = 2000
NUM_ACTIONS = 10
MEAN = 0
VARIANCE = 1
NUM_PLAYS = 1000
TEMPERATURES = [0.1, 1, 2]

rewards = np.zeros((len(TEMPERATURES), NUM_TASKS, NUM_PLAYS))
optimal_actions = np.zeros((len(TEMPERATURES), NUM_TASKS, NUM_PLAYS))

for i, temperature in enumerate(TEMPERATURES):
    for j in range(NUM_TASKS):
        q_star = np.random.normal(MEAN, VARIANCE, NUM_ACTIONS)
        q_estimates = np.zeros(NUM_ACTIONS)
        action_counts = np.zeros(NUM_ACTIONS)
        task_rewards = np.zeros(NUM_PLAYS)
        task_optimal_actions = np.zeros(NUM_PLAYS)
        for k in range(NUM_PLAYS):
            action_probabilities = np.exp(q_estimates / temperature)
            action_probabilities /= np.sum(action_probabilities)
            action = np.random.choice(NUM_ACTIONS, p=action_probabilities)
            reward = np.random.normal(q_star[action], VARIANCE)
            action_counts[action] += 1
            alpha = 1.0 / action_counts[action]
            q_estimates[action] += alpha * (reward - q_estimates[action])
            task_rewards[k] = reward
            task_optimal_actions[k] = (action == np.argmax(q_star))
        rewards[i, j, :] = task_rewards
        optimal_actions[i, j, :] = task_optimal_actions

expected_rewards = np.mean(rewards, axis=1)
percent_optimal_actions = np.mean(optimal_actions, axis=1) * 100

fig, axs = plt.subplots(2, 1, figsize=(8, 8))

for i, temperature in enumerate(TEMPERATURES):
    axs[0].plot(np.arange(NUM_PLAYS), expected_rewards[i], label=f"T={temperature}")
axs[0].set_xlabel("Plays")
axs[0].set_ylabel("Average Reward")
axs[0].set_title("Softmax Agent - Gibbs Distribution")
axs[0].legend()

for i, temperature in enumerate(TEMPERATURES):
    axs[1].plot(np.arange(NUM_PLAYS), percent_optimal_actions[i], label=f"T={temperature}")
axs[1].set_xlabel("Plays")
axs[1].set_ylabel("% Optimal Action")
axs[1].legend()

plt.tight_layout()
plt.savefig("softmax.png")
plt.show()
