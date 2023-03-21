"""
Reproduces the 10-armed testbed described in the textbook, Reinforcement Learning,
by Sutton and Barto, on reinforcement learning.

The testbed consists of 2000 randomly generated n-armed bandit tasks with n = 10.
For each action a, the rewards are selected from a normal (Gaussian) probability
distribution with mean Q*(a) and variance 1.

The program runs simulations of the greedy method and two e-greedy methods (e = 0.01
and e = 0.1), using the sample-average technique to form action-value estimates.
Each simulation consists of 1000 plays of the bandit task.

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
EPSILONS = [0, 0.01, 0.1]

rewards = np.zeros((len(EPSILONS), NUM_TASKS, NUM_PLAYS))
optimal_actions = np.zeros((len(EPSILONS), NUM_TASKS, NUM_PLAYS))

for i, epsilon in enumerate(EPSILONS):
    for j in range(NUM_TASKS):
        q_star = np.random.normal(MEAN, VARIANCE, NUM_ACTIONS)
        q_estimates = np.zeros(NUM_ACTIONS)
        action_counts = np.zeros(NUM_ACTIONS)
        task_rewards = np.zeros(NUM_PLAYS)
        task_optimal_actions = np.zeros(NUM_PLAYS)
        for k in range(NUM_PLAYS):
            if np.random.rand() < epsilon:
                action = np.random.choice(NUM_ACTIONS)
            else:
                action = np.argmax(q_estimates)
            reward = np.random.normal(q_star[action], VARIANCE)
            action_counts[action] += 1
            alpha = 1.0 / action_counts[action]
            q_estimates[action] += alpha * (reward - q_estimates[action])
            task_rewards[k] = reward
            task_optimal_actions[k] = (action == np.argmax(q_star))
        rewards[i, j, :] = task_rewards
        optimal_actions[i, j, :] = task_optimal_actions

mean_rewards = np.mean(rewards, axis=1)
percent_optimal = 100 * np.mean(optimal_actions, axis=1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for i, epsilon in enumerate(EPSILONS):
    plt.plot(mean_rewards[i], label=f'Epsilon = {epsilon}')
plt.xlabel('Plays')
plt.ylabel('Average reward')
plt.legend()

plt.subplot(1, 2, 2)
for i, epsilon in enumerate(EPSILONS):
    plt.plot(percent_optimal[i], label=f'Epsilon = {epsilon}')
plt.xlabel('Plays')
plt.ylabel('% Optimal action')
plt.ylim(0, 100)
plt.legend()

plt.savefig('10_armed_testbed.png')
plt.show()
