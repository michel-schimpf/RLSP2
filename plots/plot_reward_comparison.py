import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
seaborn.set()

plt.style.use("plots/subfigure.mplstyle")


number_rows = 5000000
values_combined = 300
monitor = np.loadtxt("plots/normal.csv", delimiter=',', skiprows=2, max_rows=number_rows,usecols=(0,1,2,4))


fig, ax = plt.subplots(1, 3, constrained_layout=True)
print(type(ax))
# time steps:
x_time_steps_values = monitor[:, 1]
size = len(x_time_steps_values) - (len(x_time_steps_values) % values_combined)
x_time_steps_values = x_time_steps_values[:size]
x_time_steps_values = x_time_steps_values.reshape((int(len(x_time_steps_values)/values_combined), values_combined))
x_time_steps_values = np.sum(x_time_steps_values, axis=1)
x_intervals = [0]
for j in x_time_steps_values:
    new_time_step = x_intervals[-1] + j
    x_intervals.append(new_time_step)
x_intervals = np.array(x_intervals)


monitor = monitor[:size]
# success rate:
successes = monitor[:, 3]
successes = successes.reshape((int(size/values_combined), values_combined))
average_success_rates = np.average(successes, axis=1)
average_success_rates = np.insert(average_success_rates, 0, 0)

# reward:
rewards = monitor[:, 0]
rewards = rewards.reshape((int(size/values_combined), values_combined))
average_rewards = np.average(rewards, axis=1)
average_rewards = np.insert(average_rewards, 0, 0)

# episode length:
lengths = monitor[:, 1]
lengths = lengths.reshape((int(size/values_combined), values_combined))
average_lengths = np.average(lengths, axis=1)
average_lengths = np.insert(average_lengths, 0, 0)
print(average_lengths)

# delete some values to make it more spiky

for _ in range(3):
    x_intervals = np.delete(x_intervals, np.arange(0, x_intervals.size, 2))
    average_success_rates = np.delete(average_success_rates, np.arange(0, len(average_success_rates), 2))
    average_rewards = np.delete(average_rewards, np.arange(0, len(average_rewards), 2))
    average_lengths = np.delete(average_lengths, np.arange(0, len(average_lengths), 2))

print(max(average_success_rates))
print(ax[0])
ax[0].plot(x_intervals, average_success_rates, linewidth=1.3, label="Success Rate")
ax[0].set_ylim([-0.05, 1])
ax[0].set_xlabel("Steps")
ax[0].set_ylabel("Success Rate")

ax[1].plot(x_intervals, average_rewards, linewidth=1.3, label="Episode Reward")
ax[1].set_ylim([-20.5,0.5,])
ax[1].set_xlabel("Steps")
ax[1].set_ylabel("Episode Reward")

ax[2].plot(x_intervals, average_lengths, linewidth=1.3, label="Episode Length")
ax[2].set_ylim([-0.5,20.5])
ax[2].set_xlabel("Steps")
ax[2].set_ylabel("Episode Length")

plt.show()