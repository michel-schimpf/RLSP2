import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
seaborn.set()

plt.style.use("plots/subfigure.mplstyle")


number_rows = 5000000
values_combined = 500
monitor = np.loadtxt("plots/peg_insert_monitor.csv", delimiter=',', skiprows=2, max_rows=number_rows)
print(monitor[1])

# time steps
time_steps = monitor[:,1]
time_steps = time_steps.reshape((int(number_rows/values_combined), values_combined))
x_time_steps_values = np.sum(time_steps, axis=1)
print(x_time_steps_values)
# number_time_steps = np.sum(x_time_steps_values)
x_intervals = [0]
for i in x_time_steps_values:
    new_time_step = x_intervals[-1] + i
    x_intervals.append(new_time_step)
print(x_intervals)
x_intervals = np.array(x_intervals)
# x value
# x_values = np.linspace(0, number_rows*20, int(number_rows/values_combined))

# rewards
rewards = monitor[:,0]
rewards = rewards.reshape((int(number_rows/values_combined), values_combined))
y_reward_values = np.average(rewards, axis=1)
y_reward_values = np.insert(y_reward_values,0,-20)

# success rate:
successes = monitor[:,3]
successes = successes.reshape((int(number_rows/values_combined), values_combined))
y_success_values = np.average(successes, axis=1)
y_success_values = np.insert(y_success_values,0,0)


# deletesome values to make it more spiky
for i in range(5):
    x_intervals = np.delete(x_intervals, np.arange(0, x_intervals.size, 2))
    y_reward_values = np.delete(y_reward_values, np.arange(0, y_reward_values.size, 2))
    y_success_values = np.delete(y_success_values, np.arange(0, y_success_values.size, 2))

figs, axs = plt.subplots(2)
axs[0].plot(x_intervals, y_reward_values, linewidth=1.0)
axs[0].set_xlabel("Steps")
axs[0].set_ylabel("Episode Reward")

axs[1].plot(x_intervals, y_success_values, linewidth=1.0)
axs[1].set_xlabel("Steps")
axs[1].set_ylabel("Success Rate")

plt.show()