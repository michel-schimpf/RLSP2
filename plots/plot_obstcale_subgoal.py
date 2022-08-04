import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
seaborn.set()

plt.style.use("plots/subfigure.mplstyle")


number_rows = 5000000
values_combined = 500
monitor = np.loadtxt("plots/PPO_0).monitor.csv", delimiter=',', skiprows=2, max_rows=number_rows,usecols=(0,1,2,4))


fig, ax = plt.subplots(constrained_layout=True)

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


# # delete some values to make it more spiky
for _ in range(4):
    x_intervals = np.delete(x_intervals, np.arange(0, x_intervals.size, 2))
    average_success_rates = np.delete(average_success_rates, np.arange(0, len(average_success_rates), 2))

ax.plot(x_intervals, average_success_rates, linewidth=1.3, label="Success Rate")
ax.set_xlabel("Steps")
ax.set_ylabel("Success Rate")
plt.legend(loc="lower right")
plt.show()