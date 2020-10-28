import csv
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv

input_spikes = read_csv("input_spikes.csv", header=None, names=["time", "neuron_id"], skiprows=1, delimiter=",",
                        dtype={"time":float, "neuron_id":int})
recurrent_lif_spikes = read_csv("recurrent_lif_spikes.csv", header=None, names=["time", "neuron_id"], skiprows=1, delimiter=",",
                                dtype={"time":float, "neuron_id":int})
recurrent_alif_spikes = read_csv("recurrent_alif_spikes.csv", header=None, names=["time", "neuron_id"], skiprows=1, delimiter=",",
                                 dtype={"time":float, "neuron_id":int})
output_data = read_csv("output.csv", header=None, index_col=False, names=["time", "pi1", "pi2", "pi_star1", "pi_star2"], skiprows=0, delimiter=",",
                        dtype={"time":float, "pi1":float, "pi2":float, "pi_star1":float, "pi_star2":float})
times = np.loadtxt("times.csv")

# When decisions are made, pi* is increased above zero
decision_mask = (output_data["pi_star1"] >= 0.0)

# Extract output data during decision time and reshape so each row contains a single trials decision
# **NOTE** for some reason reshape doesn't work correctly on pandas data frames
pi1_output = np.reshape(np.asarray(output_data["pi1"][decision_mask]), (-1, 150))
pi2_output = np.reshape(np.asarray(output_data["pi2"][decision_mask]), (-1, 150))
pi_star1_output = np.reshape(np.asarray(output_data["pi_star1"][decision_mask]), (-1, 150))
pi_star2_output = np.reshape(np.asarray(output_data["pi_star2"][decision_mask]), (-1, 150))

# Calculate sum of each output for each trial
pi1_output = np.sum(pi1_output, axis=1)
pi2_output = np.sum(pi2_output, axis=1)
pi_star1_output = np.sum(pi_star1_output, axis=1)
pi_star2_output = np.sum(pi_star2_output, axis=1)

# Stack outputs together into 2 row matrixes
pi_output = np.vstack((pi1_output, pi2_output))
pi_star_output = np.vstack((pi_star1_output, pi_star2_output))

# Find largest output for each trial
pi_decision = np.argmax(pi_output, axis=0)
pi_star_decision = np.argmax(pi_star_output, axis=0)

# Count correct decisions per epoch
correct_decision = (pi_decision == pi_star_decision)
correct_decision = np.reshape(correct_decision, (-1, 64))
correct_decision = np.sum(correct_decision, axis=1)
print(correct_decision)

# Create plot
figure, axes = plt.subplots(5, sharex=True)

axes[0].set_title("Input")
axes[0].scatter(input_spikes["time"], input_spikes["neuron_id"], s=2, edgecolors="none")
axes[0].vlines(times, 0, 40, linestyle="--", color="gray")

axes[1].set_title("LIF")
axes[1].scatter(recurrent_lif_spikes["time"], recurrent_lif_spikes["neuron_id"], s=2, edgecolors="none")
axes[1].vlines(times, 0, 50, linestyle="--", color="gray")

axes[2].set_title("ALIF")
axes[2].scatter(recurrent_alif_spikes["time"], recurrent_alif_spikes["neuron_id"], s=2, edgecolors="none")
axes[2].vlines(times, 0, 50, linestyle="--", color="gray")

axes[3].set_title("Output 1")
axes[3].plot(output_data["time"], output_data["pi1"], label="$\pi_1$")
axes[3].plot(output_data["time"], output_data["pi_star1"], label="$\pi^*_1$")
axes[3].axhline(0.5, linestyle="--", color="gray")
axes[3].vlines(times, 0.0, 1.0, linestyle="--", color="gray")
axes[3].legend(loc="upper right")

axes[4].set_title("Output 2")
axes[4].plot(output_data["time"], output_data["pi2"], label="$\pi_2$")
axes[4].plot(output_data["time"], output_data["pi_star2"], label="$\pi^*_2$")
axes[4].axhline(0.5, linestyle="--", color="gray")
axes[4].vlines(times, 0.0, 1.0, linestyle="--", color="gray")
axes[4].legend(loc="upper right")

# Show plot
plt.show()
