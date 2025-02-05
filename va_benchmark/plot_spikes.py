import csv
import matplotlib.pyplot as plt
import numpy as np

# Read CSV spikes
spikes_e = np.loadtxt("spikes_e.csv", delimiter=",", skiprows=1,
                      dtype={"names": ("time", "neuron_id"),
                             "formats": (float, int)})
spikes_i = np.loadtxt("spikes_i.csv", delimiter=",", skiprows=1,
                      dtype={"names": ("time", "neuron_id"),
                             "formats": (float, int)})

num_e = np.amax(spikes_e["neuron_id"]) + 1
num_i = np.amax(spikes_i["neuron_id"]) + 1

# Create plot
figure, axes = plt.subplots(2, sharex=True)

# Plot spikes
axes[0].scatter(spikes_e["time"], spikes_e["neuron_id"], s=2, edgecolors="none", color="red")
axes[0].scatter(spikes_i["time"], spikes_i["neuron_id"] + num_e, s=2, edgecolors="none", color="blue")

# Plot rates
bins = np.arange(0, 10000 + 1, 10)
e_rate = np.histogram(spikes_e["time"], bins=bins)[0] *  (1000.0 / 10.0) * (1.0 / num_e)
i_rate = np.histogram(spikes_i["time"], bins=bins)[0] *  (1000.0 / 10.0) * (1.0 / num_i)
axes[1].plot(bins[0:-1], e_rate, color="red")
axes[1].plot(bins[0:-1], i_rate, color="blue")

axes[0].set_title("Spikes")
axes[1].set_title("Firing rates")

axes[0].set_xlim((0, 10000))
axes[0].set_ylim((0, num_i + num_e))

axes[0].set_ylabel("Neuron number")
axes[1].set_ylabel("Mean firing rate [Hz]")

axes[1].set_xlabel("Time [ms]")

# Show plot
plt.show()

