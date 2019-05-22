import csv
import matplotlib.pyplot as plt
import numpy as np

# Read CSV spikes and weights
spikes = np.loadtxt("spikes.csv", delimiter=",", skiprows=1,
                    dtype={"names": ("time", "neuron_id"),
                           "formats": (np.float, np.int)})
weights = np.loadtxt("weights.csv", delimiter=",", skiprows=1,
                     dtype={"names": ("time", "weight"),
                            "formats": (np.float, np.float)})

# Create plot
figure, axes = plt.subplots(3, sharex=True)

# Plot spikes
axes[0].scatter(spikes["time"], spikes["neuron_id"], s=2, edgecolors="none")

# Plot rates
bins = np.arange(0, 10000 + 1, 10)
rate = np.histogram(spikes["time"], bins=bins)[0] *  (1000.0 / 10.0) * (1.0 / 2000.0)
axes[1].plot(bins[0:-1], rate)

# Plot weight evolution
axes[2].plot(weights["time"], weights["weight"])

axes[0].set_title("Spikes")
axes[1].set_title("Firing rates")
axes[2].set_title("Weight evolution")

axes[0].set_xlim((0, 10000))
axes[0].set_ylim((0, 2000))

axes[0].set_ylabel("Neuron number")
axes[1].set_ylabel("Mean firing rate [Hz]")
axes[2].set_ylabel("Mean I->E weights [nA]")

axes[2].set_xlabel("Time [ms]")

# Show plot
plt.show()

