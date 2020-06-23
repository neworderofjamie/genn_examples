import csv
import matplotlib.pyplot as plt
import numpy as np


# Read CSV spikes
spikes = np.loadtxt("spikes.csv", delimiter=",", skiprows=1,
                    dtype={"names": ("time", "neuron_id"),
                           "formats": (np.float, np.int)})

# Create plot
figure, axis = plt.subplots()

# Plot spikes
axis.scatter(spikes["time"], spikes["neuron_id"], s=2, edgecolors="none")

axis.set_xlim((0, 500))
axis.set_ylim((0, 100))

axis.set_ylabel("Neuron number")
axis.set_xlabel("Time [ms]")

# Show plot
plt.show()

