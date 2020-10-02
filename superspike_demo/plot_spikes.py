import csv
import matplotlib.pyplot as plt
import numpy as np


spike_dtype = {"names": ("time", "neuron_id"), "formats": (np.float, np.int)}
# Create plot
figure, axes = plt.subplots(2, 5)

for i in range(5):
    # Read spikes
    input_spikes = np.loadtxt("input_spikes_%u.csv" % i, delimiter=",", skiprows=1,
                              dtype=spike_dtype)
    output_spikes = np.loadtxt("output_spikes_%u.csv" % i, delimiter=",", skiprows=1,
                               dtype=spike_dtype)

    # Plot spikes
    axes[0, i].scatter(input_spikes["time"], input_spikes["neuron_id"], s=2, edgecolors="none")
    axes[1, i].scatter(output_spikes["time"], output_spikes["neuron_id"], s=2, edgecolors="none")


    #axis.set_ylabel("Neuron number")
    #axis.set_xlabel("Time [ms]")

# Show plot
plt.show()

