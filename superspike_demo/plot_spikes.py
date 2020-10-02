import csv
import matplotlib.pyplot as plt
import numpy as np


spike_dtype = {"names": ("time", "neuron_id"), "formats": (np.float, np.int)}
spike_times = list(range(0, 600, 100))

# Create plot
figure, axes = plt.subplots(3, len(spike_times), sharex=True)

for i, t in enumerate(spike_times):
    # Read spikes
    input_spikes = np.loadtxt("input_spikes_%u.csv" % t, delimiter=",", skiprows=1,
                              dtype=spike_dtype)
    hidden_spikes = np.loadtxt("hidden_spikes_%u.csv" % t, delimiter=",", skiprows=1,
                               dtype=spike_dtype)
    output_spikes = np.loadtxt("output_spikes_%u.csv" % t, delimiter=",", skiprows=1,
                               dtype=spike_dtype)

    # Plot spikes
    axes[0, i].scatter(input_spikes["time"], input_spikes["neuron_id"], s=2, edgecolors="none")
    axes[1, i].scatter(hidden_spikes["time"], hidden_spikes["neuron_id"], s=2, edgecolors="none")
    axes[2, i].scatter(output_spikes["time"], output_spikes["neuron_id"], s=2, edgecolors="none")


    #axis.set_ylabel("Neuron number")
    #axis.set_xlabel("Time [ms]")

# Show plot
plt.show()

