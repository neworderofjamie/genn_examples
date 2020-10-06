import csv
import matplotlib.pyplot as plt
import numpy as np

from glob import glob
from os import path

spike_dtype = {"names": ("time", "neuron_id"), "formats": (np.float, np.int)}


input_spikes = sorted(list(glob("input_spikes_*.csv")))

# Create plot
figure, axes = plt.subplots(3, len(input_spikes), sharex="col", sharey="row")

for i, s in enumerate(input_spikes):
    # Extract index from filename
    index = int(s[13:-4])
    
    # Read spikes
    input_spikes = np.loadtxt(s, delimiter=",", skiprows=1,
                              dtype=spike_dtype)
    hidden_spikes = np.loadtxt("hidden_spikes_%u.csv" % index, delimiter=",", skiprows=1,
                               dtype=spike_dtype)
    output_spikes = np.loadtxt("output_spikes_%u.csv" % index, delimiter=",", skiprows=1,
                               dtype=spike_dtype)

    # Plot spikes
    axes[0, i].scatter(input_spikes["time"], input_spikes["neuron_id"], s=2, edgecolors="none")
    axes[1, i].scatter(hidden_spikes["time"], hidden_spikes["neuron_id"], s=2, edgecolors="none")
    axes[2, i].scatter(output_spikes["time"], output_spikes["neuron_id"], s=2, edgecolors="none")

    axes[2, i].set_xlabel("Time [ms]")

axes[0, 0].set_ylabel("Neuron number")
axes[1, 0].set_ylabel("Neuron number")
axes[2, 0].set_ylabel("Neuron number")
# Show plot
plt.show()

