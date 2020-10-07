import csv
import matplotlib.pyplot as plt
import numpy as np

spike_dtype = {"names": ("time", "neuron_id"), "formats": (np.float, np.int)}

input_spikes = np.loadtxt("input_spikes.csv", delimiter=",", skiprows=1, dtype=spike_dtype)
                              
# Create plot
figure, axis = plt.subplots(1)

axis.scatter(input_spikes["time"], input_spikes["neuron_id"], s=2, edgecolors="none")


# Show plot
plt.show()