import matplotlib.pyplot as plt
import numpy as np

# Read CSV spikes
spikes = np.loadtxt("spikes.csv", delimiter=",", skiprows=1,
                    dtype={"names": ("time", "neuron_id"),
                           "formats": (np.float, np.int)})

# Create plot
figure, axis = plt.subplots()

# Plot voltages
axis.scatter(spikes["time"], spikes["neuron_id"], s=2, edgecolors="none")

# Show plot
plt.show()
