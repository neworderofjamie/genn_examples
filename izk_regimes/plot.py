import matplotlib.pyplot as plt
import numpy as np



# Read CSV spikes
data = np.loadtxt("voltages.csv", delimiter=",", skiprows=1,
                  dtype={"names": ("time", "neuron_id", "voltage"),
                         "formats": (float, int, float)})

# Check there are 4 neurons
neuron_ids = np.unique(data["neuron_id"])
assert len(neuron_ids) == 4

# Create plot
figure, axes = plt.subplots(2, 2)

# Loop through neurons
for n in neuron_ids:
    # Convert neuron id to axex coordinate
    x = n // 2
    y = n % 2

    # Pick rows corresponding to this neuron
    mask = data["neuron_id"] == n

    # Plot voltages
    axes[x, y].plot(data["time"][mask], data["voltage"][mask])

# Show plot
plt.show()
