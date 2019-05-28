import csv
import matplotlib.pyplot as plt
import numpy as np


# Read CSV spikes
weights = np.loadtxt("weights.csv", delimiter=",", skiprows=1,
                     dtype={"names": ("delta_t", "weight"),
                            "formats": (np.float, np.float)})

weights["weight"] = (weights["weight"] - 0.5) / 0.5

# Create plot
figure, axis = plt.subplots()

# Add axis lines
axis.axhline(0.0, color="black")
axis.axvline(0.0, color="black")


# Plot voltages
axis.plot(weights["delta_t"], weights["weight"])

# Show plot
plt.show()
