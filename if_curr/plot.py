import csv
import matplotlib.pyplot as plt
import numpy as np

# Read CSV spikes
data = np.loadtxt("voltages.csv", delimiter=",",
                  dtype={"names": ("time", "voltage", "psi", "current"),
                         "formats": (np.float, np.float, np.float, np.float)})

# Create plot
figure, axes = plt.subplots(2)

# Plot voltages
axes[0].set_title("Voltage")
axes[0].plot(data["time"], data["voltage"])

axes[1].set_title("Psi")
axes[1].plot(data["time"], data["psi"])

# Show plot
plt.show()
