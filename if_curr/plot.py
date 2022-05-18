import csv
import matplotlib.pyplot as plt
import numpy as np

# Read CSV spikes
data = np.loadtxt("voltages.csv", delimiter=",",
                  dtype={"names": ("time", "voltage", "current"),
                         "formats": (np.float, np.float, np.float)})

# Create plot
figure, axis = plt.subplots()

current_axis = axis.twinx()

# Plot voltages
axis.plot(data["time"], data["voltage"], color="red")
current_axis.plot(data["time"], data["current"], color="blue")

axis.set_xlabel("Time [ms]")
axis.set_ylabel("Membrane voltage [mV]")
current_axis.set_ylabel("Input current [nA]")

# Show plot
plt.show()
