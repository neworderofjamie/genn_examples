import matplotlib.pyplot as plt
import numpy as np

spikes = np.loadtxt("spikes.csv", delimiter=",", skiprows=1,
                    dtype={"names": ("time", "id"),
                           "formats": (np.float, np.int)})

# Create plot
figure, axis = plt.subplots()

# Plot voltages
axis.scatter(spikes["time"], spikes["id"])

# Show plot
plt.show()
