import matplotlib.pyplot as plt
import numpy as np

state = np.loadtxt("voltages.csv", delimiter=",",
                   dtype={"names": ("time", "v", "w"),
                          "formats": (np.float, np.float, np.float)})
# Create plot
figure, axes = plt.subplots(2)

# Plot voltages
axes[0].set_title("Voltage")
axes[0].plot(state["time"], state["v"])

axes[1].set_title("Adaption current")
axes[1].plot(state["time"], state["w"])

# Show plot
plt.show()
