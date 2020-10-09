import csv
import matplotlib.pyplot as plt
import numpy as np

spike_dtype = {"names": ("time", "neuron_id"), "formats": (np.float, np.int)}

input_spikes = np.loadtxt("input_spikes.csv", delimiter=",", skiprows=1, dtype=spike_dtype)
recurrent_lif_spikes = np.loadtxt("recurrent_lif_spikes.csv", delimiter=",", skiprows=1, dtype=spike_dtype)
recurrent_alif_spikes = np.loadtxt("recurrent_alif_spikes.csv", delimiter=",", skiprows=1, dtype=spike_dtype)

output_data = np.loadtxt("output.csv", delimiter=",",
                         dtype={"names": ("time", "pi1", "pi2", "pi_star1", "pi_star2"),
                                "formats": (np.float, np.float, np.float, np.float, np.float)})
# Create plot
figure, axes = plt.subplots(4, sharex=True)

axes[0].scatter(input_spikes["time"], input_spikes["neuron_id"], s=2, edgecolors="none")
axes[1].scatter(recurrent_lif_spikes["time"], recurrent_lif_spikes["neuron_id"], s=2, edgecolors="none")
axes[2].scatter(recurrent_alif_spikes["time"], recurrent_alif_spikes["neuron_id"], s=2, edgecolors="none")

axes[3].plot(output_data["time"], output_data["pi1"] - output_data["pi2"])
axes[3].axhline(0.0, linestyle="--", color="gray")

axes[0].set_title("Input")
axes[1].set_title("LIF")
axes[2].set_title("ALIF")

# Show plot
plt.show()
