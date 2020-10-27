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
times = np.loadtxt("times.csv")
#epsilon_data = np.loadtxt("epsilon.csv", delimiter=",",
#                         dtype={"names": ("time", "epsilon_0", "epsilon_1", "epsilon_2", "psi_0", "psi_1", "psi_2", "v0", "v1", "v2", "a0", "a1", "a2"),
#                                "formats": (np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float, np.float)})


# Create plot
figure, axes = plt.subplots(5, sharex=True)

axes[0].set_title("Input")
axes[0].scatter(input_spikes["time"], input_spikes["neuron_id"], s=2, edgecolors="none")
axes[0].vlines(times, 0, 40, linestyle="--", color="gray")

axes[1].set_title("LIF")
axes[1].scatter(recurrent_lif_spikes["time"], recurrent_lif_spikes["neuron_id"], s=2, edgecolors="none")
axes[1].vlines(times, 0, 50, linestyle="--", color="gray")

axes[2].set_title("ALIF")
axes[2].scatter(recurrent_alif_spikes["time"], recurrent_alif_spikes["neuron_id"], s=2, edgecolors="none")
axes[2].vlines(times, 0, 50, linestyle="--", color="gray")

axes[3].set_title("Output 1")
axes[3].plot(output_data["time"], output_data["pi1"], label="$\pi_1$")
axes[3].plot(output_data["time"], output_data["pi_star1"], label="$\pi^*_1$")
axes[3].axhline(0.5, linestyle="--", color="gray")
axes[3].vlines(times, 0.0, 1.0, linestyle="--", color="gray")
axes[3].legend(loc="upper right")

axes[4].set_title("Output 2")
axes[4].plot(output_data["time"], output_data["pi2"], label="$\pi_2$")
axes[4].plot(output_data["time"], output_data["pi_star2"], label="$\pi^*_2$")
axes[4].axhline(0.5, linestyle="--", color="gray")
axes[4].vlines(times, 0.0, 1.0, linestyle="--", color="gray")
axes[4].legend(loc="upper right")
"""
axes[4].set_title("Epsilon")
axes[4].plot(epsilon_data["time"], epsilon_data["epsilon_0"])
axes[4].plot(epsilon_data["time"], epsilon_data["epsilon_1"])
axes[4].plot(epsilon_data["time"], epsilon_data["epsilon_2"])

axes[5].set_title("Psi")
axes[5].plot(epsilon_data["time"], epsilon_data["psi_0"])
axes[5].plot(epsilon_data["time"], epsilon_data["psi_1"])
axes[5].plot(epsilon_data["time"], epsilon_data["psi_2"])

axes[6].set_title("V")
axes[6].plot(epsilon_data["time"], epsilon_data["v0"])
axes[6].plot(epsilon_data["time"], epsilon_data["v1"])
axes[6].plot(epsilon_data["time"], epsilon_data["v2"])

axes[7].set_title("A")
axes[7].plot(epsilon_data["time"], epsilon_data["a0"])
axes[7].plot(epsilon_data["time"], epsilon_data["a1"])
axes[7].plot(epsilon_data["time"], epsilon_data["a2"])
"""
# Show plot
plt.show()
