import csv
import matplotlib.pyplot as plt
import numpy as np
import re

N_full = {
  '23': {'E': 20683, 'I': 5834},
  '4' : {'E': 21915, 'I': 5479},
  '5' : {'E': 4850, 'I': 1065},
  '6' : {'E': 14395, 'I': 2948}
}

N_scaling = 1.0

def load_spikes(filename):
    # Parse filename and use to get population name and size
    match = re.match("([0-9]+)([EI])\.csv", filename)
    name = match.group(1) + match.group(2)
    num = int(N_full[match.group(1)][match.group(2)] * N_scaling)

    print(name, num)
    # Read CSV spikes
    spikes = np.loadtxt(filename, delimiter=",", skiprows=1,
                        dtype={"names": ("time", "neuron_id"),
                            "formats": (float, int)})

    return spikes["time"], spikes["neuron_id"], name, num

pop_spikes = [load_spikes("6I.csv"),
              load_spikes("6E.csv"),
              load_spikes("5I.csv"),
              load_spikes("5E.csv"),
              load_spikes("4I.csv"),
              load_spikes("4E.csv"),
              load_spikes("23I.csv"),
              load_spikes("23E.csv")]

# Find the maximum spike time and convert to seconds
duration_s = max(np.amax(t) for t, _, _, _ in pop_spikes) / 1000.0

# Create plot
figure, axes = plt.subplots(1, 2)

start_id = 0
bar_y = 0.0
for t, i, name, num in pop_spikes:
    # Plot spikes
    actor = axes[0].scatter(t, i + start_id, s=2, edgecolors="none")

    # Plot bar showing rate in matching colour
    axes[1].barh(bar_y, len(t) / (float(num) * duration_s), align="center", color=actor.get_facecolor(), ecolor="black")

    # Update offset
    start_id += num

    # Update bar pos
    bar_y += 1.0


axes[0].set_xlabel("Time [ms]")
axes[0].set_ylabel("Neuron number")

axes[1].set_xlabel("Mean firing rate [Hz]")
axes[1].set_yticks(np.arange(0.0, len(pop_spikes) * 1.0, 1.0))
axes[1].set_yticklabels(list(zip(*pop_spikes))[2])

# Show plot
plt.show()

