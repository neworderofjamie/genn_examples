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

N_scaling = 0.1

def load_spikes(filename):
    # Parse filename and use to get population name and size
    match = re.match("([0-9]+)([EI])\.csv", filename)
    name = match.group(1) + match.group(2)
    num = int(N_full[match.group(1)][match.group(2)] * N_scaling)

    print name, num
    with open(filename, "rb") as spikes_csv_file:
        spikes_csv_reader = csv.reader(spikes_csv_file, delimiter = ",")

        # Skip headers
        spikes_csv_reader.next()

        # Read data and zip into columns
        spikes_data_columns = zip(*spikes_csv_reader)

        # Convert CSV columns to numpy
        spike_times = np.asarray(spikes_data_columns[0], dtype=float)
        spike_neuron_id = np.asarray(spikes_data_columns[1], dtype=int)

        return spike_times, spike_neuron_id, name, num

pop_spikes = [load_spikes("23E.csv"),
              load_spikes("23I.csv"),
              load_spikes("4E.csv"),
              load_spikes("4I.csv"),
              load_spikes("5E.csv"),
              load_spikes("5I.csv"),
              load_spikes("6E.csv"),
              load_spikes("6I.csv")]

# Create plot
figure, axes = plt.subplots(1, 2)

start_id = 0
bar_y = 0.0
for t, i, name, num in pop_spikes:
    # Plot spikes
    actor = axes[0].scatter(t, i + start_id, s=2, edgecolors="none")

    # Plot bar showing rate in matching colour
    axes[1].barh(bar_y, len(t), align="center", color=actor.get_facecolor(), ecolor="black")

    # Update offset
    start_id += num

    # Update bar pos
    bar_y += 1.0


axes[0].set_xlabel("Time [ms]")
axes[0].set_ylabel("Neuron number")

axes[1].set_xlabel("Mean firing rate [Hz]")
axes[1].set_yticks(np.arange(0.0, len(pop_spikes) * 1.0, 1.0))
axes[1].set_yticklabels(zip(*pop_spikes)[2])

# Show plot
plt.show()

