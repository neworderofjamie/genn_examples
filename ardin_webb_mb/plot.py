import csv
import matplotlib.pyplot as plt
import numpy as np

num_pn = 360
num_kc = 20000
num_en = 1

def get_csv_columns(csv_file, headers=True):
    # Create reader
    reader = csv.reader(csv_file, delimiter=",")

    # Skip headers if required
    if headers:
        reader.next()

    # Read columns and return
    return zip(*reader)

with open("pn_spikes.csv", "rb") as pn_spikes_file, \
     open("kc_spikes.csv", "rb") as kc_spikes_file, \
     open("en_spikes.csv", "rb") as en_spikes_file:

    # Read data and zip into columns
    pn_spikes_columns = get_csv_columns(pn_spikes_file)
    kc_spikes_columns = get_csv_columns(kc_spikes_file)
    en_spikes_columns = get_csv_columns(en_spikes_file)

    # Convert CSV columns to numpy
    pn_spike_times = np.asarray(pn_spikes_columns[0], dtype=float)
    pn_spike_neuron_id = np.asarray(pn_spikes_columns[1], dtype=int)
    kc_spike_times = np.asarray(kc_spikes_columns[0], dtype=float)
    kc_spike_neuron_id = np.asarray(kc_spikes_columns[1], dtype=int)
    en_spike_times = np.asarray(en_spikes_columns[0], dtype=float)
    en_spike_neuron_id = np.asarray(en_spikes_columns[1], dtype=int)

    # Create plot
    figure, axes = plt.subplots(3, sharex=True)

    # Plot spikes that occur in first second
    axes[0].scatter(pn_spike_times, pn_spike_neuron_id, s=2, edgecolors="none", color="red")
    axes[1].scatter(kc_spike_times, kc_spike_neuron_id, s=2, edgecolors="none", color="red")
    axes[2].scatter(en_spike_times, en_spike_neuron_id, s=2, edgecolors="none", color="red")

    # Configure axes
    axes[0].set_title("PN")
    axes[1].set_title("KC")
    axes[2].set_title("EN")
    axes[0].set_ylim((0, num_pn))
    axes[1].set_ylim((0, num_kc))
    axes[2].set_ylim((-1, 1))
    axes[0].set_ylabel("Neuron number")
    axes[1].set_ylabel("Neuron number")
    axes[2].set_xlabel("Time [ms]")

    # Show plot
    plt.show()

