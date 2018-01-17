import csv
import matplotlib.pyplot as plt
import numpy as np

with open("spikes_e.csv", "rb") as spikes_e_csv_file, open("spikes_i.csv", "rb") as spikes_i_csv_file:
    spikes_e_csv_reader = csv.reader(spikes_e_csv_file, delimiter = ",")
    spikes_i_csv_reader = csv.reader(spikes_i_csv_file, delimiter = ",")

    # Skip headers
    spikes_e_csv_reader.next()
    spikes_i_csv_reader.next()

    # Read data and zip into columns
    spikes_e_data_columns = zip(*spikes_e_csv_reader)
    spikes_i_data_columns = zip(*spikes_i_csv_reader)

    # Convert CSV columns to numpy
    spike_e_times = np.asarray(spikes_e_data_columns[0], dtype=float)
    spike_i_times = np.asarray(spikes_i_data_columns[0], dtype=float)
    spike_e_neuron_id = np.asarray(spikes_e_data_columns[1], dtype=int)
    spike_i_neuron_id = np.asarray(spikes_i_data_columns[1], dtype=int)

    # Create plot
    figure, axes = plt.subplots(2, sharex=True)

    # Plot spikes
    axes[0].scatter(spike_e_times, spike_e_neuron_id, s=2, edgecolors="none")
    axes[0].scatter(spike_i_times, spike_i_neuron_id + 3200, s=2, edgecolors="none")

    # Plot rates
    bins = np.arange(0, 10000 + 1, 10)
    rate_e = np.histogram(spike_e_times, bins=bins)[0] *  (1000.0 / 10.0) * (1.0 / 3200.0)
    rate_i = np.histogram(spike_i_times, bins=bins)[0] *  (1000.0 / 10.0) * (1.0 / 800.0)
    axes[1].plot(bins[0:-1], rate_e)
    axes[1].plot(bins[0:-1], rate_i)

    axes[0].set_title("Spikes")
    axes[1].set_title("Firing rates")

    axes[0].set_xlim((0, 10000))
    axes[0].set_ylim((0, 4000))

    axes[0].set_ylabel("Neuron number")
    axes[1].set_ylabel("Mean firing rate [Hz]")

    axes[1].set_xlabel("Time [ms]")

    # Show plot
    plt.show()

