import csv
import matplotlib.pyplot as plt
import numpy as np

with open("spikes.csv", "rb") as spikes_csv_file, open("weights.csv", "rb") as weights_csv_file:
    spikes_csv_reader = csv.reader(spikes_csv_file, delimiter = ",")
    weights_csv_reader = csv.reader(weights_csv_file, delimiter = ",")

    # Skip headers
    spikes_csv_reader.next()
    weights_csv_reader.next()

    # Read data and zip into columns
    spikes_data_columns = zip(*spikes_csv_reader)
    weights_data_columns = zip(*weights_csv_reader)

    # Convert CSV columns to numpy
    spike_times = np.asarray(spikes_data_columns[0], dtype=float)
    spike_neuron_id = np.asarray(spikes_data_columns[1], dtype=int)
    weight_times = np.asarray(weights_data_columns[0], dtype=float)
    weights = np.asarray(weights_data_columns[1], dtype=float)

    # Create plot
    figure, axes = plt.subplots(3, sharex=True)

    # Plot voltages
    axes[0].scatter(spike_times, spike_neuron_id)

    # Plot rates
    bins = np.arange(0, 10000 + 1, 10)
    rate = np.histogram(spike_times, bins=bins)[0] *  (1000.0 / 10.0) * (1.0 / 2000.0)
    axes[1].plot(bins[0:-1], rate)

    # Plot weight evolution
    axes[2].plot(weight_times, weights)

    axes[0].set_title("Spikes")
    axes[1].set_title("Firing rates")
    axes[2].set_title("Weight evolution")

    axes[0].set_xlim((0, 10000))
    axes[0].set_ylim((0, 2000))

    axes[0].set_ylabel("Neuron number")
    axes[1].set_ylabel("Mean firing rate [Hz]")
    axes[2].set_ylabel("Mean I->E weights [nA]")

    axes[2].set_xlabel("Time [ms]")

    # Show plot
    plt.show()

