import csv
import matplotlib.pyplot as plt
import numpy as np

with open("spikes.csv", "rb") as spikes_csv_file:
    spikes_csv_reader = csv.reader(spikes_csv_file, delimiter = ",")

    # Skip headers
    spikes_csv_reader.next()

    # Read data and zip into columns
    spikes_data_columns = zip(*spikes_csv_reader)

    # Convert CSV columns to numpy
    spike_times = np.asarray(spikes_data_columns[0], dtype=float)
    spike_neuron_id = np.asarray(spikes_data_columns[1], dtype=int)

    max_spike_time = np.amax(spike_times)
    max_neuron_id = np.amax(spike_neuron_id)

    # Create plot
    figure, axis = plt.subplots()

    # Plot spikes
    axis.scatter(spike_times, spike_neuron_id, s=2, edgecolors="none")

    axis.set_title("Spikes")

    axis.set_xlim((0, max_spike_time))
    axis.set_ylim((0, max_neuron_id))

    axis.set_ylabel("Neuron number")
    axis.set_xlabel("Time [ms]")

    # Show plot
    plt.show()

