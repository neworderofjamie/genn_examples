import csv
import matplotlib.pyplot as plt
import numpy as np

with open("pre_trace.csv", "rb") as pre_file, open("pre_spikes.csv", "rb") as pre_spikes_file, open("post_trace.csv", "rb") as post_file, open("post_spikes.csv", "rb") as post_spikes_file:
    pre_reader = csv.reader(pre_file, delimiter = ",")
    post_reader = csv.reader(post_file, delimiter = ",")
    pre_spikes_reader = csv.reader(pre_spikes_file, delimiter = ",")
    post_spikes_reader = csv.reader(post_spikes_file, delimiter = ",")
    
    # Skip headers
    pre_spikes_reader.next()
    post_spikes_reader.next()
    
    # Read data and zip into columns
    pre_columns = zip(*pre_reader)
    post_columns = zip(*post_reader)
    pre_spikes_columns = zip(*pre_spikes_reader)
    post_spikes_columns = zip(*post_spikes_reader)
    
    # Convert CSV columns to numpy
    pre_times = np.asarray(pre_columns[0], dtype=float)
    pre_w = np.asarray(pre_columns[3], dtype=float)

    post_times = np.asarray(post_columns[0], dtype=float)
    post_w = np.asarray(post_columns[3], dtype=float)
    
    pre_spike_times = np.asarray(pre_spikes_columns[0], dtype=float)
    post_spike_times = np.asarray(post_spikes_columns[0], dtype=float)

    # Combine weights and times from pre and postsynaptic traces
    times = np.hstack((pre_times, post_times))
    w = np.hstack((pre_w, post_w))
    order = np.argsort(times)
    times = times[order]
    w = w[order]

    # Create plot
    figure, axis = plt.subplots()

    #axis.plot(pre_times, pre_z, label="Zi*")
    #axis.plot(post_times, post_z, label="Zj*")
    axis.plot(times, w)

    #axes[0].set_ylabel("Neuron number")
    #axes[1].set_ylabel("Mean firing rate [Hz]")

    axis.vlines(pre_spike_times, -5.0, -4.0, color="red", label="Pre spikes")
    axis.vlines(post_spike_times, -5.0, -4.0, color="blue", label="Post spikes")
    
    axis.set_xlabel("Time [ms]")
    axis.legend()

    # Show plot
    plt.show()

