import csv
import matplotlib.pyplot as plt
import numpy as np

num_excitatory = 800
num_inhibitory = 200
#duration_ms = 1000 * 60 * 60
duration_ms = 160000
bin_ms = 10

with open("e_spikes.csv", "rb") as e_spikes_file, \
     open("i_spikes.csv", "rb") as i_spikes_file, \
     open("rewarded_stimulus_times.csv", "rb") as rewarded_stimulus_file, \
     open("reward_times.csv", "rb") as reward_times_file:
    # Create CSV readers
    e_spikes_reader = csv.reader(e_spikes_file, delimiter=",")
    i_spikes_reader = csv.reader(i_spikes_file, delimiter=",")
    rewarded_stimus_reader = csv.reader(rewarded_stimulus_file, delimiter=",")
    reward_times_reader = csv.reader(reward_times_file, delimiter=",")

    # Skip headers
    e_spikes_reader.next()
    i_spikes_reader.next()

    # Read data and zip into columns
    e_spikes_columns = zip(*e_spikes_reader)
    i_spikes_columns = zip(*i_spikes_reader)
    rewarded_stimulus_columns = zip(*rewarded_stimus_reader)
    reward_times_columns = zip(*reward_times_reader)

    # Convert CSV columns to numpy
    e_spike_times = np.asarray(e_spikes_columns[0], dtype=float)
    e_spike_neuron_id = np.asarray(e_spikes_columns[1], dtype=int)
    i_spike_times = np.asarray(i_spikes_columns[0], dtype=float)
    i_spike_neuron_id = np.asarray(i_spikes_columns[1], dtype=int)

    rewarded_stimulus_times = np.asarray(rewarded_stimulus_columns[0], dtype=float)
    reward_times = np.asarray(reward_times_columns[0], dtype=float)

    # Create plot
    figure, axes = plt.subplots(2, sharex=True)

    # Plot spikes
    axes[0].scatter(e_spike_times, e_spike_neuron_id, s=2, edgecolors="none", color="red")
    axes[0].scatter(i_spike_times, i_spike_neuron_id + num_excitatory, s=2, edgecolors="none", color="blue")

    # Plot reward times on both axes
    axes[0].vlines(reward_times, 0, num_excitatory + num_inhibitory, color="green")
    axes[1].vlines(reward_times, 0, num_excitatory + num_inhibitory, color="green")
    axes[0].vlines(rewarded_stimulus_times, 0, num_excitatory + num_inhibitory, color="grey", linestyle="--")
    axes[1].vlines(rewarded_stimulus_times, 0, num_excitatory + num_inhibitory, color="grey", linestyle="--")

    # Plot rates
    bins = np.arange(0, duration_ms + 1, bin_ms)
    e_rate = np.histogram(e_spike_times, bins=bins)[0] *  (1000.0 / float(bin_ms)) * (1.0 / float(num_excitatory))
    i_rate = np.histogram(i_spike_times, bins=bins)[0] *  (1000.0 / float(bin_ms)) * (1.0 / float(num_inhibitory))
    axes[1].plot(bins[0:-1], e_rate, color="red")
    axes[1].plot(bins[0:-1], i_rate, color="blue")

    axes[0].set_title("Spikes")
    axes[1].set_title("Firing rates")

    axes[0].set_xlim((0, duration_ms))
    axes[0].set_ylim((0, num_excitatory + num_inhibitory))

    axes[0].set_ylabel("Neuron number")
    axes[1].set_ylabel("Mean firing rate [Hz]")

    axes[1].set_xlabel("Time [ms]")

    # Show plot
    plt.show()

