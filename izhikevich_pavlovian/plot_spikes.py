import csv
import matplotlib.pyplot as plt
import numpy as np

num_excitatory = 800
num_inhibitory = 200
num_neurons = num_excitatory + num_inhibitory
duration_ms = 60 * 60 * 1000
bin_ms = 10
display_time = 2000

def get_masks(times):
    return (np.where(times < 40000),
            np.where(times > (duration_ms - 40000)))

def get_csv_columns(csv_file, headers=True):
    # Create reader
    reader = csv.reader(csv_file, delimiter=",")

    # Skip headers if required
    if headers:
        reader.next()

    # Read columns and return
    return zip(*reader)

def plot_reward(axis, times):
    for t in times:
        axis.annotate("reward",
            xy=(t, 0), xycoords="data",
            xytext=(0, -15.0), textcoords="offset points",
            arrowprops=dict(facecolor="black", headlength=6.0),
            annotation_clip=True, ha="center", va="top")

def plot_stimuli(axis, times, ids):
    for t, i in zip(times, ids):
        colour = "green" if i == 0 else "black"
        axis.annotate("S%u" % i,
            xy=(t, num_neurons), xycoords="data",
            xytext=(0, 15.0), textcoords="offset points",
            arrowprops=dict(facecolor=colour, edgecolor=colour, headlength=6.0),
            annotation_clip=True, ha="center", va="bottom", color=colour)

with open("e_spikes.csv", "rb") as e_spikes_file, \
     open("i_spikes.csv", "rb") as i_spikes_file, \
     open("stimulus_times.csv", "rb") as stimuli_file, \
     open("reward_times.csv", "rb") as reward_times_file:

    # Read data and zip into columns
    e_spikes_columns = get_csv_columns(e_spikes_file)
    i_spikes_columns = get_csv_columns(i_spikes_file)
    stimuli_columns = get_csv_columns(stimuli_file, False)
    reward_times_columns = get_csv_columns(reward_times_file, False)

    # Convert CSV columns to numpy
    e_spike_times = np.asarray(e_spikes_columns[0], dtype=float)
    e_spike_neuron_id = np.asarray(e_spikes_columns[1], dtype=int)
    i_spike_times = np.asarray(i_spikes_columns[0], dtype=float)
    i_spike_neuron_id = np.asarray(i_spikes_columns[1], dtype=int)
    stimuli_times = np.asarray(stimuli_columns[0], dtype=float)
    stimuli_id = np.asarray(stimuli_columns[1], dtype=int)
    reward_times = np.asarray(reward_times_columns[0], dtype=float)

    # Get masks for events in first and last seconds
    e_spike_first_second, e_spike_last_second = get_masks(e_spike_times)
    i_spike_first_second, i_spike_last_second = get_masks(i_spike_times)
    stimuli_first_second, stimuli_last_second = get_masks(stimuli_times)
    reward_times_first_second, reward_times_last_second = get_masks(reward_times)

    # Find the earliest rewarded stimuli in first and last seconds
    rewarded_stimuli_time_first_second = stimuli_times[stimuli_first_second][np.where(stimuli_id[stimuli_first_second] == 0)[0][0]]
    rewarded_stimuli_time_last_second = stimuli_times[stimuli_last_second][np.where(stimuli_id[stimuli_last_second] == 0)[0][0]]

    # Find the corresponding stimuli
    corresponding_reward_first_second = reward_times[reward_times_first_second][np.where(reward_times[reward_times_first_second] > rewarded_stimuli_time_first_second)[0][0]]
    corresponding_reward_last_second = reward_times[reward_times_last_second][np.where(reward_times[reward_times_last_second] > rewarded_stimuli_time_last_second)[0][0]]

    padding_first_second = (display_time - (corresponding_reward_first_second - rewarded_stimuli_time_first_second)) / 2
    padding_last_second = (display_time - (corresponding_reward_last_second - rewarded_stimuli_time_last_second)) / 2

    # Create plot
    figure, axes = plt.subplots(2)

    # Plot spikes that occur in first second
    axes[0].scatter(e_spike_times[e_spike_first_second], e_spike_neuron_id[e_spike_first_second], s=2, edgecolors="none", color="red")
    axes[0].scatter(i_spike_times[i_spike_first_second], i_spike_neuron_id[i_spike_first_second] + num_excitatory, s=2, edgecolors="none", color="blue")

    # Plot reward times and rewarded stimuli that occur in first second
    plot_reward(axes[0], reward_times[reward_times_first_second]);
    plot_stimuli(axes[0], stimuli_times[stimuli_first_second], stimuli_id[stimuli_first_second])

    # Plot spikes that occur in final second
    axes[1].scatter(e_spike_times[e_spike_last_second], e_spike_neuron_id[e_spike_last_second], s=2, edgecolors="none", color="red")
    axes[1].scatter(i_spike_times[i_spike_last_second], i_spike_neuron_id[i_spike_last_second] + num_excitatory, s=2, edgecolors="none", color="blue")

    # Plot reward times and rewarded stimuli that occur in final second
    plot_reward(axes[1], reward_times[reward_times_last_second]);
    plot_stimuli(axes[1], stimuli_times[stimuli_last_second], stimuli_id[stimuli_last_second])

    # Configure axes
    axes[0].set_title("Before")
    axes[1].set_title("After")
    axes[0].set_xlim((rewarded_stimuli_time_first_second - padding_first_second, corresponding_reward_first_second + padding_first_second))
    axes[1].set_xlim((rewarded_stimuli_time_last_second - padding_last_second, corresponding_reward_last_second + padding_last_second))
    axes[0].set_ylim((0, num_neurons))
    axes[1].set_ylim((0, num_neurons))
    axes[0].set_ylabel("Neuron number")
    axes[0].set_ylabel("Neuron number")
    axes[0].set_xlabel("Time [ms]")
    axes[1].set_xlabel("Time [ms]")

    # Show plot
    plt.show()

