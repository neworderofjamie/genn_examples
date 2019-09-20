import matplotlib.pyplot as plt
import numpy as np
import sys

num_neurons = 1000 if len(sys.argv) <= 1 else int(sys.argv[1])
num_excitatory = int(0.8 * num_neurons)
num_inhibitory = int(0.2 * num_neurons)

duration_ms = 60 * 60 * 1000
bin_ms = 10
display_time = 2000

def get_masks(times):
    return (np.where(times < 50000),
            np.where(times > (duration_ms - 50000)))

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

def read_spikes(filename):
    return np.loadtxt(filename, delimiter=",", skiprows=1,
                      dtype={"names": ("time", "id"),
                             "formats": (np.float, np.int)})

# Read spikes
e_spikes = read_spikes("e_spikes.csv")
i_spikes = read_spikes("i_spikes.csv")

# Read stimuli
stimuli = np.loadtxt("stimulus_times.csv", delimiter=",",
                     dtype={"names": ("time", "id"),
                            "formats": (np.float, np.int)})

# Read rewards
reward_times = np.loadtxt("reward_times.csv", dtype=np.float)

# Get masks for events in first and last seconds
e_spike_first_second, e_spike_last_second = get_masks(e_spikes["time"])
i_spike_first_second, i_spike_last_second = get_masks(i_spikes["time"])
stimuli_first_second, stimuli_last_second = get_masks(stimuli["time"])
reward_times_first_second, reward_times_last_second = get_masks(reward_times)

# Find the earliest rewarded stimuli in first and last seconds
rewarded_stimuli_time_first_second = stimuli["time"][stimuli_first_second][np.where(stimuli["id"][stimuli_first_second] == 0)[0][0]]
rewarded_stimuli_time_last_second = stimuli["time"][stimuli_last_second][np.where(stimuli["id"][stimuli_last_second] == 0)[0][0]]

# Find the corresponding stimuli
corresponding_reward_first_second = reward_times[reward_times_first_second][np.where(reward_times[reward_times_first_second] > rewarded_stimuli_time_first_second)[0][0]]
corresponding_reward_last_second = reward_times[reward_times_last_second][np.where(reward_times[reward_times_last_second] > rewarded_stimuli_time_last_second)[0][0]]

padding_first_second = (display_time - (corresponding_reward_first_second - rewarded_stimuli_time_first_second)) / 2
padding_last_second = (display_time - (corresponding_reward_last_second - rewarded_stimuli_time_last_second)) / 2

# Create plot
figure, axes = plt.subplots(2)

# Plot spikes that occur in first second
axes[0].scatter(e_spikes["time"][e_spike_first_second], e_spikes["id"][e_spike_first_second], s=2, edgecolors="none", color="red")
axes[0].scatter(i_spikes["time"][i_spike_first_second], i_spikes["id"][i_spike_first_second] + num_excitatory, s=2, edgecolors="none", color="blue")

# Plot reward times and rewarded stimuli that occur in first second
plot_reward(axes[0], reward_times[reward_times_first_second]);
plot_stimuli(axes[0], stimuli["time"][stimuli_first_second], stimuli["id"][stimuli_first_second])

# Plot spikes that occur in final second
axes[1].scatter(e_spikes["time"][e_spike_last_second], e_spikes["id"][e_spike_last_second], s=2, edgecolors="none", color="red")
axes[1].scatter(i_spikes["time"][i_spike_last_second], i_spikes["id"][i_spike_last_second] + num_excitatory, s=2, edgecolors="none", color="blue")

# Plot reward times and rewarded stimuli that occur in final second
plot_reward(axes[1], reward_times[reward_times_last_second]);
plot_stimuli(axes[1], stimuli["time"][stimuli_last_second], stimuli["id"][stimuli_last_second])

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

