import csv
import matplotlib.pyplot as plt
import numpy as np

stimuli_time = 40 + 200
num_stimuli = 14
duration_ms = stimuli_time * num_stimuli
num_pn = 36 * 10
num_kc = 20000
num_en = 1
plot_synapse = False

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
     open("en_spikes.csv", "rb") as en_spikes_file, \
     open("kc_en_syn.csv", "rb") as kc_en_syn_file:

    # Read data and zip into columns
    pn_spikes_columns = get_csv_columns(pn_spikes_file)
    kc_spikes_columns = get_csv_columns(kc_spikes_file)
    en_spikes_columns = get_csv_columns(en_spikes_file)

    if plot_synapse:
        kc_en_syn_columns = get_csv_columns(kc_en_syn_file, False)

    # Convert CSV columns to numpy
    pn_spike_times = np.asarray(pn_spikes_columns[0], dtype=float)
    pn_spike_neuron_id = np.asarray(pn_spikes_columns[1], dtype=int)
    kc_spike_times = np.asarray(kc_spikes_columns[0], dtype=float)
    kc_spike_neuron_id = np.asarray(kc_spikes_columns[1], dtype=int)
    en_spike_times = np.asarray(en_spikes_columns[0], dtype=float)
    en_spike_neuron_id = np.asarray(en_spikes_columns[1], dtype=int)

    if plot_synapse:
        kc_en_tag = np.asarray(kc_en_syn_columns[2], dtype=float)
        kc_en_weight = np.asarray(kc_en_syn_columns[3], dtype=float)

        # Reshape continuous tag and weight columns so they can be directly plotted
        kc_en_tag = kc_en_tag.reshape((-1, num_kc * num_en))
        kc_en_weight = kc_en_weight.reshape((-1, num_kc * num_en))

    # Calculate activation level for each stimuli
    for i in range(num_stimuli):
        print "Stimuli %u:" % i
        start_time = i * stimuli_time
        end_time = start_time + stimuli_time

        pn_spike_mask = (pn_spike_times >= start_time) & (pn_spike_times < end_time)
        kc_spike_mask = (kc_spike_times >= start_time) & (kc_spike_times < end_time)

        print "\tPN:%u" % len(np.unique(pn_spike_neuron_id[pn_spike_mask]))
        print "\tKC:%u" % len(np.unique(kc_spike_neuron_id[kc_spike_mask]))

    # Calculate activations per timestep
    bins = np.arange(duration_ms + 1)
    pn_activations = np.histogram(pn_spike_times, bins=bins)[0]
    kc_activations = np.histogram(kc_spike_times, bins=bins)[0]
    en_activations = np.histogram(en_spike_times, bins=bins)[0]

    # Create plot
    figure, axes = plt.subplots(4 if plot_synapse else 3, 2, sharex='col')

    # Plot spikes that occur in first second
    axes[0, 0].scatter(pn_spike_times, pn_spike_neuron_id, s=2, edgecolors="none", color="red")
    axes[1, 0].scatter(kc_spike_times, kc_spike_neuron_id, s=2, edgecolors="none", color="red")
    axes[2, 0].scatter(en_spike_times, en_spike_neuron_id, s=2, edgecolors="none", color="red")

    # Plot activations
    axes[0, 1].plot(bins[:-1], pn_activations)
    axes[1, 1].plot(bins[:-1], kc_activations)
    axes[2, 1].plot(bins[:-1], en_activations)

    if plot_synapse:
        axes[3, 0].plot(kc_en_tag)
        axes[3, 1].plot(kc_en_weight)

    # Configure axes
    axes[0, 0].set_title("PN")
    axes[1, 0].set_title("KC")
    axes[2, 0].set_title("EN")
    axes[0, 0].set_ylim((0, num_pn))
    axes[1, 0].set_ylim((0, num_kc))
    axes[2, 0].set_ylim((-1, 1))
    axes[0, 0].set_ylabel("Neuron number")
    axes[1, 0].set_ylabel("Neuron number")
    axes[2, 0].set_ylabel("Neuron number")
    axes[3 if plot_synapse else 2, 0].set_xlabel("Time [ms]")

    if plot_synapse:
        figure2, axes2 = plt.subplots(2)

        axes2[0].hist(kc_en_tag[-1,:], bins=10)
        axes2[1].hist(kc_en_weight[-1,:], bins=10)

        axes2[0].set_title("Tag")
        axes2[1].set_title("Weight")

    # Show plot
    plt.show()

