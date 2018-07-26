import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import re

from scipy.stats import gaussian_kde

N_full = {
  '23': {'E': 20683, 'I': 5834},
  '4' : {'E': 21915, 'I': 5479},
  '5' : {'E': 4850, 'I': 1065},
  '6' : {'E': 14395, 'I': 2948}
}

N_scaling = 1.0
duration = 9.0

def load_spikes(filename):
    # Parse filename and use to get population name and size
    match = re.match("([0-9]+)([EI])\.csv", filename)
    name = match.group(1) + match.group(2)
    num = int(N_full[match.group(1)][match.group(2)] * N_scaling)

    with open(filename, "rb") as spikes_csv_file:
        spikes_csv_reader = csv.reader(spikes_csv_file, delimiter = ",")

        # Skip headers
        spikes_csv_reader.next()

        # Read data and zip into columns
        spikes_data_columns = zip(*spikes_csv_reader)

        # Convert CSV columns to numpy
        spike_times = np.asarray(spikes_data_columns[0], dtype=float)
        spike_neuron_id = np.asarray(spikes_data_columns[1], dtype=int)

        post_transient = (spike_times > 1000.0)
        spike_times = spike_times[post_transient]
        spike_neuron_id = spike_neuron_id[post_transient]

        return spike_times, spike_neuron_id, name, num

def calc_histogram(data, smoothing):
    # Calculate rate histogram using Freedman Diaconis Estimator
    hist, bin_edges = np.histogram(data, bins="fd", density=True)

    # Smooth histogram
    #hist_kde = gaussian_kde(hist, smoothing)
    #hist_smooth = hist_kde.evaluate(bin_edges[:-1])
    hist_smooth = hist
    
    # Discard empty bins
    populated_bins = (hist_smooth > 1E-15)
    hist_smooth = hist_smooth[populated_bins]
    bin_x = bin_edges[:-1][populated_bins]
    
    return bin_x, hist_smooth

def calc_rate_hist(spike_times, spike_ids, num, duration):
     # Calculate histogram of spike IDs to get each neuron's firing rate
    rate, _ = np.histogram(spike_ids, bins=range(num + 1))
    rate = np.divide(rate, duration, dtype=float)
    
    return calc_histogram(rate, 0.3)
    
       
def calc_cv_isi_hist(spike_times, spike_ids, num, duration):
    # Loop through neurons
    cv_isi = np.empty(num)
    num_spiking = 0
    for n in range(num):
        # Get mask of spikes from this neuron and use to extract their times
        mask = (spike_ids == n)
        neuron_spike_times = spike_times[mask]
        
        # Calculate ISI, mean and variance
        neuron_isi = np.diff(neuron_spike_times)
        neuron_mean = np.mean(neuron_isi)
        neuron_std = np.std(neuron_isi)
        
        # Calculate CV ISI and add to vector if it's not going to be NaN
        if neuron_std > 0.0:
            cv_isi[num_spiking] = neuron_mean / neuron_std
            num_spiking += 1
    
    return calc_histogram(cv_isi[:num_spiking], 0.04)
    
pop_spikes = [load_spikes("6I.csv"),
              load_spikes("6E.csv"),
              load_spikes("5I.csv"),
              load_spikes("5E.csv"),
              load_spikes("4I.csv"),
              load_spikes("4E.csv"),
              load_spikes("23I.csv"),
              load_spikes("23E.csv")]

# Create plot
main_fig, main_axes = plt.subplots(1, 2)

pop_rate_fig, pop_rate_axes = plt.subplots(4, 2, sharey="row", sharex="col")
pop_cv_isi_fig, pop_cv_isi_axes = plt.subplots(4, 2, sharey="row", sharex="col")

start_id = 0
bar_y = 0.0
for i, (spike_times, spike_ids, name, num) in enumerate(pop_spikes):
    # Plot spikes
    actor = main_axes[0].scatter(spike_times, spike_ids + start_id, s=1, edgecolors="none")

    # Plot bar showing rate in matching colour
    main_axes[1].barh(bar_y, len(spike_times) / float(num), align="center", color=actor.get_facecolor(), ecolor="black")

    # Calculate statistics
    rate_bin_x, rate_hist = calc_rate_hist(spike_times, spike_ids, num, duration)
    isi_bin_x, isi_hist = calc_cv_isi_hist(spike_times, spike_ids, num, duration)

    # Plot rate histogram
    pop_rate_axis = pop_rate_axes[3 - (i / 2), 1 - (i % 2)]
    pop_rate_axis.set_title(name)
    pop_rate_axis.plot(rate_bin_x, rate_hist)
    
    # Plot rate histogram
    pop_cv_isi_axis = pop_cv_isi_axes[3 - (i / 2), 1 - (i % 2)]
    pop_cv_isi_axis.set_title(name)
    pop_cv_isi_axis.plot(isi_bin_x, isi_hist)
    
    # Update offset
    start_id += num

    # Update bar pos
    bar_y += 1.0

for i in range(2):
    pop_rate_axes[-1, i].set_xlim((0.0, 20.0))
    pop_cv_isi_axes[-1, i].set_xlim((0.0, 1.5))

main_axes[0].set_xlabel("Time [ms]")
main_axes[0].set_ylabel("Neuron number")

main_axes[1].set_xlabel("Mean firing rate [Hz]")
main_axes[1].set_yticks(np.arange(0.0, len(pop_spikes) * 1.0, 1.0))
main_axes[1].set_yticklabels(zip(*pop_spikes)[2])

# Show plot
plt.show()

