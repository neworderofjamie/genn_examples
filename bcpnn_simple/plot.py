import matplotlib.pyplot as plt
import numpy as np

# Read data and zip into columns
pre_data = np.loadtxt("pre_trace.csv", delimiter=",")
post_data = np.loadtxt("post_trace.csv", delimiter=",")
pre_spikes_data = np.loadtxt("pre_spikes.csv")
post_spikes_data = np.loadtxt("post_spikes.csv")

# Convert CSV columns to numpy
pre_times = pre_data[:,0]
pre_w = pre_data[:,3]

post_times = post_data[:,0]
post_w = post_data[:,3]

pre_spike_times = pre_spikes_data[:,0]
post_spike_times = post_spikes_data[:,0]

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

