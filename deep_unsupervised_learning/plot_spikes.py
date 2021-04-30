import matplotlib.pyplot as plt
import numpy as np
import sys

from glob import glob
from mpl_toolkits.mplot3d import Axes3D


hyper_params = {"input": (44, 16, 1),
                "conv1": (20, 6, 8),
                "conv2": (8, 1, 16),
                "output": 1000}

# Get layer name
layer = sys.argv[1] if len(sys.argv) > 1 else "input"
suffix = sys.argv[2] if len(sys.argv) > 2 else "spikes"
timestep = int(sys.argv[3]) if len(sys.argv) > 3 else 0


# Load spikes
spikes = np.loadtxt("%s_%s_%d.csv" % (layer, suffix, timestep), skiprows=1, delimiter=",")


print("%u spikes" % spikes.shape[0])


try:
    iter(hyper_params[layer])

    # Pull corresponding hyperparameters from dictionary
    width, height, num_channels = hyper_params[layer]

    # Convert neuron IDs to row, col and channel
    spike_row = (spikes[:,1] // num_channels) // width
    spike_col = (spikes[:,1] // num_channels) % width
    spike_chan = spikes[:,1] % num_channels

    hist, _ = np.histogramdd((spike_row, spike_col, spikes[:,0]), bins=(height, width, 1000))
    print("Mean:%f, Std:%f, Max:%f" % (np.average(hist), np.std(hist), np.amax(hist)))
    print(np.bincount(spike_chan.astype(int)))

    # Create 3D plot
    fig = plt.figure()
    axis = fig.add_subplot(111, projection="3d")

    axis.scatter(spike_col, spikes[:,0], spike_row, c=spike_chan, s=1)
    axis.set_xlim((0, width))
    axis.set_ylim((0, 100))
    axis.set_zlim((0, height))
    axis.set_xlabel("X")
    axis.set_ylabel("time")
    axis.set_zlabel("Y")
except TypeError:
    num_neurons = hyper_params[layer]
    hist, _ = np.histogram(spikes[:,0], bins=1000)

    classification = np.argmax(np.bincount(spikes[:,1].astype(int), minlength=num_neurons))

    print("Mean:%f, Std:%f, Max:%f" % (np.average(hist), np.std(hist), np.amax(hist)))
    print("Most active neuron:%u" % classification)

    # Create 3D plot
    fig, axis = plt.subplots()

    axis.scatter(spikes[:,0], spikes[:,1], s=1)
    axis.set_xlim((0, 100))
    axis.set_xlabel("time")
    axis.set_ylabel("spike")

    axis.annotate("", xy=(100.0, classification), xycoords="data", color="red",
                 xytext=(5.0, 0.0), textcoords="offset points", annotation_clip=True,
                 arrowprops=dict(facecolor="red", edgecolor="red", headlength=4.0))

plt.show()
