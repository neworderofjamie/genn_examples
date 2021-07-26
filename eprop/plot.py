import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
from pandas import read_csv
from argparse import ArgumentParser

from tonic_classifier_parser import parse_arguments

MAX_STIMULI_TIMES = {"smnist": 1568.0 * 2.0, "shd": 1369.140625 * 2.0}

# Build command line parse
parser = ArgumentParser(add_help=False)
parser.add_argument("--epoch", type=int, default=0)
parser.add_argument("--batch", type=int, default=0)
parser.add_argument("--trial", type=int, action="append", required=True)
output_directory = parse_arguments(parser, description="Plot eProp spike trains")[1]

# Create plot
figure, axes = plt.subplots(2, len(args.trial), sharex="col", sharey="row")

# Loop through trials we want to plot
for i, trial in enumerate(args.trial):
    # Load data
    input_spikes = read_csv(os.path.join(output_directory, "input_spikes_%u_%u_%u.csv" % (args.epoch, args.batch, i)), header=None, names=["time", "neuron_id"], skiprows=1, delimiter=",",
                            dtype={"time":float, "neuron_id":int})
    recurrent_alif_spikes = read_csv(os.path.join(output_directory, "recurrent_spikes_%u_%u_%u.csv" % (args.epoch, args.batch, i)), header=None, names=["time", "neuron_id"], skiprows=1, delimiter=",",
                                     dtype={"time":float, "neuron_id":int})

    # Plot input spikes
    axes[0, i].scatter(input_spikes["time"], input_spikes["neuron_id"], s=1, edgecolors="none")

    # Plot ALIF spikes
    axes[1, i].scatter(recurrent_alif_spikes["time"], recurrent_alif_spikes["neuron_id"], s=2, edgecolors="none")
    
    axes[0, i].set_xlim((0.0, MAX_STIMULI_TIMES[args.dataset]))

    if i == 0:
        axes[0, i].set_ylabel("Input\nneuron")
        axes[1, i].set_ylabel("Recurrent\nALIF neuron")

    axes[1, i].set_xlabel("Time [ms]")


# Show plot
plt.show()
