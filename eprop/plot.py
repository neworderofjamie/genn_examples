import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pandas import read_csv
from argparse import ArgumentParser

# Build command line parse
parser = ArgumentParser(description="Train eProp classifier")
parser.add_argument("--dt", type=float, default=1.0)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--num-recurrent-alif", type=int, default=256)
parser.add_argument("--dataset", choices=["smnist", "shd"], required=True)
parser.add_argument("--epoch", type=int, default=0)
parser.add_argument("--batch", type=int, default=0)
parser.add_argument("--trial", type=int, action="append", required=True)
args = parser.parse_args()

MAX_STIMULI_TIMES = {"smnist": 1568.0, "shd": 1369.140625}

# Build file suffix
name_suffix = "%u" % (args.num_recurrent_alif)

# Load data
input_spikes = read_csv("%s_input_spikes_%s_%u_%u.csv" % (args.dataset, name_suffix, args.epoch, args.batch), header=None, names=["time", "neuron_id"], skiprows=1, delimiter=",",
                        dtype={"time":float, "neuron_id":int})
recurrent_alif_spikes = read_csv("%s_recurrent_spikes_%s_%u_%u.csv" % (args.dataset, name_suffix, args.epoch, args.batch), header=None, names=["time", "neuron_id"], skiprows=1, delimiter=",",
                                 dtype={"time":float, "neuron_id":int})

# Create plot
figure, axes = plt.subplots(2, len(args.trial), sharex="col", sharey="row")

# Loop through trials we want to plot
for i, trial in enumerate(args.trial):
    # Convert trial and batch into index into epoch
    epoch_index = (args.batch * args.batch_size) + trial

    # Calculate start and end time of this trial
    trial_start_time = trial * MAX_STIMULI_TIMES[args.dataset]
    trial_end_time = trial_start_time + MAX_STIMULI_TIMES[args.dataset]

    # Extract data from this trial
    trial_input_spikes = input_spikes[(input_spikes["time"] >= trial_start_time) & (input_spikes["time"] < trial_end_time)]
    trial_recurrent_alif_spikes = recurrent_alif_spikes[(recurrent_alif_spikes["time"] >= trial_start_time) & (recurrent_alif_spikes["time"] < trial_end_time)]

    # Plot input spikes
    axes[0, i].scatter(trial_input_spikes["time"], trial_input_spikes["neuron_id"], s=1, edgecolors="none")

    # Plot ALIF spikes
    axes[1, i].scatter(trial_recurrent_alif_spikes["time"], trial_recurrent_alif_spikes["neuron_id"], s=2, edgecolors="none")
    
    axes[0, i].set_xlim((trial_start_time, trial_end_time))

    if i == 0:
        axes[0, i].set_ylabel("Input\nneuron")
        axes[1, i].set_ylabel("Recurrent\nALIF neuron")

    axes[1, i].set_xlabel("Time [ms]")
    



# Show plot
plt.show()
