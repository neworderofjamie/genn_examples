import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pandas import read_csv

epoch = 0
batch = 0
trials = [511]

DT = 1.0
TRIAL_TIMESTEPS = (28 * 28 * 2) + 20

# Load data
input_spikes = read_csv("input_spikes_%u_%u.csv" % (epoch, batch), header=None, names=["time", "neuron_id"], skiprows=1, delimiter=",",
                        dtype={"time":float, "neuron_id":int})
recurrent_alif_spikes = read_csv("recurrent_alif_spikes_%u_%u.csv" % (epoch, batch), header=None, names=["time", "neuron_id"], skiprows=1, delimiter=",",
                                 dtype={"time":float, "neuron_id":int})
output_data = np.loadtxt("output_%u_%u.csv" % (epoch, batch), delimiter=",", usecols=range(33))

# Create plot
figure = plt.figure()

# Create gridspec so we can jam in lots of tiny outputs
gs = figure.add_gridspec(14, len(trials))

# Loop through trials we want to plot
for i, trial in enumerate(trials):
    # Convert trial and batch into index into epoch
    epoch_index = (batch * 512) + trial

    # Calculate start and end time of this trial
    trial_start_time = trial * TRIAL_TIMESTEPS * DT
    trial_end_time = trial_start_time + (TRIAL_TIMESTEPS * DT)

    # Extract data from this trial
    trial_input_spikes = input_spikes[(input_spikes["time"] >= trial_start_time) & (input_spikes["time"] < trial_end_time)]
    trial_recurrent_alif_spikes = recurrent_alif_spikes[(recurrent_alif_spikes["time"] >= trial_start_time) & (recurrent_alif_spikes["time"] < trial_end_time)]
    trial_output = output_data[(output_data[:,0] >= trial_start_time) & (output_data[:,0] < trial_end_time)]

    # Plot input spikes
    input_axis = figure.add_subplot(gs[0:2, i])
    input_axis.scatter(trial_input_spikes["time"], trial_input_spikes["neuron_id"], s=1, edgecolors="none")

    # Plot ALIF spikes
    alif_axis = figure.add_subplot(gs[2:4, i], sharex=input_axis)
    alif_axis.scatter(trial_recurrent_alif_spikes["time"], trial_recurrent_alif_spikes["neuron_id"], s=2, edgecolors="none")
    
    if i == 0:
        input_axis.set_ylabel("Input\nneuron")
        alif_axis.set_ylabel("Recurrent\nALIF neuron")
    
    # Loop through outputs
    for o in range(10):
        output_axis = figure.add_subplot(gs[4 + o, i], sharex=input_axis)
        output_axis.plot(trial_output[:,0], trial_output[:,1 + o], label="Pi")
        output_axis.plot(trial_output[:,0], trial_output[:,17 + o], label="E")
        output_axis.set_ylim((-1,1))
        output_axis.axhline(0.0, linestyle="--", color="gray", alpha=0.2)
        
        if i == 0:
            output_axis.set_ylabel("Out%u" % o)
        
        if o == 0:
            output_axis.legend(ncol=4)
        elif o == 9:
            output_axis.set_xlabel("Time [ms]")


# Show plot
plt.show()
