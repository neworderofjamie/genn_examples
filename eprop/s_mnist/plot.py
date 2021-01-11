import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pandas import read_csv
from struct import unpack
from scipy.special import softmax

epoch = 0
batch = 0
trials = [0,100, 200, 400]

DT = 1.0
TRIAL_TIMESTEPS = (28 * 28 * 2) + 20

def gen_spikes(_x, n_input=100, cue_duration=20):
    # Calculate thresholds for each pair of neurons
    thresholds = (np.linspace(0., 254., (n_input - 1) // 2)).astype(np.uint8)

    # Flatten image data
    _x = np.reshape(_x, (-1,))
    
    # Get mask of pixels below each threshold (cols)
    lower = _x[:, None] < thresholds[None, :]
    
    # Get mask of pixels above each threshold (cols)
    higher = _x[:, None] >= thresholds[None, :]
    
    # Get mask of pixels where each threshold is positively crossed
    transition_onset = np.logical_and(lower[:-1], higher[1:])
    
    # Get mask of pixels where each threshold is negatively crossed
    transition_offset = np.logical_and(higher[:-1], lower[1:])
    
    # Convert to float and add padding rows
    onset_spikes = transition_onset.astype(np.float32)
    onset_spikes = np.concatenate((onset_spikes, np.zeros_like(onset_spikes[:1])), 0)
    offset_spikes = transition_offset.astype(np.float32)
    offset_spikes = np.concatenate((offset_spikes, np.zeros_like(offset_spikes[:1])), 0)

    touch_spikes = np.equal(_x, 255).astype(np.float32)[..., None]
    out_spikes = np.concatenate((onset_spikes, offset_spikes, touch_spikes), -1)
    
    # Add axis and use to repeat each neuron's spikes twice
    out_spikes = np.tile(out_spikes[:, None], (1, 2, 1))
    
    # Reshape into per-neuron spikes
    out_spikes = np.reshape(out_spikes, (-1, n_input - 1))
    
    # Add 20 timesteps of silence
    out_spikes = np.concatenate((out_spikes, np.zeros_like(out_spikes[:cue_duration])), 0)
    
    signal_spikes = np.concatenate(
        (np.zeros_like(out_spikes[:-cue_duration, :1]), np.ones_like(touch_spikes[:cue_duration])), 0)
    out_spikes = np.concatenate((out_spikes, signal_spikes), -1)

    return out_spikes

def load_image_data(filename):
    with open(filename, mode="rb") as file:
        image_data = file.read()

        # Unpack header from first 16 bytes of buffer
        magic, num_items, num_rows, num_cols = unpack('>IIII', image_data[:16])
        assert magic == 2051
        assert num_rows == 28
        assert num_cols == 28

        # Convert remainder of buffer to numpy bytes
        image_data_np = np.frombuffer(image_data[16:], dtype=np.uint8)

        # Reshape data into individual images
        image_data_np = np.reshape(image_data_np, (num_items, num_rows * num_cols))
        
        return image_data_np

def load_label_data(filename):
    with open(filename, mode="rb") as file:
        label_data = file.read()

        # Unpack header from first 8 bytes of buffer
        magic, num_items = unpack('>II', label_data[:8])
        assert magic == 2049

        # Convert remainder of buffer to numpy bytes
        label_data_np = np.frombuffer(label_data[8:], dtype=np.uint8)
        assert label_data_np.shape == (num_items,)

        return label_data_np

# Load data
input_spikes = read_csv("input_spikes_%u_%u.csv" % (epoch, batch), header=None, names=["time", "neuron_id"], skiprows=1, delimiter=",",
                        dtype={"time":float, "neuron_id":int})
recurrent_alif_spikes = read_csv("recurrent_alif_spikes_%u_%u.csv" % (epoch, batch), header=None, names=["time", "neuron_id"], skiprows=1, delimiter=",",
                                 dtype={"time":float, "neuron_id":int})
output_data = np.loadtxt("output_%u_%u.csv" % (epoch, batch), delimiter=",", usecols=range(49))

# Load MNIST labels and images
mnist_images = load_image_data("mnist/train-images.idx3-ubyte")
mnist_labels = load_label_data("mnist/train-labels.idx1-ubyte")

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

    # Generate 'correct' spikes using Franz's code
    spikes = gen_spikes(mnist_images[epoch_index])
    times, ids = np.where(spikes > 0.5)
    
    # Plot input spikes
    input_axis = figure.add_subplot(gs[0:2, i])
    input_axis.scatter(trial_input_spikes["time"], trial_input_spikes["neuron_id"], s=1, edgecolors="none")
    input_axis.scatter(times + trial_start_time, ids, s=1, edgecolors="none")
    
    # Plot ALIF spikes
    alif_axis = figure.add_subplot(gs[2:4, i], sharex=input_axis)
    alif_axis.scatter(trial_recurrent_alif_spikes["time"], trial_recurrent_alif_spikes["neuron_id"], s=2, edgecolors="none")

    # Calculate softmax for each timestep
    host_softmax = softmax(trial_output[:,1:17], axis=1)

    # Loop through outputs
    for o in range(10):
        output_axis = figure.add_subplot(gs[4 + o, i], sharex=input_axis)
        output_axis.plot(trial_output[:,0], trial_output[:,1 + o], label="Y")
        output_axis.plot(trial_output[:,0], trial_output[:,17 + o], label="Pi")
        output_axis.plot(trial_output[:,0], trial_output[:,33 + o], label="E")
        output_axis.plot(trial_output[:,0], host_softmax[:,o], label="softmax")
        
        output_axis.set_ylim((-1,1))
        
        if mnist_labels[epoch_index] == o:
            rect = patches.Rectangle((trial_start_time + (28 * 28 * 2), -1), 20, 2.0,
                                     linewidth=1, facecolor="red", alpha=0.2)
            output_axis.add_patch(rect)
        
        if o == 0:
            output_axis.legend(ncol=4)


# Show plot
plt.show()
