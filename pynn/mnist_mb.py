import numpy as np
import matplotlib.pyplot as plt
from os import path
from struct import unpack

from gzip import decompress
from urllib import request
from pygenn import genn_model, genn_wrapper

def get_image_data(url, filename, correct_magic):
    if path.exists(filename):
        print("Loading existing data")
        return np.load(filename)
    else:
        print("Downloading dataset")
        with request.urlopen(url) as response:
            print("Decompressing dataset")
            image_data = decompress(response.read())

            # Unpack header from first 16 bytes of buffer
            magic, num_items, num_rows, num_cols = unpack('>IIII', image_data[:16])
            assert magic == correct_magic
            assert num_rows == 28
            assert num_cols == 28

            # Convert remainder of buffer to numpy bytes
            image_data_np = np.frombuffer(image_data[16:], dtype=np.uint8)

            # Reshape data into individual images
            image_data_np = np.reshape(image_data_np, (num_items, num_rows * num_cols))

            # Write to disk
            np.save(filename, image_data_np)

            return image_data_np

def get_label_data(url, filename, correct_magic):
    if path.exists(filename):
        print("Loading existing data")
        return np.load(filename)
    else:
        print("Downloading dataset")
        with request.urlopen(url) as response:
            print("Decompressing dataset")
            label_data = decompress(response.read())

            # Unpack header from first 8 bytes of buffer
            magic, num_items = unpack('>II', label_data[:8])
            assert magic == correct_magic

            # Convert remainder of buffer to numpy bytes
            label_data_np = np.frombuffer(label_data[8:], dtype=np.uint8)
            assert label_data_np.shape == (num_items,)

            # Write to disk
            np.save(filename, label_data_np)

            return label_data_np

def get_training_data():
    images = get_image_data("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "training_images.npy", 2051)
    labels = get_label_data("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "training_labels.npy", 2049)
    assert images.shape[0] == labels.shape[0]

    return images, labels

def get_testing_data():
    images = get_image_data("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "testing_images.npy", 2051)
    labels = get_label_data("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "testing_labels.npy", 2049)
    assert images.shape[0] == labels.shape[0]

    return images, labels

def record_current_spikes(pop, spikes, dt):
    current_spikes = pop.current_spikes
    current_spike_times = np.ones(current_spikes.shape) * dt

    if spikes is None:
        return (np.copy(current_spikes), current_spike_times)
    else:
        return (np.hstack((spikes[0], current_spikes)),
                np.hstack((spikes[1], current_spike_times)))

DT = 1.0
INPUT_SCALE = 90.0

NUM_PN = 28 * 28
NUM_KC = 10000
NUM_MBON = 10
NUM_PRESENT_TIMESTEPS = 20
NUM_REST_TIMESTEPS = 10

PN_KC_WEIGHT = 0.5

# Load MNIST data
training_images, training_labels = get_training_data()
testing_images, testing_labels = get_testing_data()

assert training_images.shape[1] == NUM_PN
assert testing_images.shape[1] == NUM_PN
assert np.max(training_labels) == (NUM_MBON - 1)
assert np.max(testing_labels) == (NUM_MBON - 1)

# GeNN current source model
cs_model = genn_model.create_custom_current_source_class(
    "cs_model",
    var_name_types=[("magnitude", "scalar")],
    injection_code="$(injectCurrent, $(magnitude));")

pn_params = {
    "C": 1.0,
    "TauM": 20.0,
    "Vrest": -60.0,
    "Vreset": -60.0,
    "Vthresh": -50.0,
    "Ioffset": 0.0,
    "TauRefrac": 100.0}

kc_params = {
    "C": 0.2,
    "TauM": 20.0,
    "Vrest": -60.0,
    "Vreset": -60.0,
    "Vthresh": -50.0,
    "Ioffset": 0.0,
    "TauRefrac": 2.0}

lif_init = {"V": -60.0, "RefracTime": 0.0}

pn_kc_init = {"g": PN_KC_WEIGHT}

pn_kc_fixed_prob = {"prob": 10.0 / NUM_PN}
cs_init = {"magnitude": 0.0}

post_syn_params = {"tau": 3.0}

# Create model
model = genn_model.GeNNModel("float", "mnist_mb")
model.dT = DT

# Create populations
pn = model.add_neuron_population("pn", NUM_PN, "LIF", pn_params, lif_init)
kc = model.add_neuron_population("kc", NUM_KC, "LIF", kc_params, lif_init)

pn_input = model.add_current_source("pn_input", cs_model, "pn" , {}, cs_init)

model.add_synapse_population("pn_kc", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
    pn, kc,
    "StaticPulse", {}, pn_kc_init, {}, {},
    "ExpCurr", post_syn_params, {},
    genn_model.init_connectivity("FixedProbabilityNoAutapse", pn_kc_fixed_prob))


model.build()
model.load()

single_example_timesteps = NUM_PRESENT_TIMESTEPS + NUM_REST_TIMESTEPS

# Get views to efficiently access state variables
pn_input_current_view = pn_input.vars["magnitude"].view
pn_refrac_time_view = pn.vars["RefracTime"].view

pn_spikes = None
kc_spikes = None

while model.timestep < (single_example_timesteps * 1):
    # Calculate the timestep within the presentation
    timestep_in_example = model.timestep % single_example_timesteps
    example = 5#int(model.timestep // single_example_timesteps)
    
    # If this is the first timestep of the presentation
    if timestep_in_example == 0:
        input_vector = training_images[example] / float(np.sum(training_images[example]))
        pn_input_current_view[:] = input_vector * INPUT_SCALE
        model.push_var_to_device("pn_input", "magnitude")
        
        pn_refrac_time_view[:] = 0.0
        model.push_var_to_device("pn", "RefracTime")
        
    # Otherwise, if this timestep is the start of the resting period
    elif timestep_in_example == NUM_PRESENT_TIMESTEPS:
        pn_input_current_view[:] = 0.0
        model.push_var_to_device("pn_input", "magnitude")
        
    model.step_time()
    
    model.pull_current_spikes_from_device("pn")
    model.pull_current_spikes_from_device("kc")
    
    pn_spikes = record_current_spikes(pn, pn_spikes, model.t)
    kc_spikes = record_current_spikes(kc, kc_spikes, model.t)
    
print(len(kc_spikes[0]))
fig, axes = plt.subplots(2, sharex=True)
axes[0].scatter(pn_spikes[1], pn_spikes[0], s=1)
axes[1].scatter(kc_spikes[1], kc_spikes[0], s=1)

#axes[0].vlines(np.arange(0, single_example_timesteps * 100, single_example_timesteps), ymin=0, ymax=NUM_PN)
#axes[1].vlines(np.arange(0, single_example_timesteps * 100, single_example_timesteps), ymin=0, ymax=NUM_KC)
plt.show()
