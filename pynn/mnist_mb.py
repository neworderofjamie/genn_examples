import numpy as np
import matplotlib.pyplot as plt
from os import path
from struct import unpack

from gzip import decompress
from urllib import request
from pygenn import genn_model, genn_wrapper

import sys

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

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
DT = 1.0
INPUT_SCALE = 0.01

NUM_PN = 28 * 28
NUM_KC = 20000
NUM_MBON = 10
NUM_PRESENT_TIMESTEPS = 20
NUM_REST_TIMESTEPS = 1000

PN_KC_SPARSITY = float(10 * NUM_KC) / float(NUM_KC * NUM_PN)

# PN params - large refractory period so only spikes once per presentation and increased capacitance
PN_PARAMS = {
    "C": 1.0,
    "TauM": 20.0,
    "Vrest": -60.0,
    "Vreset": -60.0,
    "Vthresh": -50.0,
    "Ioffset": 0.0,
    "TauRefrac": 100.0}

# KC params - standard LIF neurons
KC_PARAMS = {
    "C": 0.2,
    "TauM": 20.0,
    "Vrest": -60.0,
    "Vreset": -60.0,
    "Vthresh": -50.0,
    "Ioffset": 0.0,
    "TauRefrac": 2.0}

# GGN params - huge threshold so, essentially, non-spiking
GGN_PARAMS = {
    "C": 0.2,
    "TauM": 20.0,
    "Vrest": -60.0,
    "Vreset": -60.0,
    "Vthresh": 10000.0,
    "Ioffset": 0.0,
    "TauRefrac": 2.0}

# MBON params - standard LIF neurons
MBON_PARAMS = {
    "C": 0.2,
    "TauM": 20.0,
    "Vrest": -60.0,
    "Vreset": -60.0,
    "Vthresh": -50.0,
    "Ioffset": 0.0,
    "TauRefrac": 2.0}

PN_KC_WEIGHT = 0.35
PN_KC_TAU_SYN = 3.0

KC_GGN_WEIGHT = 0.015
KC_GGN_TAU_SYN = 5.0

GGN_KC_WEIGHT = -4.0
GGN_KC_TAU_SYN = 4.0
GGN_KC_PARAMS = {"Vmid": -54.1,
                 "Vslope": 1.0,
                 "Vthresh": -60.0}

KC_MBON_WEIGHT = {"min": 0.00, "max": 0.05}
KC_MBON_TAU_SYN = 8.0
KC_MBON_PARAMS = {"tau": 10.0,
                  "rho": 0.0,
                  "eta": 0.01,
                  "wMin": 0.0,
                  "wMax": 0.1}

MBON_MBON_WEIGHT = -5.0
MBON_MBON_TAU_SYN = 3.0
# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
# Load MNIST data
training_images, training_labels = get_training_data()
testing_images, testing_labels = get_testing_data()

assert training_images.shape[1] == NUM_PN
assert testing_images.shape[1] == NUM_PN
assert np.max(training_labels) == (NUM_MBON - 1)
assert np.max(testing_labels) == (NUM_MBON - 1)

# Current source model, allowing current to be injected into neuron from variable
cs_model = genn_model.create_custom_current_source_class(
    "cs_model",
    var_name_types=[("magnitude", "scalar")],
    injection_code="$(injectCurrent, $(magnitude));")

# Model for graded synapses with exponential activation
graded_synapse_model = genn_model.create_custom_weight_update_class(
    "graded_synapse_model",
    param_names=["Vmid", "Vslope", "Vthresh"],
    var_name_types=[("g", "scalar")],
    event_code="$(addToInSyn, DT * $(g) * max(0.0, 1.0 / (1.0 + exp(($(Vmid) - $(V_pre)) / $(Vslope)))));",
    event_threshold_condition_code="$(V_pre) > $(Vthresh)")

lateral_inhibition = genn_model.create_custom_init_var_snippet_class(
    "lateral_inhibition",
    param_names=["g"],
    var_init_code="$(value)=($(id_pre)==$(id_post)) ? 0.0 : $(g);")

# STDP synapse with additive weight dependence
symmetric_stdp = genn_model.create_custom_weight_update_class(
    "symmetric_stdp",
    param_names=["tau", "rho", "eta", "wMin", "wMax"],
    var_name_types=[("g", "scalar")],
    sim_code=
        """
        $(addToInSyn, $(g));
        const scalar dt = $(t) - $(sT_post);
        const scalar timing = exp(-dt / $(tau)) - $(rho);
        const scalar newWeight = $(g) + ($(eta) * timing);
        $(g) = fmin($(wMax), fmax($(wMin), newWeight));
        """,
    learn_post_code=
        """
        const scalar dt = $(t) - $(sT_pre);
        const scalar timing = exp(-dt / $(tau));
        const scalar newWeight = $(g) + ($(eta) * timing);
        $(g) = fmin($(wMax), fmax($(wMin), newWeight));
        """,
    is_pre_spike_time_required=True,
    is_post_spike_time_required=True)

# Create model
model = genn_model.GeNNModel("float", "mnist_mb")
model.dT = DT
model._model.set_seed(1337)

# Create neuron populations
lif_init = {"V": -60.0, "RefracTime": 0.0}
pn = model.add_neuron_population("pn", NUM_PN, "LIF", PN_PARAMS, lif_init)
kc = model.add_neuron_population("kc", NUM_KC, "LIF", KC_PARAMS, lif_init)
ggn = model.add_neuron_population("ggn", 1, "LIF", GGN_PARAMS, lif_init)
mbon = model.add_neuron_population("mbon", NUM_MBON, "LIF", MBON_PARAMS, lif_init)

# Create current source to deliver input to network
pn_input = model.add_current_source("pn_input", cs_model, "pn" , {}, {"magnitude": 0.0})

# Create synapse populations
model.add_synapse_population("pn_kc", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
    pn, kc,
    "StaticPulse", {}, {"g": PN_KC_WEIGHT}, {}, {},
    "ExpCurr", {"tau": PN_KC_TAU_SYN}, {},
    genn_model.init_connectivity("FixedProbabilityNoAutapse", {"prob": PN_KC_SPARSITY}))

model.add_synapse_population("kc_ggn", "DENSE_GLOBALG", genn_wrapper.NO_DELAY,
                             kc, ggn,
                             "StaticPulse", {}, {"g": KC_GGN_WEIGHT}, {}, {},
                             "ExpCurr", {"tau": KC_GGN_TAU_SYN}, {})

model.add_synapse_population("ggn_kc", "DENSE_GLOBALG", genn_wrapper.NO_DELAY,
                             ggn, kc,
                             graded_synapse_model, GGN_KC_PARAMS, {"g": GGN_KC_WEIGHT}, {}, {},
                             "ExpCurr", {"tau": GGN_KC_TAU_SYN}, {})

kc_mbon = model.add_synapse_population("kc_mbon", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                       kc, mbon,
                                       symmetric_stdp, KC_MBON_PARAMS, {"g": genn_model.init_var("Uniform", KC_MBON_WEIGHT)}, {}, {},
                                       "ExpCurr", {"tau": KC_MBON_TAU_SYN}, {})

model.add_synapse_population("mbon_mbon", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                             mbon, mbon,
                             "StaticPulse", {}, {"g": genn_model.init_var(lateral_inhibition, {"g": MBON_MBON_WEIGHT})}, {}, {},
                             "ExpCurr", {"tau": MBON_MBON_TAU_SYN}, {})

# Build model and load it
model.build()
model.load()

single_example_timesteps = NUM_PRESENT_TIMESTEPS + NUM_REST_TIMESTEPS

# Get views to efficiently access state variables
pn_input_current_view = pn_input.vars["magnitude"].view
pn_refrac_time_view = pn.vars["RefracTime"].view
ggn_v_view = ggn.vars["V"].view
kc_mbon_view = kc_mbon.vars["g"].view

model.pull_var_from_device("kc_mbon", "g")
before = np.copy(kc_mbon_view)
plot = True

label_spikes = np.zeros((NUM_MBON, NUM_MBON), dtype=int)

pn_spikes = None
kc_spikes = None
mbon_spikes = None

NUM_STIM = 100
while model.timestep < (single_example_timesteps * NUM_STIM):
    # Calculate the timestep within the presentation
    timestep_in_example = model.timestep % single_example_timesteps
    example = int(model.timestep // single_example_timesteps)
    
    # If this is the first timestep of the presentation
    if timestep_in_example == 0:
        # Get image
        input_vector = training_images[example]
        
        # Normalize
        pn_input_current_view[:] = input_vector * INPUT_SCALE
        model.push_var_to_device("pn_input", "magnitude")
        
        pn_refrac_time_view[:] = 0.0
        model.push_var_to_device("pn", "RefracTime")
        
    # Otherwise, if this timestep is the start of the resting period
    elif timestep_in_example == NUM_PRESENT_TIMESTEPS:
        pn_input_current_view[:] = 0.0
        model.push_var_to_device("pn_input", "magnitude")
        
    model.step_time()
    
    if plot:
        model.pull_current_spikes_from_device("pn")
        pn_spikes = record_current_spikes(pn, pn_spikes, model.t)

    model.pull_current_spikes_from_device("kc")
    kc_spikes = record_current_spikes(kc, kc_spikes, model.t)

    model.pull_current_spikes_from_device("mbon")
    label_spikes[training_labels[example]] += np.bincount(mbon.current_spikes, minlength=NUM_MBON)
    mbon_spikes = record_current_spikes(mbon, mbon_spikes, model.t)

model.pull_var_from_device("kc_mbon", "g")
after = np.copy(kc_mbon_view)

stim_bins = np.arange(0, single_example_timesteps * NUM_STIM, single_example_timesteps)

if plot:
    fig, axes = plt.subplots(4, sharex=True)

    axes[0].set_title("PN")
    axes[0].scatter(pn_spikes[1], pn_spikes[0], s=1)

    axes[1].set_title("KC")
    axes[1].scatter(kc_spikes[1], kc_spikes[0], s=1)

    axes[2].set_title("MBON")
    axes[2].scatter(mbon_spikes[1], mbon_spikes[0], s=1)

kc_spike_counts = np.histogram(kc_spikes[1], bins=stim_bins)[0]
mbon_spike_counts = np.histogram(mbon_spikes[1], bins=stim_bins)[0]
print("KC spikes: min=%f, max=%f, mean=%f" % (np.amin(kc_spike_counts), np.amax(kc_spike_counts), np.average(kc_spike_counts)))
print("MBON spikes: min=%f, max=%f, mean=%f" % (np.amin(mbon_spike_counts), np.amax(mbon_spike_counts), np.average(mbon_spike_counts)))

print("Before: min=%f, max=%f" % (np.amin(before), np.amax(before)))
print("After: min=%f, max=%f" % (np.amin(after), np.amax(after)))
if plot:
    axes[3].plot(stim_bins[:-1], kc_spike_counts)

    axes[0].vlines(stim_bins, ymin=0, ymax=NUM_PN, linestyle="--", color="gray")
    axes[1].vlines(stim_bins, ymin=0, ymax=NUM_KC, linestyle="--", color="gray")
    axes[2].vlines(stim_bins, ymin=0, ymax=NUM_MBON, linestyle="--", color="gray")
    axes[3].vlines(stim_bins, ymin=0, ymax=np.amax(kc_spikes), linestyle="--", color="gray")
    '''
    axes[4].hist(before, bins=10)
    axes[5].hist(before, bins=10)
    for i, a in enumerate(axes[6:]):
        a.set_title(str(i))
        a.bar(range(NUM_MBON), label_spikes[i])
    '''
    fig.savefig("test.png")
    plt.show()
