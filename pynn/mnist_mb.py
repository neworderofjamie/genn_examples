import numpy as np
import matplotlib.pyplot as plt
from os import path
from struct import unpack

from itertools import combinations
from pygenn import genn_model, genn_wrapper

import sys

def get_image_data(raw_filename, filename, correct_magic):
    if path.exists(filename):
        print("Loading existing data")
        return np.load(filename)
    else:
        with open(raw_filename, "rb") as f:
            image_data = f.read()
            
            # Unpack header from first 16 bytes of buffer
            magic, num_items, num_rows, num_cols = unpack('>IIII', image_data[:16])
            assert magic == correct_magic
            assert num_rows == 28
            assert num_cols == 28

            # Convert remainder of buffer to numpy bytes
            image_data_np = np.frombuffer(image_data[16:], dtype=np.uint8)

            # Reshape data into individual images
            image_data_np = np.reshape(image_data_np, (num_items, num_rows * num_cols))

            # Convert image data to float and normalise
            image_data_np = image_data_np.astype(np.float)
            image_magnitude = np.sum(image_data_np, axis=1)
            for i in range(num_items):
                image_data_np[i] /= image_magnitude[i]

            # Write to disk
            np.save(filename, image_data_np)

            return image_data_np

def get_label_data(raw_filename, filename, correct_magic):
    if path.exists(filename):
        print("Loading existing data")
        return np.load(filename)
    else:
        with open(raw_filename, "rb") as f:
            label_data = f.read()

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
    images = get_image_data("train-images-idx3-ubyte", "training_images.npy", 2051)
    labels = get_label_data("train-labels-idx1-ubyte", "training_labels.npy", 2049)
    assert images.shape[0] == labels.shape[0]

    return images, labels

def get_testing_data():
    images = get_image_data("t10k-images-idx3-ubyte", "testing_images.npy", 2051)
    labels = get_label_data("t10k-labels-idx1-ubyte", "testing_labels.npy", 2049)
    assert images.shape[0] == labels.shape[0]

    return images, labels

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
DT = 0.02
INPUT_SCALE = 400.0

TRAIN = False
RECORD_V = not TRAIN

NUM_PN = 28 * 28
NUM_KC = 20000
NUM_MBON = 10
PRESENT_TIME_MS = 20.0

PRESENT_TIMESTEPS = int(round(PRESENT_TIME_MS / DT))

PN_KC_COL_LENGTH = 10

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

# MBON params - standard LIF neurons
MBON_PARAMS = {
    "C": 0.2,
    "TauM": 20.0,
    "Vrest": -60.0,
    "Vreset": -60.0,
    "Vthresh": -50.0,
    "Ioffset": 0.0,
    "TauRefrac": 2.0}

MBON_STIMULUS_CURRENT = 5.0
PN_KC_WEIGHT = 0.3
PN_KC_TAU_SYN = 3.0
KC_MBON_WEIGHT = 0.0
KC_MBON_TAU_SYN = 3.0
KC_MBON_PARAMS = {"tau": 10.0,
                  "rho": 0.005,
                  "eta": 0.001,
                  "wMin": 0.0,
                  "wMax": 0.1}


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

# STDP synapse with additive weight dependence
symmetric_stdp = genn_model.create_custom_weight_update_class(
    "symmetric_stdp",
    param_names=["tau", "rho", "eta", "wMin", "wMax"],
    var_name_types=[("g", "scalar")],
    sim_code=
        """
        const scalar dt = $(t) - $(sT_post);
        const scalar timing = exp(-dt / $(tau)) - $(rho);
        const scalar newWeight = $(g) + ($(eta) * timing);
        $(g) = fmin($(wMax), fmax($(wMin), newWeight));
        """,
    learn_post_code=
        """
        const scalar dt = $(t) - $(sT_pre);
        const scalar timing = exp(-dt / $(tau)) - $(rho);
        const scalar newWeight = $(g) + ($(eta) * timing);
        $(g) = fmin($(wMax), fmax($(wMin), newWeight));
        """,
    is_pre_spike_time_required=True,
    is_post_spike_time_required=True)

# custom IF neuron for gain control
IF_neuron = genn_model.create_custom_neuron_class(
    "IF_neuron",
    param_names=["theta"],
    var_name_types=[("V", "scalar")],
    sim_code=
    """
    $(V)+= $(Isyn);
    """,
    threshold_condition_code=
    """
    $(V) >= $(theta)
    """,
    reset_code=
    """
    $(V)= 0.0;
    """)

# Create model
model = genn_model.GeNNModel("float", "mnist_mb")
model.dT = DT
model._model.set_seed(1337)

# Create neuron populations
lif_init = {"V": -60.0, "RefracTime": 0.0}
pn = model.add_neuron_population("pn", NUM_PN, "LIF", PN_PARAMS, lif_init)
kc = model.add_neuron_population("kc", NUM_KC, "LIF", KC_PARAMS, lif_init)
mbon = model.add_neuron_population("mbon", NUM_MBON, "LIF", MBON_PARAMS, lif_init)
ggn= model.add_neuron_population("ggn", 1, IF_neuron, {"theta": 100}, {"V": 0.0})

# Turn on spike recording
pn.spike_recording_enabled = not TRAIN
kc.spike_recording_enabled = not TRAIN
mbon.spike_recording_enabled = not TRAIN
ggn.spike_recording_enabled= not TRAIN

# Create current sources to deliver input and supervision to network
pn_input = model.add_current_source("pn_input", cs_model, pn , {}, {"magnitude": 0.0})
mbon_input = model.add_current_source("mbon_input", cs_model, mbon , {}, {"magnitude": 0.0})

# Create synapse populations
pn_kc = model.add_synapse_population("pn_kc", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
    pn, kc,
    "StaticPulse", {}, {"g": PN_KC_WEIGHT}, {}, {},
    "ExpCurr", {"tau": PN_KC_TAU_SYN}, {},
    genn_model.init_connectivity("FixedNumberPreWithReplacement", {"colLength": PN_KC_COL_LENGTH}))

kc_ggn = model.add_synapse_population("kc_ggn", "DENSE_GLOBALG", genn_wrapper.NO_DELAY, kc, ggn, "StaticPulse", {}, {"g": 1.0}, {}, {}, "DeltaCurr", {}, {})

ggn_kc = model.add_synapse_population("ggn_kc", "DENSE_GLOBALG", genn_wrapper.NO_DELAY, ggn, kc, "StaticPulse", {}, {"g": -5.0}, {}, {}, "ExpCurr", {"tau": 5.0}, {})

if TRAIN:
    kc_mbon = model.add_synapse_population("kc_mbon", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                           kc, mbon,
                                           symmetric_stdp, KC_MBON_PARAMS, {"g": KC_MBON_WEIGHT}, {}, {},
                                           "ExpCurr", {"tau": KC_MBON_TAU_SYN}, {})
else:
    kc_mbon = model.add_synapse_population("kc_mbon", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                           kc, mbon,
                                           "StaticPulse", {}, {"g": np.load("kc_mbon_g.npy").flatten()}, {}, {},
                                           "ExpCurr", {"tau": KC_MBON_TAU_SYN}, {})

mbon_mbon = model.add_synapse_population("mbon_mbon", "DENSE_GLOBALG", genn_wrapper.NO_DELAY, mbon, mbon, "StaticPulse", {}, {"g": -5.0}, {}, {}, "ExpCurr", {"tau": 5.0}, {})
    
# Build model and load it
model.build()
model.load(num_recording_timesteps=PRESENT_TIMESTEPS)

# Get views to efficiently access state variables
pn_input_current_view = pn_input.vars["magnitude"].view
pn_refrac_time_view = pn.vars["RefracTime"].view
pn_v_view = pn.vars["V"].view
ggn_v_view= ggn.vars["V"].view
kc_refrac_time_view = kc.vars["RefracTime"].view
kc_v_view = kc.vars["V"].view
mbon_input_current_view = mbon_input.vars["magnitude"].view
mbon_refrac_time_view = mbon.vars["RefracTime"].view
mbon_v_view = mbon.vars["V"].view
kc_mbon_g_view = kc_mbon.vars["g"].view

pn_kc_insyn_view = pn_kc._assign_ext_ptr_array("inSyn", NUM_KC, "scalar")
kc_mbon_insyn_view = kc_mbon._assign_ext_ptr_array("inSyn", NUM_MBON, "scalar")
ggn_kc_insyn_view= ggn_kc._assign_ext_ptr_array("inSyn", NUM_KC, "scalar")
mbon_mbon_insyn_view= mbon_mbon._assign_ext_ptr_array("inSyn", NUM_MBON, "scalar")

if TRAIN:
    kc_spike_time_view = kc._assign_ext_ptr_array("sT", NUM_KC, "scalar")
    mbon_spike_time_view = mbon._assign_ext_ptr_array("sT", NUM_MBON, "scalar")

plot = not TRAIN

images = training_images# if TRAIN else testing_images
labels = training_labels# if TRAIN else testing_labels

# Loop through stimuli
pn_spikes = ([], [])
kc_spikes = ([], [])
mbon_spikes = ([], [])
ggn_spikes= ([], [])
mbon_v = []
ggn_v= []
if TRAIN:
    NUM_STIM = 60000
else:
    NUM_STIM = 2000
for s in range(NUM_STIM):
    if s % 500 == 0:
        print(s)
    # Set training image
    pn_input_current_view[:] = images[s] * INPUT_SCALE
    pn_input.push_var_to_device("magnitude")
    
    # Turn on correct output neuron
    if TRAIN:
        mbon_input_current_view[:] = 0
        mbon_input_current_view[labels[s]] = MBON_STIMULUS_CURRENT
        mbon_input.push_var_to_device("magnitude")
    
    # Loop through stimuli presentation
    for i in range(PRESENT_TIMESTEPS):
        model.step_time()
        
        if RECORD_V:
            mbon.pull_var_from_device("V")
            mbon_v.append(np.copy(mbon_v_view))
            #ggn.pull_var_from_device("V")
            #ggn_v.append(np.copy(ggn_v_view))
    
    # Reset neurons
    pn_refrac_time_view[:] = 0.0
    pn_v_view[:] = -60.0
    kc_refrac_time_view[:] = 0.0
    kc_v_view[:] = -60.0
    mbon_refrac_time_view[:] = 0.0
    mbon_v_view[:] = -60.0
    ggn_v_view[:]= 0.0;
    pn_kc_insyn_view[:] = 0.0
    kc_mbon_insyn_view[:] = 0.0
    ggn_kc_insyn_view[:] = 0.0
    mbon_mbon_insyn_view[:] = 0.0
    pn.push_var_to_device("RefracTime")
    pn.push_var_to_device("V")
    kc.push_var_to_device("RefracTime")
    kc.push_var_to_device("V")
    mbon.push_var_to_device("RefracTime")
    mbon.push_var_to_device("V")
    ggn.push_var_to_device("V")
    model.push_var_to_device("pn_kc", "inSyn")
    model.push_var_to_device("kc_mbon", "inSyn")
    model.push_var_to_device("ggn_kc", "inSyn")
    model.push_var_to_device("mbon_mbon", "inSyn")

    if TRAIN:
        kc_spike_time_view[:] = -np.finfo(np.float32).max
        mbon_spike_time_view[:] = -np.finfo(np.float32).max
        model.push_var_to_device("SpikeTimes", "mbon")
        model.push_var_to_device("SpikeTimes", "kc")

    if plot:
        model.pull_recording_buffers_from_device();

        pn_spike_times, pn_spike_ids = pn.spike_recording_data
        kc_spike_times, kc_spike_ids = kc.spike_recording_data
        mbon_spike_times, mbon_spike_ids = mbon.spike_recording_data
        ggn_spike_times, ggn_spike_ids = ggn.spike_recording_data
        
        pn_spikes[0].append(pn_spike_times)
        pn_spikes[1].append(pn_spike_ids)

        kc_spikes[0].append(kc_spike_times)
        kc_spikes[1].append(kc_spike_ids)

        mbon_spikes[0].append(mbon_spike_times)
        mbon_spikes[1].append(mbon_spike_ids)

        ggn_spikes[0].append(ggn_spike_times)
        ggn_spikes[1].append(ggn_spike_ids)
        
# Save weights
if TRAIN:
    kc_mbon.pull_var_from_device("g")
    kc_mbon_g_view = np.reshape(kc_mbon_g_view, (NUM_KC, NUM_MBON))
    np.save("kc_mbon_g.npy", kc_mbon_g_view)

if plot:
    spike_fig, spike_axes = plt.subplots(4 if RECORD_V else 3, sharex="col")

    # Plot spikes
    spike_axes[0].scatter(np.concatenate(pn_spikes[0]), np.concatenate(pn_spikes[1]), s=1)
    spike_axes[1].scatter(np.concatenate(kc_spikes[0]), np.concatenate(kc_spikes[1]), s=1)
    spike_axes[2].scatter(np.concatenate(mbon_spikes[0]), np.concatenate(mbon_spikes[1]), s=1)
    spike_axes[2].scatter(np.concatenate(ggn_spikes[0]), np.concatenate(ggn_spikes[1]), c="r", s=1)
    # Plot voltages
    if RECORD_V:
        spike_axes[3].plot(np.arange(0.0, PRESENT_TIME_MS * NUM_STIM, DT), np.vstack(mbon_v))
#        spike_axes[3].plot(np.arange(0.0, PRESENT_TIME_MS * NUM_STIM, DT), np.vstack(ggn_v),"--")
    
    # Mark stimuli changes on figure
    stimuli_bounds = np.arange(0.0, NUM_STIM * PRESENT_TIME_MS, PRESENT_TIME_MS)
    spike_axes[0].vlines(stimuli_bounds, ymin=0, ymax=NUM_PN, linestyle="--")
    spike_axes[1].vlines(stimuli_bounds, ymin=0, ymax=NUM_KC, linestyle="--")
    spike_axes[2].vlines(stimuli_bounds, ymin=0, ymax=NUM_MBON, linestyle="--")
    
    # Label axes
    spike_axes[0].set_title("PN")
    spike_axes[1].set_title("KC")
    spike_axes[2].set_title("MBON")
    
    if RECORD_V:
        spike_axes[3].set_title("MBON/GGN")
    
    # Show classification output
    for b, t, s, l in zip(stimuli_bounds, mbon_spikes[0], mbon_spikes[1], labels):
        if len(s) > 0:
            first_spike = np.argmin(t)
            classification = s[first_spike]
            #classification = np.argmax(np.bincount(s, minlength=NUM_MBON))
            colour = "green" if classification == l else "red"
            spike_axes[2].hlines(classification, b, b + PRESENT_TIME_MS, 
                                 color=colour, alpha=0.5)
    
    # Show training labels
    for i, x in enumerate(stimuli_bounds):
        spike_axes[0].text(x, 20, labels[i])
    
    if TRAIN:
        fig, axis = plt.subplots()
        axis.hist(kc_mbon_g_view.flatten(), bins=100)
    #unique_neurons = [np.unique(k) for k in kc_spikes[1]]
    #for (i, a), (j, b) in combinations(enumerate(unique_neurons), 2):
    #    print("label %u vs %u = %u" % (labels[i], labels[j], len(np.intersect1d(a, b))))
    
    plt.show()
