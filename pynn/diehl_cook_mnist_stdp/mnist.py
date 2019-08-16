import numpy as np 
import matplotlib.pyplot as plt 

import time
import os 
from struct import unpack

from gzip import decompress
from urllib import request
from pygenn import genn_model, genn_wrapper


# ********************************************************************************
#                      Methods
# ********************************************************************************
def get_image_data(url, filename, correct_magic):
    if os.path.exists(filename):
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
    if os.path.exists(filename):
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

def accuracy(predictions, y_list):
    return np.sum(np.array(predictions) == np.array(y_list)) / float(len(y_list)) * 100

# ********************************************************************************
#                      Model Definitions
# ********************************************************************************

# Poisson model
poisson_model = genn_model.create_custom_neuron_class(
    "poisson_model",
    var_name_types=[("timeStepToSpike", "scalar"),("frequency","scalar")],
    sim_code="""
    if($(timeStepToSpike) <= 0.0f) {
        $(timeStepToSpike) += 1.0 / $(frequency);
    }
    $(timeStepToSpike) -= 1.0;
    """,
    threshold_condition_code="$(timeStepToSpike) <= 0.0"
)

# LIF neuron model
# excitatory neurons
lif_e_model = genn_model.create_custom_neuron_class(
    "lif_e_model",
    param_names=["Tau", "Erest", "Vreset", "Vthres", "RefracPeriod", "tauTheta", "thetaPlus"],
    var_name_types=[("V","scalar"), ("RefracTime", "scalar"), ("theta", "scalar"), ("SpikeNumber", "unsigned int")],
    sim_code="""
    if ($(RefracTime) <= 0.0) {
        const scalar alpha = ($(Isyn) * $(Tau)) + $(Erest);
        $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
    }
    else {
        $(RefracTime) -= DT;
    }
    $(theta) = $(theta) * $(ExpTtheta);
    """,
    reset_code="""
    $(V) = $(Vreset);
    $(RefracTime) = $(RefracPeriod);
    $(SpikeNumber) += 1;
    $(theta) += $(thetaPlus);
    """,
    threshold_condition_code="$(RefracTime) <= 0.0 && $(V) > ($(theta) + $(Vthres))",
    derived_params=[
        ("ExpTC", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))()),
        ("ExpTtheta", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[5]))())
    ]
)

# inhibitory neurons
lif_i_model = genn_model.create_custom_neuron_class(
    "lif_i_model",
    param_names=["Tau","Erest","Vreset","Vthres","RefracPeriod"],
    var_name_types=[("V","scalar"),("RefracTime","scalar")],
    sim_code="""
    if ($(RefracTime) <= 0.0)  {
        const scalar alpha = ($(Isyn) * $(Tau)) + $(Erest);
        $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
    }
    else  {
        $(RefracTime) -= DT;
    }
    """,
    reset_code="""
    $(V) = $(Vreset);
    $(RefracTime) = $(RefracPeriod);
    """,
    threshold_condition_code="$(RefracTime) <= 0.0 && $(V) >= $(Vthres)",
    derived_params=[
        ("ExpTC", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())
    ]
)

# STDP
stdp_model = genn_model.create_custom_weight_update_class(
    "stdp_model",
    param_names=["tauMinus", "gMax", "Xtar", "mu", "eta"],
    var_name_types=[("g", "scalar")],
    pre_var_name_types=[("Xpre", "scalar")],

    sim_code=
        """
        $(addToInSyn, $(g));
        """,

    learn_post_code=
        """
        const scalar dt = $(t) - $(sT_pre);
        if(dt > 0) {
            const scalar expXpre = $(Xpre) * exp(-dt / $(tauMinus));
            const scalar newG = $(g) + ($(eta) * (expXpre - $(Xtar)) * pow(($(gMax) - $(g)),$(mu)));
            $(g) = fmin($(gMax), fmax(0.0, newG));
        }
        """,

    pre_spike_code=
        """
        const scalar dt = $(t) - $(sT_pre);
        if(dt > 0) {
            $(Xpre) = ($(Xpre) * exp(-dt / $(tauMinus))) + 1.0;
        }
        """,

    is_pre_spike_time_required=True)

lateral_inhibition = genn_model.create_custom_init_var_snippet_class(
    "lateral_inhibition",
    param_names=['weight'],
    var_init_code="$(value)=($(id_pre)==$(id_post)) ? 0.0 : $(weight);"
)

# ********************************************************************************
#                      Data
# ********************************************************************************
start = time.time()
training_images, training_labels = get_training_data()
end = time.time()
print('time needed to load training set:', end - start)
 
start = time.time()
testing_images, testing_labels = get_testing_data()
end = time.time()
print('time needed to load test set:', end - start)

# ********************************************************************************
#                      Parameters and Hyperparameters
# ********************************************************************************

# Global 
dt = 1.0

# Architecture
num_examples = training_images.shape[0]
n_input = 784
n_e = 100
n_i = n_e
single_example_time = 350
resting_time = 150
train_timesteps = num_examples * (single_example_time + resting_time)
input_intensity = 2.
start_input_intensity = input_intensity

# Neuron
v_reset_e = -65
v_reset_i = -45
refrac_period_i = 2.0
e_rev_exc = 0.0
e_rev_inh = -100.0

# STDP
g_max = 1.0 / 1000.0
x_tar = 0.4
eta = 0.0000001
mu = 0.2

# Neuron group parameters
lif_e_params = {
    "Tau": 100.0, 
    "Erest": -65.0,
    "Vreset": -65.0,
    "Vthres": -52.0 - 20.0,
    "RefracPeriod": 5.0,
    "tauTheta": 1e7,
    "thetaPlus": 0.05
}

lif_i_params = {
    "Tau": 10.0, 
    "Erest": -60.0,
    "Vreset": -45.0,
    "Vthres": -40.0,
    "RefracPeriod": 2.0}

# Neuron group initial values
lif_e_init = {"V": v_reset_e - 40.0, "RefracTime": 0.0, "SpikeNumber": 0, "theta": 20.0}
lif_i_init = {"V": v_reset_i - 40.0, "RefracTime": 0.0}

poisson_init = {
    "timeStepToSpike": 0.0,
    "frequency": 0.0}

post_syn_e_params = {"tau": 1.0, "E":e_rev_exc}
post_syn_i_params = {"tau": 2.0, "E":e_rev_inh}

stdp_init = {"g":genn_model.init_var("Uniform",{"min":0.003 / 1000.0, "max":0.3 / 1000.0})}
stdp_params = {"tauMinus": 20.0,"gMax": g_max,"Xtar":x_tar,"mu":mu, "eta":eta}
stdp_pre_init = {"Xpre": 0.0}

static_e_init = {"g":17.0 / 1000.0}
static_i_init = {"g":genn_model.init_var(lateral_inhibition,{"weight":-10.0 / 1000.0})}

# ********************************************************************************
#                      Model Instances
# ********************************************************************************

model = genn_model.GeNNModel("float","mnist")
model.dT = dt

# Neuron populations
poisson_pop = model.add_neuron_population("poisson_pop", n_input, poisson_model, {}, poisson_init)

lif_e_pop = model.add_neuron_population("lif_e_pop", n_e, lif_e_model, lif_e_params, lif_e_init)

lif_i_pop = model.add_neuron_population("lif_i_pop", n_i, lif_i_model, lif_i_params, lif_i_init)

input_e_pop = model.add_synapse_population("input_e_pop", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
    poisson_pop, lif_e_pop,
    stdp_model, stdp_params, stdp_init, stdp_pre_init, {},
    "ExpCond", post_syn_e_params, {})

syn_e_pop = model.add_synapse_population("syn_e_pop", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
    lif_e_pop, lif_i_pop,
    "StaticPulse", {}, static_e_init, {}, {},
    "ExpCond", post_syn_e_params, {},
    genn_model.init_connectivity("OneToOne",{}))

syn_i_pop = model.add_synapse_population("syn_i_pop", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
    lif_i_pop, lif_e_pop,
    "StaticPulse", {}, static_i_init, {}, {},
    "ExpCond", post_syn_i_params, {})


# ********************************************************************************
#                      Building and Simulation
# ********************************************************************************

print("Building Model")
model.build()
print("Loading Model")
model.load()

frequency_view = poisson_pop.vars["frequency"].view
time_to_spike_view = poisson_pop.vars["timeStepToSpike"].view
spike_number_view = lif_e_pop.vars["SpikeNumber"].view
weight_view = input_e_pop.vars["g"].view
theta_view = lif_e_pop.vars["theta"].view

print("Simulating")

# Simulate

spike_number_view[:] = 0
model.push_var_to_device("lif_e_pop", "SpikeNumber")

i=0

while model.timestep < train_timesteps:
    # Calculate the timestep within the presentation
    timestep_in_example = model.timestep % (single_example_time + resting_time)

    # If this is the first timestep of the presentation
    if timestep_in_example == 0:
        # Calculate index of example
        example = int(model.timestep // (single_example_time + resting_time))
        print("Example %u" % example)

         # Divide training pixel values by 4 to get Hz
        rates_hz = training_images[example%60000] / 4.0

        # Scale these by timestep in seconds to get spikes per timestep
        frequency_view[:] = rates_hz * (dt / 1000.0)
        time_to_spike_view[:] = 0.0
        model.push_state_to_device("poisson_pop")
    # Otherwise, if this timestep is the start of the resting period
    elif timestep_in_example == single_example_time:
        # Set poisson rates to 0(ish)
        frequency_view[:] = 0.000001
        time_to_spike_view[:] = 0.0
        model.push_state_to_device("poisson_pop")

    # Advance simulation
    model.step_time()

# Save theta
model.pull_var_from_device("lif_e_pop", "theta")
np.save("theta.npy", theta_view)

# Save weights
model.pull_var_from_device("input_e_pop", "g")
np.save("weights.npy", weight_view)

assert(False)
'''
- All excitatory neurons spike the same number of times since its an all to all connection
'''
        
# ********************************************************************************
#                      Training and Classification
# ********************************************************************************

print()
print("Classifying examples")

# Set SpikeNumber to 0
lif_e_pop.set_var("SpikeNumber",0)

spike_number_record = np.zeros((n_e,10))

i=0
old_spike_number = np.zeros((n_e))
spike_number_record = np.zeros((n_e,10))
current_t = model.t
while model.t < current_t + runtime:
    model.step_time()
    if model.t >= current_t + (single_example_time + resting_time) * (i+1):
        # print(spike_number_view)
        # print(current_spike_number - old_spike_number)
        model.pull_var_from_device("lif_e_pop", "SpikeNumber")
        spike_number_record[:,label] += spike_number_view

        print(spike_number_record)
        print("Example: {} Label: {}".format(i,label))
        i += 1
        rates = list(training['x'][i%60000,:,:].reshape((n_input)) / 8000. * input_intensity)
        label = int(training['y'][i%num_examples])

        poisson_pop.set_var('frequency', rates)
        poisson_pop.set_var('timeStepToSpike',0.0)
        model.push_state_to_device("poisson_pop")

        lif_e_pop.set_var("SpikeNumber",0)
        model.push_var_to_device("lif_e_pop", "SpikeNumber")


neuron_labels = np.argmax(spike_number_record,axis=1)
print()
print("Neuron labels")
print(neuron_labels)
# print(spike_number_record)

# ********************************************************************************
#                      Evaluation on Training set
# ********************************************************************************
"""
print()
print()
print("Evaluating on training set")

# Set eta to 0
input_e_pop.set_var("eta",0.0)

# Set SpikeNumber to 0
lif_e_pop.set_var("SpikeNumber",0)
lif_i_pop.set_var("SpikeNumber",0)
model.push_state_to_device("lif_e_pop")
model.push_state_to_device("lif_i_pop")

predictions = []
y_list = list(training['y'][:num_examples].reshape((num_examples)))

for i in range(num_examples):
    digit_count = np.empty((10))
    rates = list(training['x'][i%num_examples,:,:].reshape((n_input)) / 8000. * input_intensity)
    label = int(training['y'][i%num_examples])

    poisson_pop.set_var('frequency', rates)
    poisson_pop.set_var('timeStepToSpike',0.0)
    lif_e_pop.set_var("SpikeNumber",0)
    model.push_state_to_device("lif_e_pop")
    lif_i_pop.set_var("SpikeNumber",0)
    model.push_state_to_device("lif_i_pop")
    spike_number_view = lif_e_pop.vars["SpikeNumber"].view 
    model.push_state_to_device("poisson_pop")
    
    while model.t < single_example_time:
        model.step_time()
        model.pull_state_from_device(input_e_pop)
        model.pull_state_from_device(lif_e_pop)
    
    for j in range(n_e):
        # print(spike_number_view[j])
        digit_count[neuron_labels[j]] += spike_number_view[j]
    
    pred = np.argmax(digit_count,axis=0)
    predictions.append(pred)

# print(predictions[:50])
# print(y_list[:50])
print("Accuracy: {}%".format(accuracy(predictions,y_list)))
"""
