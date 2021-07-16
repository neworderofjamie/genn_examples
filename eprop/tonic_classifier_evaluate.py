import csv
import numpy as np
import tonic
import matplotlib.pyplot as plt

from time import perf_counter
from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY

from dataloader import DataLoader
# Eprop imports
#import eprop

TIMESTEP_MS = 1.0
TIMING_ENABLED = True
SHD = True

MAX_STIMULI_TIME = 1369.140625 if SHD else 1568.0
MAX_SPIKES_PER_STIMULI = 14917 if SHD else 10000
#CUE_TIME = 20.0

BATCH_SIZE = 512

RECORD = False

NUM_RECURRENT_NEURONS = 800
NUM_OUTPUT_NEURONS = 32 if SHD else 16


STIMULI_TIMESTEPS = int(np.ceil(MAX_STIMULI_TIME / TIMESTEP_MS))
TRIAL_TIMESTEPS = STIMULI_TIMESTEPS # + CUE_TIMESTEPS

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def write_spike_file(filename, data):
    np.savetxt(filename, np.column_stack(data), fmt=["%f","%d"], 
               delimiter=",", header="Time [ms], Neuron ID")

#----------------------------------------------------------------------------
# Neuron models
#----------------------------------------------------------------------------
recurrent_alif_model = genn_model.create_custom_neuron_class(
    "recurrent_alif",
    param_names=["TauM", "TauAdap", "Vthresh", "TauRefrac", "Beta"],
    var_name_types=[("V", "scalar"), ("A", "scalar"), ("RefracTime", "scalar")],
    derived_params=[("Alpha", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))()),
                    ("Rho", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[1]))())],

    sim_code="""
    $(V) = ($(Alpha) * $(V)) + $(Isyn);
    $(A) *= $(Rho);
    if ($(RefracTime) > 0.0) {
      $(RefracTime) -= DT;
    }
    """,
    reset_code="""
    $(RefracTime) = $(TauRefrac);
    $(V) -= $(Vthresh);
    $(A) += 1.0;
    """,
    threshold_condition_code="""
    $(RefracTime) <= 0.0 && $(V) >= ($(Vthresh) + ($(Beta) * $(A)))
    """,
    is_auto_refractory_required=False)

output_classification_model = genn_model.create_custom_neuron_class(
    "output_classification",
    param_names=["TauOut"],
    var_name_types=[("Y", "scalar"), ("B", "scalar")],
    derived_params=[("Kappa", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())],

    sim_code="""
    $(Y) = ($(Kappa) * $(Y)) + $(Isyn) + $(B);
    """,
    is_auto_refractory_required=False)

# Create dataset
if SHD:
    dataset = tonic.datasets.SHD(save_to='./data', train=False)
else:
    dataset = tonic.datasets.SMNIST(save_to='./data', train=False)

# Create loader
data_loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)

# Calculate number of input neurons from sensor size
# **NOTE** we add one as we use an additional neuron to 
num_input_neurons = np.product(dataset.sensor_size) 

# Calculate number of valid outputs from classes
num_outputs = len(dataset.classes)

# ----------------------------------------------------------------------------
# Neuron initialisation
# ----------------------------------------------------------------------------
# Recurrent population
recurrent_params = {"TauM": 20.0, "TauAdap": 2000.0, "Vthresh": 0.6, "TauRefrac": 5.0, "Beta": 0.0174}
recurrent_vars = {"V": 0.0, "A": 0.0, "RefracTime": 0.0}

# Output population
output_params = {"TauOut": 20.0}
output_vars = {"Y": 0.0}
output_vars["B"] = np.load("b_output_trained.npy")

# ----------------------------------------------------------------------------
# Synapse initialisation
# ----------------------------------------------------------------------------
# Input->recurrent synapse parameters
input_recurrent_vars = {"g": np.load("g_input_recurrent_trained.npy")}

# Recurrent->recurrent synapse parameters
recurrent_recurrent_vars = {"g": np.load("g_recurrent_recurrent_trained.npy")}

# Recurrent->output synapse parameters
recurrent_output_vars = {"g": np.load("g_recurrent_output_trained.npy")}

# ----------------------------------------------------------------------------
# Model description
# ----------------------------------------------------------------------------
model = genn_model.GeNNModel("float", "tonic_classifier_evaluate")
model.dT = TIMESTEP_MS
model.timing_enabled = TIMING_ENABLED

# Add neuron populations
input = model.add_neuron_population("Input", num_input_neurons, "SpikeSourceArray",
                                    {}, {"startSpike": None, "endSpike": None})
recurrent = model.add_neuron_population("Recurrent", NUM_RECURRENT_NEURONS, recurrent_alif_model,
                                        recurrent_params, recurrent_vars)
output = model.add_neuron_population("Output", NUM_OUTPUT_NEURONS, output_classification_model,
                                     output_params, output_vars)

# Allocate memory for input spikes and labels
input.set_extra_global_param("spikeTimes", np.zeros(BATCH_SIZE * MAX_SPIKES_PER_STIMULI, dtype=np.float32))

# Turn on recording
input.spike_recording_enabled = True
recurrent.spike_recording_enabled = True

# Add synapse populations
input_recurrent = model.add_synapse_population(
    "InputRecurrent", "DENSE_INDIVIDUALG", NO_DELAY,
    input, recurrent,
    "StaticPulse", {}, input_recurrent_vars, {}, {},
    "DeltaCurr", {}, {})

recurrent_recurrent = model.add_synapse_population(
    "RecurrentRecurrent", "DENSE_INDIVIDUALG", NO_DELAY,
    recurrent, recurrent,
    "StaticPulse", {}, recurrent_recurrent_vars, {}, {},
    "DeltaCurr", {}, {})

recurrent_output = model.add_synapse_population(
    "RecurrentOutput", "DENSE_INDIVIDUALG", NO_DELAY,
    recurrent, output,
    "StaticPulse", {}, recurrent_output_vars, {}, {},
    "DeltaCurr", {}, {})

# Build and load model
model.build()
model.load(num_recording_timesteps=TRIAL_TIMESTEPS * BATCH_SIZE)

# Get views
input_neuron_start_spike = input.vars["startSpike"].view
input_neuron_end_spike = input.vars["endSpike"].view
input_spike_times_view = input.extra_global_params["spikeTimes"].view

output_y_view = output.vars["Y"].view

# Open file
performance_file = open("performance_evaluate.csv", "w")
performance_csv = csv.writer(performance_file, delimiter=",")
performance_csv.writerow(("Batch", "Num trials", "Number correct"))

# Get new data iterator for new epoch
data_iter = iter(data_loader)
for batch_idx, batch_data in enumerate(data_iter):
    print("\tBatch %u" % batch_idx)
    batch_start_time = perf_counter()

    # Reset time
    model.timestep = 0
    model.t = 0.0

    # Get duration of each stimuli in batch
    batch_events, batch_labels = zip(*batch_data)

    # Concatenate together all spike times, offsetting so each stimuli ends at the start of the cue time of each trial
    spike_times = np.concatenate([(i * 1000.0 * MAX_STIMULI_TIME) + e[:,0]
                                  for i, e in enumerate(batch_events)])

    spike_ids = np.concatenate([e[:,1] for e in batch_events]).astype(int)

    # Indirectly sort spikes, first by neuron id and then by time
    spike_order = np.lexsort((spike_times, spike_ids))

    # Use this to re-order spike ids
    spike_ids = spike_ids[spike_order]

    # Check that spike times will fit in view, copy them and push them
    assert len(spike_times) <= len(input_spike_times_view)
    input_spike_times_view[0:len(spike_ids)] = spike_times[spike_order] / 1000.0
    input.push_extra_global_param_to_device("spikeTimes")

    # Calculate start and end spike indices
    input_neuron_end_spike[:] = np.cumsum(np.bincount(spike_ids, minlength=num_input_neurons))
    input_neuron_start_spike[:] = np.concatenate(([0], input_neuron_end_spike[:-1]))
    input.push_var_to_device("startSpike")
    input.push_var_to_device("endSpike")

    # Loop through trials in batch
    num_correct = 0
    for trial, label in enumerate(batch_labels):
        # Loop through timesteps in each trial
        classification_output = np.zeros(num_outputs)
        for i in range(TRIAL_TIMESTEPS):
            model.step_time()

            # Pull Pi from device and add to total
            output.pull_var_from_device("Y")
            classification_output += output_y_view[:num_outputs]

        # If maximum output matches label, increment counter
        if np.argmax(classification_output) == label:
            num_correct += 1

    print("\t\t%u / %u correct" % (num_correct, len(batch_events)))
    performance_csv.writerow((batch_idx, len(batch_events), num_correct))
    performance_file.flush()

    if RECORD:
        # Download recording data
        model.pull_recording_buffers_from_device()

        # Write spikes
        write_spike_file("input_spikes_%u_%u.csv" % (epoch, batch_idx), input.spike_recording_data)
        write_spike_file("recurrent_spikes_%u_%u.csv" % (epoch, batch_idx), recurrent.spike_recording_data)

    batch_end_time = perf_counter()
    print("\t\tTime:%f ms" % ((batch_end_time - batch_start_time) * 1000.0))
performance_file.close()
if TIMING_ENABLED:
    print("Init: %f" % model.init_time)
    print("Init sparse: %f" % model.init_sparse_time)
    print("Neuron update: %f" % model.neuron_update_time)
    print("Presynaptic update: %f" % model.presynaptic_update_time)
    print("Synapse dynamics: %f" % model.synapse_dynamics_time)
    print("Gradient learning custom update: %f" % model.get_custom_update_time("GradientLearn"))
    print("Gradient learning custom update transpose: %f" % model.get_custom_update_transpose_time("GradientLearn"))
   
