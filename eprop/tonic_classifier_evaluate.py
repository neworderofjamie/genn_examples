import csv
import numpy as np
import tonic
import matplotlib.pyplot as plt

from time import perf_counter
from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

from dataloader import DataLoader
# Eprop imports
#import eprop

TIMESTEP_MS = 1.0
TIMING_ENABLED = True
SHD = True

MAX_STIMULI_TIME = 1369.140625 if SHD else 1568.0
MAX_SPIKES_PER_STIMULI = 14917 if SHD else 10000
#CUE_TIME = 20.0

BATCH_SIZE = 64

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
    var_name_types=[("Y", "scalar"), ("YSum", "scalar"), ("B", "scalar", VarAccess_READ_ONLY)],
    derived_params=[("Kappa", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())],

    sim_code="""
    // Reset YSum at start of each batch
    if($(t) == 0.0) {
        $(YSum) = 0.0;
    }
    
    $(Y) = ($(Kappa) * $(Y)) + $(Isyn) + $(B);
    
    $(YSum) += $(Y);
    """,
    is_auto_refractory_required=False)

# Create dataset
if SHD:
    dataset = tonic.datasets.SHD(save_to='./data', train=False)
else:
    dataset = tonic.datasets.SMNIST(save_to='./data', train=False)

# Create loader
data_loader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)

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
output_vars = {"Y": 0.0, "YSum": 0.0, "B": np.load("b_output_trained.npy")}

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
model.batch_size = BATCH_SIZE

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
model.load(num_recording_timesteps=TRIAL_TIMESTEPS)

# Get views
input_neuron_start_spike = input.vars["startSpike"].view
input_neuron_end_spike = input.vars["endSpike"].view
input_spike_times_view = input.extra_global_params["spikeTimes"].view

output_y_sum_view = output.vars["YSum"].view

# Open file
performance_file = open("performance_evaluate.csv", "w")
performance_csv = csv.writer(performance_file, delimiter=",")
performance_csv.writerow(("Batch", "Num trials", "Number correct"))

# Get new data iterator for new epoch
data_iter = iter(data_loader)
total_num = 0;
total_num_correct = 0
for batch_idx, batch_data in enumerate(data_iter):
    print("Batch %u" % batch_idx)
    batch_start_time = perf_counter()

    # Reset time
    model.timestep = 0
    model.t = 0.0

    # Unzip batch data into events and labels
    batch_events, batch_labels = zip(*batch_data)
    
    # Sort events first by neuron id and then by time and use to order spike times
    batch_spike_times = [e[np.lexsort((e[:,1], e[:,1])),0]
                         for e in batch_events]
    
    # Convert events ids to integer
    batch_id_int = [e[:,1].astype(int) for e in batch_events]
    
    # Calculate starting index of spikes in each stimuli across the batch
    # **NOTE** we calculate extra end value for use if padding is required
    cum_spikes_per_stimuli = np.concatenate(([0], np.cumsum([len(e) for e in batch_id_int])))
    
    # Add this cumulative sum onto the cumulative sum of spikes per neuron
    # **NOTE** zip will stop before extra cum_spikes_per_stimuli value
    end_spikes = np.vstack([c + np.cumsum(np.bincount(e, minlength=num_input_neurons)) 
                            for e, c in zip(batch_id_int, cum_spikes_per_stimuli)])
    
    start_spikes = np.empty((len(batch_events), num_input_neurons), dtype=int)
    start_spikes[:,0] = cum_spikes_per_stimuli[:-1]
    start_spikes[:,1:] = end_spikes[:,:-1]
    
    if len(batch_events) != BATCH_SIZE:
        spike_padding = np.ones((BATCH_SIZE - len(batch_events), num_input_neurons), dtype=int) * cum_spikes_per_stimuli[-1]
        end_spikes = np.vstack((end_spikes, spike_padding))
        start_spikes = np.vstack((start_spikes, spike_padding))
    
    # Concatenate together all spike times
    spike_times = np.concatenate(batch_spike_times)

    # Check that spike times will fit in view, copy them and push them
    assert len(spike_times) <= len(input_spike_times_view)
    input_spike_times_view[0:len(spike_times)] = spike_times / 1000.0
    input.push_extra_global_param_to_device("spikeTimes")

    # Calculate start and end spike indices
    input_neuron_end_spike[:] = end_spikes
    input_neuron_start_spike[:] = start_spikes
    input.push_var_to_device("startSpike")
    input.push_var_to_device("endSpike")

    # Loop through timesteps
    num_correct = 0
    classification_output = np.zeros((len(batch_events), num_outputs))
    for i in range(TRIAL_TIMESTEPS):
        model.step_time()

    # Pull sum of outputs from device
    output.pull_var_from_device("YSum")
    
    # If maximum output matches label, increment counter
    num_correct += np.sum(np.argmax(output_y_sum_view[:len(batch_events),:], axis=1) == batch_labels)

    print("\t%u / %u correct = %f %%" % (num_correct, len(batch_events), 100.0 * num_correct / len(batch_events)))
    total_num += len(batch_events)
    total_num_correct += num_correct
    
    performance_csv.writerow((batch_idx, len(batch_events), num_correct))
    performance_file.flush()

    if RECORD:
        # Download recording data
        model.pull_recording_buffers_from_device()

        # Write spikes
        for i, s in enumerate(input.spike_recording_data):
            write_spike_file("input_spikes_%u_%u.csv" % (batch_idx, i), s)
        for i, s in enumerate(recurrent.spike_recording_data):
            write_spike_file("recurrent_spikes_%u_%u.csv" % (batch_idx, i), s)
    
    batch_end_time = perf_counter()
    print("\t\tTime:%f ms" % ((batch_end_time - batch_start_time) * 1000.0))

print("%u / %u correct = %f %%" % (total_num_correct, total_num, 100.0 * total_num_correct / total_num))

performance_file.close()
if TIMING_ENABLED:
    print("Init: %f" % model.init_time)
    print("Init sparse: %f" % model.init_sparse_time)
    print("Neuron update: %f" % model.neuron_update_time)
    print("Presynaptic update: %f" % model.presynaptic_update_time)
   
