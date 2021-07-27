import csv
import numpy as np
import os
import tonic
import matplotlib.pyplot as plt
import random

from argparse import ArgumentParser
from time import perf_counter
from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

from tonic_classifier_parser import parse_arguments
import dataloader

# Eprop imports
#import eprop

# Build command line parse
parser = ArgumentParser(add_help=False)
parser.add_argument("--dt", type=float, default=1.0)
parser.add_argument("--timing", action="store_true")
parser.add_argument("--record", action="store_true")
parser.add_argument("--warmup", action="store_true")
parser.add_argument("--backend")
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--trained-epoch", type=int, default=49)

name_suffix, output_directory, args = parse_arguments(parser, description="Evaluate eProp classifier")
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

MAX_STIMULI_TIMES = {"smnist": 1568.0 * 2.0, "shd": 1369.140625 * 2.0}
MAX_SPIKES_PER_STIMULI = {"smnist": 10088 * 2, "shd": 14917 * 2}

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
    param_names=["TauOut", "TrialTime"],
    var_name_types=[("Y", "scalar"), ("YSum", "scalar"), ("B", "scalar", VarAccess_READ_ONLY)],
    derived_params=[("Kappa", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())],

    sim_code="""
    // Reset YSum at start of each batch
    if($(t) == 0.0) {
        $(YSum) = 0.0;
    }

    $(Y) = ($(Kappa) * $(Y)) + $(Isyn) + $(B);
    
    if($(t) > ($(TrialTime) * 0.5)) {
        $(YSum) += $(Y);
    }
    """,
    is_auto_refractory_required=False)

    
# Create dataset
if args.dataset == "shd":
    dataset = tonic.datasets.SHD(save_to='./data', train=False)
    train_dataset = tonic.datasets.SHD(save_to='./data', train=True)
elif args.dataset == "smnist":
    dataset = tonic.datasets.SMNIST(save_to='./data', train=False)
    train_dataset = tonic.datasets.SMNIST(save_to='./data', train=True)
else:
    raise RuntimeError("Unknown dataset '%s'" % args.dataset)

# Create loader
start_processing_time = perf_counter()
data_loader = dataloader.DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
end_process_time = perf_counter()
print("Data processing time:%f ms" % ((end_process_time - start_processing_time) * 1000.0))

# Calculate number of input neurons from sensor size
# **NOTE** we add one as we use an additional neuron to 
num_input_neurons = np.product(dataset.sensor_size) 

# Calculate number of valid outputs from classes
num_outputs = len(dataset.classes)

# Round up to power-of-two
# **NOTE** not really necessary for evaluation - could slice up weights
num_output_neurons = int(2**(np.ceil(np.log2(num_outputs))))

# ----------------------------------------------------------------------------
# Neuron initialisation
# ----------------------------------------------------------------------------
# Recurrent population
recurrent_params = {"TauM": 20.0, "TauAdap": 2000.0, "Vthresh": 0.6, "TauRefrac": 5.0, "Beta": 0.0174}
recurrent_vars = {"V": 0.0, "A": 0.0, "RefracTime": 0.0}

# Output population
output_params = {"TauOut": 20.0, "TrialTime": MAX_STIMULI_TIMES[args.dataset]}
output_vars = {"Y": 0.0, "YSum": 0.0, "B": np.load(os.path.join(output_directory, "b_output_%u.npy" % args.trained_epoch))}

# ----------------------------------------------------------------------------
# Synapse initialisation
# ----------------------------------------------------------------------------
# Input->recurrent synapse parameters
input_recurrent_vars = {"g": np.load(os.path.join(output_directory, "g_input_recurrent_%u.npy" % args.trained_epoch))}

# Recurrent->recurrent synapse parameters
if not args.feedforward:
    recurrent_recurrent_vars = {"g": np.load(os.path.join(output_directory, "g_recurrent_recurrent_%u.npy" % args.trained_epoch))}

# Recurrent->output synapse parameters
recurrent_output_vars = {"g": np.load(os.path.join(output_directory, "g_recurrent_output_%u.npy" % args.trained_epoch))}

# ----------------------------------------------------------------------------
# Model description
# ----------------------------------------------------------------------------
model = genn_model.GeNNModel("float", "%s_tonic_classifier_evaluate_%s" % (args.dataset, name_suffix), backend=args.backend)
model.dT = args.dt
model.timing_enabled = args.timing
model.batch_size = args.batch_size

# Add neuron populations
input = model.add_neuron_population("Input", num_input_neurons, "SpikeSourceArray",
                                    {}, {"startSpike": None, "endSpike": None})
recurrent = model.add_neuron_population("Recurrent", args.num_recurrent_alif, recurrent_alif_model,
                                        recurrent_params, recurrent_vars)
output = model.add_neuron_population("Output", num_output_neurons, output_classification_model,
                                     output_params, output_vars)

# Allocate memory for input spikes and labels
input.set_extra_global_param("spikeTimes", np.zeros(args.batch_size * MAX_SPIKES_PER_STIMULI[args.dataset], dtype=np.float32))

# Turn on recording
input.spike_recording_enabled = args.record
recurrent.spike_recording_enabled = args.record

# Add synapse populations
input_recurrent = model.add_synapse_population(
    "InputRecurrent", "DENSE_INDIVIDUALG", NO_DELAY,
    input, recurrent,
    "StaticPulse", {}, input_recurrent_vars, {}, {},
    "DeltaCurr", {}, {})

if not args.feedforward:
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
stimuli_timesteps = int(np.ceil(MAX_STIMULI_TIMES[args.dataset] / args.dt))
model.build()
model.load(num_recording_timesteps=stimuli_timesteps)

# Get views
input_neuron_start_spike = input.vars["startSpike"].view
input_neuron_end_spike = input.vars["endSpike"].view
input_spike_times_view = input.extra_global_params["spikeTimes"].view

recurrent_a_view = recurrent.vars["A"].view
recurrent_v_view = recurrent.vars["V"].view
output_y_sum_view = output.vars["YSum"].view

# Open file
performance_file = open(os.path.join(output_directory, "performance_evaluate_%u.csv" % args.trained_epoch), "w")
performance_csv = csv.writer(performance_file, delimiter=",")
performance_csv.writerow(("Batch", "Num trials", "Number correct"))

# Pull arbitrary pre-processed stimuli from data loader
warmup_events = data_loader._preprocess(train_dataset[0][0])

# If we should warmup the state of the network
if args.warmup:
    assert False

    # Loop through batches of (pre-processed) data
    data_iter = iter(data_loader)
    for events, _ in data_iter:
        # Add max stimuli time to events then concatenate with warmup
        offset_events = [dataloader.PreprocessedEvents(e.end_spikes, e.spike_times + (0.5 * MAX_STIMULI_TIMES[args.dataset]))
                         for e in events]
        events_with_warmup = [dataloader.concatenate_events([warmup_events, e]) for e in offset_events]
        
        # Transform data into batch
        batched_data = dataloader.batch_events(events_with_warmup, args.batch_size)

        # Reset time
        model.timestep = 0
        model.t = 0.0
        
        # Check that spike times will fit in view, copy them and push them
        assert len(batched_data.spike_times) <= len(input_spike_times_view)
        input_spike_times_view[0:len(batched_data.spike_times)] = batched_data.spike_times
        input.push_extra_global_param_to_device("spikeTimes")

        # Calculate start and end spike indices
        input_neuron_end_spike[:] = batched_data.end_spikes
        input_neuron_start_spike[:] = dataloader.get_start_spikes(batched_data.end_spikes)
        input.push_var_to_device("startSpike")
        input.push_var_to_device("endSpike")

        # Loop through timesteps
        for i in range(stimuli_timesteps):
            model.step_time()

total_num = 0;
total_num_correct = 0
start_time = perf_counter()
# Loop through batches of (pre-processed) data
data_iter = iter(data_loader)
for batch_idx, (events, labels) in enumerate(data_iter):
    print("Batch %u" % batch_idx)
    batch_start_time = perf_counter()

     # Add max stimuli time to events then concatenate with warmup
    offset_events = [dataloader.PreprocessedEvents(e.end_spikes, e.spike_times + (0.5 * MAX_STIMULI_TIMES[args.dataset]))
                     for e in events]
    events_with_warmup = [dataloader.concatenate_events([warmup_events, e]) for e in offset_events]
    
    # Transform data into batch
    batched_data = dataloader.batch_events(events_with_warmup, args.batch_size)

    # Reset time
    model.timestep = 0
    model.t = 0.0

    # Check that spike times will fit in view, copy them and push them
    assert len(batched_data.spike_times) <= len(input_spike_times_view)
    input_spike_times_view[0:len(batched_data.spike_times)] = batched_data.spike_times
    input.push_extra_global_param_to_device("spikeTimes")

    # Calculate start and end spike indices
    input_neuron_end_spike[:] = batched_data.end_spikes
    input_neuron_start_spike[:] = dataloader.get_start_spikes(batched_data.end_spikes)
    input.push_var_to_device("startSpike")
    input.push_var_to_device("endSpike")
    
    # Zero adaptation
    recurrent_a_view[:] = 0.0
    recurrent_v_view[:] = 0.0
    recurrent.push_var_to_device("A")
    recurrent.push_var_to_device("V")
    
    # Loop through timesteps
    num_correct = 0
    classification_output = np.zeros((len(labels), num_outputs))
    for i in range(stimuli_timesteps):
        model.step_time()

    # Pull sum of outputs from device
    output.pull_var_from_device("YSum")
            
    # If maximum output matches label, increment counter
    if args.batch_size == 1:
        num_correct += np.sum(np.argmax(output_y_sum_view) == labels)
    else:
        num_correct += np.sum(np.argmax(output_y_sum_view[:len(labels),:], axis=1) == labels)

    print("\t%u / %u correct = %f %%" % (num_correct, len(labels), 100.0 * num_correct / len(labels)))
    total_num += len(labels)
    total_num_correct += num_correct
    
    performance_csv.writerow((batch_idx, len(labels), num_correct))
    performance_file.flush()

    if args.record:
        # Download recording data
        model.pull_recording_buffers_from_device()

        # Write spikes
        for i, s in enumerate(input.spike_recording_data):
            write_spike_file(os.path.join(output_directory, "input_spikes_%u_%u.csv" % (batch_idx, i)), s)
        for i, s in enumerate(recurrent.spike_recording_data):
            write_spike_file(os.path.join(output_directory, "recurrent_spikes_%u_%u.csv" % (batch_idx, i)), s)
    
    batch_end_time = perf_counter()
    print("\t\tTime:%f ms" % ((batch_end_time - batch_start_time) * 1000.0))

end_time = perf_counter()
print("%u / %u correct = %f %%" % (total_num_correct, total_num, 100.0 * total_num_correct / total_num))
print("Time:%f ms" % ((end_time - start_time) * 1000.0))

performance_file.close()
if args.timing:
    print("Init: %f" % model.init_time)
    print("Init sparse: %f" % model.init_sparse_time)
    print("Neuron update: %f" % model.neuron_update_time)
    print("Presynaptic update: %f" % model.presynaptic_update_time)
   
