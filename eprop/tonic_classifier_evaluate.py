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

from dataloader import DataLoader
from tonic_preprocessor import preprocess_data
# Eprop imports
#import eprop

# Build command line parse
parser = ArgumentParser(description="Train eProp classifier")
parser.add_argument("--dt", type=float, default=1.0)
parser.add_argument("--timing", action="store_true")
parser.add_argument("--record", action="store_true")
parser.add_argument("--warmup", action="store_true")
parser.add_argument("--backend")
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--num-recurrent-alif", type=int, default=256)
parser.add_argument("--dataset", choices=["smnist", "shd"])
parser.add_argument("--trained-epoch", type=int, default=49)
parser.add_argument("--suffix", default="")
args = parser.parse_args()

MAX_STIMULI_TIMES = {"smnist": 1568.0, "shd": 1369.140625}
MAX_SPIKES_PER_STIMULI = {"smnist": 10088, "shd": 14917}

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


recurrent_reset_model = genn_model.create_custom_custom_update_class(
    "recurrent_reset",
    var_refs=[("V", "scalar"), ("A", "scalar"), ("RefracTime", "scalar")],
    update_code="""
    $(V) = 0.0;
    $(A) = 0.0;
    $(RefracTime) = 0.0;
    """)

output_reset_model = genn_model.create_custom_custom_update_class(
    "output_reset",
    var_refs=[("Y", "scalar"), ("YSum", "scalar")],
    update_code="""
    $(Y) = 0.0;
    $(YSum) = 0.0;
    """)
    
# Create dataset
if args.dataset == "shd":
    dataset = tonic.datasets.SHD(save_to='./data', train=False)
elif args.dataset == "smnist":
    dataset = tonic.datasets.SMNIST(save_to='./data', train=False)
else:
    raise RuntimeError("Unknown dataset '%s'" % args.dataset)

# Determine output directory name and create if it doesn't exist
name_suffix = "%u%s" % (args.num_recurrent_alif, args.suffix)
output_directory = "%s_%s" % (args.dataset, name_suffix)
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

# Create loader
data_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)

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
output_params = {"TauOut": 20.0}
output_vars = {"Y": 0.0, "YSum": 0.0, "B": np.load("%s_%s%s/b_%s_output_%s_%u.npy" % (args.dataset, name_suffix, args.suffix, args.dataset, name_suffix, args.trained_epoch))}

# ----------------------------------------------------------------------------
# Synapse initialisation
# ----------------------------------------------------------------------------
# Input->recurrent synapse parameters
input_recurrent_vars = {"g": np.load(os.path.join(output_directory, "g_input_recurrent_%u.npy" % args.trained_epoch))}

# Recurrent->recurrent synapse parameters
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

output_y_sum_view = output.vars["YSum"].view

# Open file
performance_file = open(os.path.join(output_directory, "performance_evaluate_%u.csv" % args.trained_epoch), "w")
performance_csv = csv.writer(performance_file, delimiter=",")
performance_csv.writerow(("Batch", "Num trials", "Number correct"))

# Read batches of data using data loader
start_processing_time = perf_counter()
batch_data = preprocess_data(data_loader, args.batch_size, num_input_neurons)
end_process_time = perf_counter()
print("Data processing time:%f ms" % ((end_process_time - start_processing_time) * 1000.0))

# If we should warmup the state of the network
if args.warmup:
    for batch_idx, (start_spikes, end_spikes, spike_times, batch_labels) in enumerate(batch_data):
        # Reset time
        model.timestep = 0
        model.t = 0.0
        
        # Check that spike times will fit in view, copy them and push them
        if len(spike_times) > len(input_spike_times_view):
            print(len(spike_times), len(input_spike_times_view))
        assert len(spike_times) <= len(input_spike_times_view)
        input_spike_times_view[0:len(spike_times)] = spike_times / 1000.0
        input.push_extra_global_param_to_device("spikeTimes")

        # Calculate start and end spike indices
        input_neuron_end_spike[:] = end_spikes
        input_neuron_start_spike[:] = start_spikes
        input.push_var_to_device("startSpike")
        input.push_var_to_device("endSpike")

        # Loop through timesteps
        for i in range(stimuli_timesteps):
            model.step_time()

total_num = 0;
total_num_correct = 0
start_time = perf_counter()
for batch_idx, (start_spikes, end_spikes, spike_times, batch_labels) in enumerate(batch_data):
    print("Batch %u" % batch_idx)
    batch_start_time = perf_counter()

    # Reset time
    model.timestep = 0
    model.t = 0.0

    # Check that spike times will fit in view, copy them and push them
    if len(spike_times) > len(input_spike_times_view):
        print(len(spike_times), len(input_spike_times_view))
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
    classification_output = np.zeros((len(batch_labels), num_outputs))
    for i in range(stimuli_timesteps):
        model.step_time()

    # Pull sum of outputs from device
    output.pull_var_from_device("YSum")
            
    # If maximum output matches label, increment counter
    if args.batch_size == 1:
        num_correct += np.sum(np.argmax(output_y_sum_view) == batch_labels)
    else:
        num_correct += np.sum(np.argmax(output_y_sum_view[:len(batch_labels),:], axis=1) == batch_labels)

    print("\t%u / %u correct = %f %%" % (num_correct, len(batch_labels), 100.0 * num_correct / len(batch_labels)))
    total_num += len(batch_labels)
    total_num_correct += num_correct
    
    performance_csv.writerow((batch_idx, len(batch_labels), num_correct))
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
   
