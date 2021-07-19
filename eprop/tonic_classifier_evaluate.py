import csv
import numpy as np
import tonic
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from time import perf_counter
from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY

from dataloader import DataLoader
# Eprop imports
#import eprop

# Build command line parse
parser = ArgumentParser(description="Train eProp classifier")
parser.add_argument("--dt", type=float, default=1.0)
parser.add_argument("--timing", action="store_true")
parser.add_argument("--record", action="store_true")
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--num-recurrent-alif", type=int, default=256)
parser.add_argument("--dataset", choices=["smnist", "shd"])
args = parser.parse_args()

MAX_STIMULI_TIMES = {"smnist": 1568.0, "shd": 1369.140625}
MAX_SPIKES_PER_STIMULI = {"smnist": 10000, "shd": 14917}

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
if args.dataset == "shd":
    dataset = tonic.datasets.SHD(save_to='./data', train=False)
elif args.dataset == "smnist":
    dataset = tonic.datasets.SMNIST(save_to='./data', train=False)
else:
    raise RuntimeError("Unknown dataset '%s'" % args.dataset)

# Build file suffix
name_suffix = "%u" % (args.num_recurrent_alif)

# Create loader
data_loader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size)

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
output_vars = {"Y": 0.0, "YSum": 0.0, "B": np.load("b_%s_output_%s.npy" % (args.dataset, name_suffix))}

# ----------------------------------------------------------------------------
# Synapse initialisation
# ----------------------------------------------------------------------------
# Input->recurrent synapse parameters
input_recurrent_vars = {"g": np.load("g_%s_input_recurrent_%s.npy" % (args.dataset, name_suffix))}

# Recurrent->recurrent synapse parameters
recurrent_recurrent_vars = {"g": np.load("g_%s_recurrent_recurrent_%s.npy" % (args.dataset, name_suffix))}

# Recurrent->output synapse parameters
recurrent_output_vars = {"g": np.load("g_%s_recurrent_output_%s.npy" % (args.dataset, name_suffix))}

# ----------------------------------------------------------------------------
# Model description
# ----------------------------------------------------------------------------
model = genn_model.GeNNModel("float", "%s_tonic_classifier_evaluate_%s" % (args.dataset, name_suffix))
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
performance_file = open("%s_performance_evaluate_%s.csv" % (args.dataset, name_suffix), "w")
performance_csv = csv.writer(performance_file, delimiter=",")
performance_csv.writerow(("Batch", "Num trials", "Number correct"))

# Read batches of data using data loader
start_processing_time = perf_counter()
batch_data = []
data_iter = iter(data_loader)
for d in data_iter:
    # Unzip batch data into events and labels
    if args.batch_size == 1:
        batch_events = [d[0]]
        batch_labels = [d[1]]
    else:
        batch_events, batch_labels = zip(*d)
    
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
    
    # Build start spikes array
    start_spikes = np.empty((len(batch_events), num_input_neurons), dtype=int)
    start_spikes[:,0] = cum_spikes_per_stimuli[:-1]
    start_spikes[:,1:] = end_spikes[:,:-1]
    
    # If this isn't a full batch
    if len(batch_events) != args.batch_size:
        spike_padding = np.ones((args.batch_size - len(batch_events), num_input_neurons), dtype=int) * cum_spikes_per_stimuli[-1]
        end_spikes = np.vstack((end_spikes, spike_padding))
        start_spikes = np.vstack((start_spikes, spike_padding))
    
    # Concatenate together all spike times
    spike_times = np.concatenate(batch_spike_times)
    
    # Add tuple of pre-processed data to list
    batch_data.append((start_spikes, end_spikes, spike_times, batch_labels))

end_process_time = perf_counter()
print("Data processing time:%f ms" % ((end_process_time - start_processing_time) * 1000.0))

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
            write_spike_file("%s_input_spikes_%s_%u_%u.csv" % (args.dataset, name_suffix, batch_idx, i), s)
        for i, s in enumerate(recurrent.spike_recording_data):
            write_spike_file("%s_recurrent_spikes_%s_%u_%u.csv" % (args.dataset, name_suffix, batch_idx, i), s)
    
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
   
