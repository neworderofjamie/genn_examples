import csv
import numpy as np
import os
import tonic
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from time import perf_counter
from random import shuffle
from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY
from pygenn.genn_wrapper.CUDABackend import DeviceSelect_MANUAL


from tonic_classifier_parser import parse_arguments
import dataloader

# Eprop imports
import eprop

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999

WEIGHT_0 = 1.0

# Build command line parse
parser = ArgumentParser(add_help=False)
parser.add_argument("--timing", action="store_true")
parser.add_argument("--record", action="store_true")
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--num-epochs", type=int, default=50)
parser.add_argument("--resume-epoch", type=int, default=None)
parser.add_argument("--cuda-visible-devices", action="store_true")
parser.add_argument("--no-download-dataset", action="store_true")
parser.add_argument("--use-nccl", action="store_true")
parser.add_argument("--hold-back-validate", type=int, default=None)

name_suffix, output_directory, args = parse_arguments(parser, description="Train eProp classifier")

# Seed RNG, leaving random to match GeNN behaviour if seed is zero
np.random.seed(None if args.seed == 0 else args.seed)

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def write_spike_file(filename, data):
    np.savetxt(filename, np.column_stack(data), fmt=["%f","%d"], 
               delimiter=",", header="Time [ms], Neuron ID")

def update_adam(learning_rate, adam_step, optimiser_custom_updates):
    first_moment_scale = 1.0 / (1.0 - (ADAM_BETA1 ** adam_step))
    second_moment_scale = 1.0 / (1.0 - (ADAM_BETA2 ** adam_step))

    # Loop through optimisers and set
    for o in optimiser_custom_updates:
        o.extra_global_params["alpha"].view[:] = learning_rate
        o.extra_global_params["firstMomentScale"].view[:] = first_moment_scale
        o.extra_global_params["secondMomentScale"].view[:] = second_moment_scale



# If we're using NCCL
dataset_slice_end = None if args.hold_back_validate is None else -args.hold_back_validate
if args.use_nccl:
    from mpi4py import MPI

    # Get communicator
    comm = MPI.COMM_WORLD

    # Get our rank and number of ranks
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    print("Rank %u/%u" % (rank, num_ranks))

    # Distribute desired total batch size amongst ranks
    assert (args.batch_size % num_ranks) == 0
    batch_size = args.batch_size // num_ranks

    # Slice dataset between ranks
    dataset_slice = slice(rank, dataset_slice_end, num_ranks)
else:
    batch_size = args.batch_size
    dataset_slice = slice(0, dataset_slice_end)

# Create output directory if it doesn't exist (only on first rank if NCCL)
first_rank = (not args.use_nccl or rank == 0)
if first_rank and not os.path.exists(output_directory):
    os.mkdir(output_directory)

# Create dataset
sensor_size = None
polarity = False
if args.dataset == "shd":
    dataset = tonic.datasets.SHD(save_to='./data', train=True, download=not args.no_download_dataset)
    sensor_size = dataset.sensor_size
elif args.dataset == "smnist":
    dataset = tonic.datasets.SMNIST(save_to='./data', train=True, duplicate=False, num_neurons=79, 
                                    download=not args.no_download_dataset)
    sensor_size = dataset.sensor_size
elif args.dataset == "dvs_gesture":
    transform = tonic.transforms.Compose([
        tonic.transforms.Downsample(spatial_factor=0.25)])
    dataset = tonic.datasets.DVSGesture(save_to='./data', train=True, transform=transform, 
                                        download=not args.no_download_dataset)
    sensor_size = (32, 32)
    polarity = True
else:
    raise RuntimeError("Unknown dataset '%s'" % args.dataset)

# Create loader
start_processing_time = perf_counter()
data_loader = dataloader.DataLoader(dataset, shuffle=True, batch_size=batch_size,
                                    sensor_size=sensor_size, polarity=polarity,
                                    dataset_slice=dataset_slice)
end_process_time = perf_counter()
print("Data processing time:%f ms" % ((end_process_time - start_processing_time) * 1000.0))

# Calculate number of input neurons from sensor size
num_input_neurons = np.product(sensor_size) 
if polarity:
    num_input_neurons *= 2

# Calculate number of valid outputs from classes
num_outputs = len(dataset.classes)

# Round up to power-of-two
num_output_neurons = int(2**(np.ceil(np.log2(num_outputs))))

# Mapping of output neuron numbers to models
output_neuron_models = {16: eprop.output_classification_model_16,
                        32: eprop.output_classification_model_32}

# ----------------------------------------------------------------------------
# Neuron initialisation
# ----------------------------------------------------------------------------
# Recurrent population
recurrent_alif_params = {"TauM": 20.0, "TauAdap": 2000.0, "Vthresh": 0.6, "TauRefrac": 5.0, "Beta": 0.0174}
recurrent_alif_vars = {"V": 0.0, "A": 0.0, "RefracTime": 0.0, "E": 0.0}
recurrent_lif_params = {"TauM": 20.0, "Vthresh": 0.6, "TauRefrac": 5.0}
recurrent_lif_vars = {"V": 0.0, "RefracTime": 0.0, "E": 0.0}

# Output population
output_params = {"TauOut": 20.0, "TrialTime": data_loader.max_stimuli_time}
output_vars = {"Y": 0.0, "Pi": 0.0, "E": 0.0, "DeltaB": 0.0}
if args.resume_epoch is None:
    output_vars["B"] = 0.0
else:
    output_vars["B"] = np.load(os.path.join(output_directory, "b_output_%u.npy" % args.resume_epoch))

# ----------------------------------------------------------------------------
# Synapse initialisation
# ----------------------------------------------------------------------------
# eProp parameters common across all populations
eprop_lif_params = {"TauE": 20.0, "CReg": args.regularizer_strength, "FTarget": 10.0, "TauFAvg": 500.0}
eprop_alif_params = {"TauE": 20.0, "TauA": 2000.0, "CReg": args.regularizer_strength,
                     "FTarget": 10.0, "TauFAvg": 500.0, "Beta": 0.0174}
eprop_pre_vars = {"ZFilter": 0.0}
eprop_post_vars = {"Psi": 0.0, "FAvg": 0.0}

# Input->recurrent synapse parameters
if args.num_recurrent_alif > 0:
    input_recurrent_alif_vars = {"eFiltered": 0.0, "epsilonA": 0.0, "DeltaG": 0.0}
    if args.resume_epoch is None:
        input_recurrent_alif_vars["g"] = genn_model.init_var("Normal", {"mean": 0.0, "sd": WEIGHT_0 / np.sqrt(num_input_neurons)})
    else:
        input_recurrent_alif_vars["g"] = np.load(os.path.join(output_directory, "g_input_recurrent_%u.npy" % args.resume_epoch))
if args.num_recurrent_lif > 0:
    input_recurrent_lif_vars = {"eFiltered": 0.0, "DeltaG": 0.0}
    if args.resume_epoch is None:
        input_recurrent_lif_vars["g"] = genn_model.init_var("Normal", {"mean": 0.0, "sd": WEIGHT_0 / np.sqrt(num_input_neurons)})
    else:
        input_recurrent_lif_vars["g"] = np.load(os.path.join(output_directory, "g_input_recurrent_lif_%u.npy" % args.resume_epoch))

# Recurrent->recurrent synapse parameters
if not args.feedforward:
    if args.num_recurrent_alif > 0:
        recurrent_alif_recurrent_alif_vars = {"eFiltered": 0.0, "epsilonA": 0.0, "DeltaG": 0.0}
        if args.resume_epoch is None:
            recurrent_alif_recurrent_alif_vars["g"] = genn_model.init_var("Normal", {"mean": 0.0, "sd": WEIGHT_0 / np.sqrt(args.num_recurrent_alif)})
        else:
            recurrent_alif_recurrent_alif_vars["g"] = np.load(os.path.join(output_directory, "g_recurrent_recurrent_%u.npy" % args.resume_epoch))
    if args.num_recurrent_lif > 0:
        recurrent_lif_recurrent_lif_vars = {"eFiltered": 0.0, "DeltaG": 0.0}
        if args.resume_epoch is None:
            recurrent_lif_recurrent_lif_vars["g"] = genn_model.init_var("Normal", {"mean": 0.0, "sd": WEIGHT_0 / np.sqrt(args.num_recurrent_lif)})
        else:
            recurrent_lif_recurrent_lif_vars["g"] = np.load(os.path.join(output_directory, "g_recurrent_lif_recurrent_lif_%u.npy" % args.resume_epoch))
    
# Recurrent->output synapse parameters
recurrent_output_params = {"TauE": 20.0}
recurrent_output_pre_vars = {"ZFilter": 0.0}
if args.num_recurrent_alif > 0:
    recurrent_alif_output_vars = {"DeltaG": 0.0}
    if args.resume_epoch is None:
        recurrent_alif_output_vars["g"] = genn_model.init_var("Normal", {"mean": 0.0, "sd": WEIGHT_0 / np.sqrt(args.num_recurrent_alif)})
    else:
        recurrent_alif_output_vars["g"] = np.load(os.path.join(output_directory, "g_recurrent_output_%u.npy" % args.resume_epoch))
if args.num_recurrent_lif > 0:
    recurrent_lif_output_vars = {"DeltaG": 0.0}
    if args.resume_epoch is None:
        recurrent_lif_output_vars["g"] = genn_model.init_var("Normal", {"mean": 0.0, "sd": WEIGHT_0 / np.sqrt(args.num_recurrent_lif)})
    else:
        recurrent_lif_output_vars["g"] = np.load(os.path.join(output_directory, "g_recurrent_lif_output_%u.npy" % args.resume_epoch))

# Optimiser initialisation
adam_params = {"beta1": ADAM_BETA1, "beta2": ADAM_BETA2, "epsilon": 1E-8}
adam_vars = {"m": 0.0, "v": 0.0}

# Batch reduction initialisation
gradient_batch_reduce_vars = {"reducedGradient": 0.0}

# ----------------------------------------------------------------------------
# Model description
# ----------------------------------------------------------------------------
# If we should respect CUDA_VISIBLE_DEVICES, 
# use manual device ID-based device selection (note default is 0)
kwargs = {}
if args.cuda_visible_devices:
    kwargs["selectGPUByDeviceID"] = True
    kwargs["deviceSelectMethod"] = DeviceSelect_MANUAL

# If we should use NCCL, turn on flag to generate NCCL reductions
if args.use_nccl:
    kwargs["enableNCCLReductions"] = True

model = genn_model.GeNNModel("float", "%s_tonic_classifier_%s" % (args.dataset, name_suffix),
                             **kwargs)
model.dT = args.dt
model.timing_enabled = args.timing
model.batch_size = batch_size
model._model.set_seed(args.seed)

# Add neuron populations
input = model.add_neuron_population("Input", num_input_neurons, "SpikeSourceArray",
                                    {}, {"startSpike": None, "endSpike": None})
if args.num_recurrent_alif > 0:
    recurrent_alif = model.add_neuron_population("RecurrentALIF", args.num_recurrent_alif, eprop.recurrent_alif_model,
                                                 recurrent_alif_params, recurrent_alif_vars)
    recurrent_alif.spike_recording_enabled = args.record
if args.num_recurrent_lif > 0:
    recurrent_lif = model.add_neuron_population("RecurrentLIF", args.num_recurrent_lif, eprop.recurrent_lif_model,
                                                recurrent_lif_params, recurrent_lif_vars)
    recurrent_lif.spike_recording_enabled = args.record

output = model.add_neuron_population("Output", num_output_neurons, output_neuron_models[num_output_neurons],
                                     output_params, output_vars)

# Allocate memory for input spikes and labels
input.set_extra_global_param("spikeTimes", np.zeros(batch_size * data_loader.max_spikes_per_stimuli, dtype=np.float32))
output.set_extra_global_param("labels", np.zeros(batch_size, dtype=np.uint8))

# Turn on recording
input.spike_recording_enabled = args.record

# (For now) check that there aren't both LIF and ALIF recurrent neurons
assert not (args.num_recurrent_alif > 0 and args.num_recurrent_lif > 0)

# Add synapse populations
if args.num_recurrent_alif > 0:
    input_recurrent_alif = model.add_synapse_population(
        "InputRecurrentALIF", "DENSE_INDIVIDUALG", NO_DELAY,
        input, recurrent_alif,
        eprop.eprop_alif_model, eprop_alif_params, input_recurrent_alif_vars, eprop_pre_vars, eprop_post_vars,
        "DeltaCurr", {}, {})
    recurrent_alif_output = model.add_synapse_population(
        "RecurrentALIFOutput", "DENSE_INDIVIDUALG", NO_DELAY,
        recurrent_alif, output,
        eprop.output_learning_model, recurrent_output_params, recurrent_alif_output_vars, recurrent_output_pre_vars, {},
        "DeltaCurr", {}, {})
    output_recurrent_alif = model.add_synapse_population(
        "OutputRecurrentALIF", "DENSE_INDIVIDUALG", NO_DELAY,
        output, recurrent_alif,
        eprop.feedback_model, {}, {"g": 0.0}, {}, {},
        "DeltaCurr", {}, {})
    output_recurrent_alif.ps_target_var = "ISynFeedback"

if args.num_recurrent_lif > 0:
    input_recurrent_lif = model.add_synapse_population(
        "InputRecurrentLIF", "DENSE_INDIVIDUALG", NO_DELAY,
        input, recurrent_lif,
        eprop.eprop_lif_model, eprop_lif_params, input_recurrent_lif_vars, eprop_pre_vars, eprop_post_vars,
        "DeltaCurr", {}, {})
    recurrent_lif_output = model.add_synapse_population(
        "RecurrentLIFOutput", "DENSE_INDIVIDUALG", NO_DELAY,
        recurrent_lif, output,
        eprop.output_learning_model, recurrent_output_params, recurrent_lif_output_vars, recurrent_output_pre_vars, {},
        "DeltaCurr", {}, {})
    output_recurrent_lif = model.add_synapse_population(
        "OutputRecurrentLIF", "DENSE_INDIVIDUALG", NO_DELAY,
        output, recurrent_lif,
        eprop.feedback_model, {}, {"g": 0.0}, {}, {},
        "DeltaCurr", {}, {})
    output_recurrent_lif.ps_target_var = "ISynFeedback"

if not args.feedforward:
    if args.num_recurrent_alif > 0:
        recurrent_alif_recurrent_alif = model.add_synapse_population(
            "RecurrentALIFRecurrentALIF", "DENSE_INDIVIDUALG", NO_DELAY,
            recurrent_alif, recurrent_alif,
            eprop.eprop_alif_model, eprop_alif_params, recurrent_alif_recurrent_alif_vars, eprop_pre_vars, eprop_post_vars,
            "DeltaCurr", {}, {})
    if args.num_recurrent_lif > 0:
        recurrent_lif_recurrent_lif = model.add_synapse_population(
            "RecurrentLIFRecurrentLIF", "DENSE_INDIVIDUALG", NO_DELAY,
            recurrent_lif, recurrent_lif,
            eprop.eprop_lif_model, eprop_lif_params, recurrent_lif_recurrent_lif_vars, eprop_pre_vars, eprop_post_vars,
            "DeltaCurr", {}, {})

# Add custom update for calculating initial tranpose weights
if args.num_recurrent_alif > 0:
    model.add_custom_update("recurrent_alif_hidden_transpose", "CalculateTranspose", "Transpose",
                            {}, {}, {"variable": genn_model.create_wu_var_ref(recurrent_alif_output, "g", output_recurrent_alif, "g")})
if args.num_recurrent_lif > 0:
    model.add_custom_update("recurrent_lif_hidden_transpose", "CalculateTranspose", "Transpose",
                            {}, {}, {"variable": genn_model.create_wu_var_ref(recurrent_lif_output, "g", output_recurrent_lif, "g")})

# Add custom updates for reducing gradients across the batch
if args.num_recurrent_alif > 0:
    input_recurrent_alif_reduction_var_refs = {"gradient": genn_model.create_wu_var_ref(input_recurrent_alif, "DeltaG")}
    input_recurrent_alif_reduction = model.add_custom_update("input_recurrent_alif_reduction", "GradientBatchReduce", eprop.gradient_batch_reduce_model, 
                                                             {}, gradient_batch_reduce_vars, input_recurrent_alif_reduction_var_refs)

    recurrent_alif_output_reduction_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_alif_output, "DeltaG")}
    recurrent_alif_output_reduction = model.add_custom_update("recurrent_alif_output_reduction", "GradientBatchReduce", eprop.gradient_batch_reduce_model, 
                                                              {}, gradient_batch_reduce_vars, recurrent_alif_output_reduction_var_refs)
if args.num_recurrent_lif > 0:
    input_recurrent_lif_reduction_var_refs = {"gradient": genn_model.create_wu_var_ref(input_recurrent_lif, "DeltaG")}
    input_recurrent_lif_reduction = model.add_custom_update("input_recurrent_lif_reduction", "GradientBatchReduce", eprop.gradient_batch_reduce_model, 
                                                            {}, gradient_batch_reduce_vars, input_recurrent_lif_reduction_var_refs)
    recurrent_lif_output_reduction_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_lif_output, "DeltaG")}
    recurrent_lif_output_reduction = model.add_custom_update("recurrent_lif_output_reduction", "GradientBatchReduce", eprop.gradient_batch_reduce_model, 
                                                             {}, gradient_batch_reduce_vars, recurrent_lif_output_reduction_var_refs)

if not args.feedforward:
    if args.num_recurrent_alif > 0:
        recurrent_alif_recurrent_alif_reduction_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_alif_recurrent_alif, "DeltaG")}
        recurrent_alif_recurrent_alif_reduction = model.add_custom_update("recurrent_alif_recurrent_alif_reduction", "GradientBatchReduce", eprop.gradient_batch_reduce_model,
                                                                          {}, gradient_batch_reduce_vars, recurrent_alif_recurrent_alif_reduction_var_refs)
    if args.num_recurrent_lif > 0:
        recurrent_lif_recurrent_lif_reduction_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_lif_recurrent_lif, "DeltaG")}
        recurrent_lif_recurrent_lif_reduction = model.add_custom_update("recurrent_lif_recurrent_lif_reduction", "GradientBatchReduce", eprop.gradient_batch_reduce_model,
                                                                          {}, gradient_batch_reduce_vars, recurrent_lif_recurrent_lif_reduction_var_refs)

output_bias_reduction_var_refs = {"gradient": genn_model.create_var_ref(output, "DeltaB")}
output_bias_reduction = model.add_custom_update("output_bias_reduction", "GradientBatchReduce", eprop.gradient_batch_reduce_model, 
                                                {}, gradient_batch_reduce_vars, output_bias_reduction_var_refs)

# Add custom updates for updating reduced weights using Adam optimiser
optimisers = []
if args.num_recurrent_alif > 0:
    input_recurrent_alif_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(input_recurrent_alif_reduction, "reducedGradient"),
                                               "variable": genn_model.create_wu_var_ref(input_recurrent_alif, "g")}
    input_recurrent_alif_optimiser = model.add_custom_update("input_recurrent_alif_optimiser", "GradientLearn", eprop.adam_optimizer_model,
                                                             adam_params, adam_vars, input_recurrent_alif_optimiser_var_refs)
    recurrent_alif_output_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_alif_output_reduction, "reducedGradient"),
                                                "variable": genn_model.create_wu_var_ref(recurrent_alif_output, "g" , output_recurrent_alif, "g")}
    recurrent_alif_output_optimiser = model.add_custom_update("recurrent_alif_output_optimiser", "GradientLearn", eprop.adam_optimizer_model,
                                                              adam_params, adam_vars, recurrent_alif_output_optimiser_var_refs)
    optimisers.extend([recurrent_alif_output_optimiser, input_recurrent_alif_optimiser])

if args.num_recurrent_lif > 0:
    input_recurrent_lif_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(input_recurrent_lif_reduction, "reducedGradient"),
                                              "variable": genn_model.create_wu_var_ref(input_recurrent_lif, "g")}
    input_recurrent_lif_optimiser = model.add_custom_update("input_recurrent_lif_optimiser", "GradientLearn", eprop.adam_optimizer_model,
                                                            adam_params, adam_vars, input_recurrent_lif_optimiser_var_refs)
    recurrent_lif_output_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_lif_output_reduction, "reducedGradient"),
                                               "variable": genn_model.create_wu_var_ref(recurrent_lif_output, "g" , output_recurrent_lif, "g")}
    recurrent_lif_output_optimiser = model.add_custom_update("recurrent_lif_output_optimiser", "GradientLearn", eprop.adam_optimizer_model,
                                                             adam_params, adam_vars, recurrent_lif_output_optimiser_var_refs)
    optimisers.extend([recurrent_lif_output_optimiser, input_recurrent_lif_optimiser])

if not args.feedforward:
    if args.num_recurrent_alif > 0:
        recurrent_alif_recurrent_alif_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_alif_recurrent_alif_reduction, "reducedGradient"),
                                                            "variable": genn_model.create_wu_var_ref(recurrent_alif_recurrent_alif, "g")}
        recurrent_alif_recurrent_alif_optimiser = model.add_custom_update("recurrent_alif_recurrent_alif_optimiser", "GradientLearn", eprop.adam_optimizer_model,
                                                                          adam_params, adam_vars, recurrent_alif_recurrent_alif_optimiser_var_refs)
        optimisers.append(recurrent_alif_recurrent_alif_optimiser)
    if args.num_recurrent_lif > 0:
        recurrent_lif_recurrent_lif_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_lif_recurrent_lif_reduction, "reducedGradient"),
                                                          "variable": genn_model.create_wu_var_ref(recurrent_lif_recurrent_lif, "g")}
        recurrent_lif_recurrent_lif_optimiser = model.add_custom_update("recurrent_lif_recurrent_alif_optimiser", "GradientLearn", eprop.adam_optimizer_model,
                                                                        adam_params, adam_vars, recurrent_lif_recurrent_lif_optimiser_var_refs)
        optimisers.append(recurrent_lif_recurrent_lif_optimiser)
output_bias_optimiser_var_refs = {"gradient": genn_model.create_var_ref(output_bias_reduction, "reducedGradient"),
                                  "variable": genn_model.create_var_ref(output, "B")}
output_bias_optimiser = model.add_custom_update("output_bias_optimiser", "GradientLearn", eprop.adam_optimizer_model,
                                                adam_params, adam_vars, output_bias_optimiser_var_refs)
optimisers.append(output_bias_optimiser)

# Build and load model
stimuli_timesteps = int(np.ceil(data_loader.max_stimuli_time / args.dt))

# Build model (only on first rank if using NCCL)
if first_rank:
    model.build()

# If we're using NCCL, wait for all ranks to reach this point
if args.use_nccl:
    comm.Barrier()

model.load(num_recording_timesteps=stimuli_timesteps)

# If we're using NCCL
if args.use_nccl:
    # Generate unique ID for our NCCL 'clique' on first rank
    if rank == 0:
        model._slm.nccl_generate_unique_id()

    # Broadcast our  NCCL clique ID across all ranks
    nccl_unique_id_view = model._slm.nccl_assign_external_unique_id()
    comm.Bcast(nccl_unique_id_view, root=0)

    # Initialise NCCL communicator
    model._slm.nccl_init_communicator(rank, num_ranks)

# Calculate initial transpose feedback weights
model.custom_update("CalculateTranspose")

learning_rate = args.learning_rate

# Get views
input_neuron_start_spike = input.vars["startSpike"].view
input_neuron_end_spike = input.vars["endSpike"].view
input_spike_times_view = input.extra_global_params["spikeTimes"].view
output_labels_view = output.extra_global_params["labels"].view
output_pi_view = output.vars["Pi"].view
output_e_view = output.vars["E"].view
output_b_view = output.vars["B"].view

if args.num_recurrent_alif > 0:
    input_recurrent_alif_g_view = input_recurrent_alif.vars["g"].view
    recurrent_alif_output_g_view = recurrent_alif_output.vars["g"].view
if args.num_recurrent_lif > 0:
    input_recurrent_lif_g_view = input_recurrent_lif.vars["g"].view
    recurrent_lif_output_g_view = recurrent_lif_output.vars["g"].view

if not args.feedforward:
    if args.num_recurrent_alif > 0:
        recurrent_alif_recurrent_alif_g_view = recurrent_alif_recurrent_alif.vars["g"].view
    if args.num_recurrent_lif > 0:
        recurrent_lif_recurrent_lif_g_view = recurrent_lif_recurrent_lif.vars["g"].view

# Open file
if first_rank:
    if args.resume_epoch is None:
        performance_file = open(os.path.join(output_directory, "performance.csv"), "w")
        performance_csv = csv.writer(performance_file, delimiter=",")
        performance_csv.writerow(("Epoch", "Batch", "Num trials", "Number correct"))
    else:
        performance_file = open(os.path.join(output_directory, "performance.csv"), "a")
        performance_csv = csv.writer(performance_file, delimiter=",")

# Loop through epochs
epoch_start = 0 if args.resume_epoch is None else (args.resume_epoch + 1)
adam_step = 1
start_time = perf_counter()
for epoch in range(epoch_start, args.num_epochs):
    # If learning rate decay is on
    if args.learning_rate_decay_epochs != 0:
        # If we should decay learning rate this epoch
        if epoch != 0 and (epoch % args.learning_rate_decay_epochs) == 0:
            learning_rate *= args.learning_rate_decay

    if first_rank:
        print("Epoch %u - Learning rate %f" % (epoch, learning_rate))

    # Loop through batches of (preprocessed) data
    data_iter = iter(data_loader)
    for batch_idx, (events, labels) in enumerate(data_iter):
        if first_rank:
            print("\tBatch %u" % batch_idx)
        batch_start_time = perf_counter()

        # Reset time
        model.timestep = 0
        model.t = 0.0

        # Transform data into batch
        batched_data = dataloader.batch_events(events, batch_size)

        # Check that spike times will fit in view, copy them and push them
        assert len(batched_data.spike_times) <= len(input_spike_times_view)
        input_spike_times_view[0:len(batched_data.spike_times)] = batched_data.spike_times
        input.push_extra_global_param_to_device("spikeTimes")

        # Calculate start and end spike indices
        input_neuron_end_spike[:] = batched_data.end_spikes
        input_neuron_start_spike[:] = dataloader.get_start_spikes(batched_data.end_spikes)
        input.push_var_to_device("startSpike")
        input.push_var_to_device("endSpike")

         # Copy labels into output
        output_labels_view[0:len(labels)] = labels
        output.push_extra_global_param_to_device("labels")

        # Loop through timesteps
        classification_output = np.zeros((len(labels), num_outputs))
        for i in range(stimuli_timesteps):
            model.step_time()

            # Pull Pi from device and add to total
            # **TODO** sum Pis on device
            output.pull_var_from_device("Pi")
            if batch_size == 1:
                classification_output += output_pi_view[:num_outputs]
            else:
                classification_output += output_pi_view[:len(labels), :num_outputs]

        # Calculate number of outputs which match label
        num_correct = np.sum(np.argmax(classification_output[:len(labels),:], axis=1) == labels)
        num_total = len(labels)
        
        # If we're using NCCL, sum up correct across batch
        if args.use_nccl:
            num_correct = comm.allreduce(sendobj=num_correct, op=MPI.SUM)
            num_total = comm.allreduce(sendobj=num_total, op=MPI.SUM)

        if first_rank:
            print("\t\t%u / %u correct = %f %%" % (num_correct, num_total, 100.0 * num_correct / num_total))
            performance_csv.writerow((epoch, batch_idx, num_total, num_correct))
            performance_file.flush()

        # Update Adam optimiser scaling factors
        update_adam(learning_rate, adam_step, optimisers)
        adam_step += 1

        # Now batch is complete, reduce and then apply gradients
        model.custom_update("GradientBatchReduce")
        model.custom_update("GradientLearn")

        if args.record:
            # Download recording data
            model.pull_recording_buffers_from_device()

            # Calculate rank offset
            rank_offset = (rank * batch_size) if args.use_nccl else 0

            # Write spikes
            for i, s in enumerate(input.spike_recording_data):
                write_spike_file(os.path.join(output_directory, "input_spikes_%u_%u_%u.csv" % (epoch, batch_idx, rank_offset + i)), s)
            if args.num_recurrent_alif > 0:
                for i, s in enumerate(recurrent_alif.spike_recording_data):
                    write_spike_file(os.path.join(output_directory, "recurrent_spikes_%u_%u_%u.csv" % (epoch, batch_idx, rank_offset + i)), s)
            if args.num_recurrent_lif > 0:
                for i, s in enumerate(recurrent_lif.spike_recording_data):
                    write_spike_file(os.path.join(output_directory, "recurrent_lif_spikes_%u_%u_%u.csv" % (epoch, batch_idx, rank_offset + i)), s)

        batch_end_time = perf_counter()
        if first_rank:
            print("\t\tTime:%f ms" % ((batch_end_time - batch_start_time) * 1000.0))

    # Save weights and biases to disk
    if first_rank:
        if args.num_recurrent_alif > 0:
            input_recurrent_alif.pull_var_from_device("g")
            recurrent_alif_output.pull_var_from_device("g")

            np.save(os.path.join(output_directory, "g_input_recurrent_%u.npy" % epoch), input_recurrent_alif_g_view)
            np.save(os.path.join(output_directory, "g_recurrent_output_%u.npy" % epoch), recurrent_alif_output_g_view)
        if args.num_recurrent_lif > 0:
            input_recurrent_lif.pull_var_from_device("g")
            recurrent_lif_output.pull_var_from_device("g")

            np.save(os.path.join(output_directory, "g_input_recurrent_lif_%u.npy" % epoch), input_recurrent_lif_g_view)
            np.save(os.path.join(output_directory, "g_recurrent_lif_output_%u.npy" % epoch), recurrent_lif_output_g_view)

        if not args.feedforward:
            if args.num_recurrent_alif > 0:
                recurrent_alif_recurrent_alif.pull_var_from_device("g")
                np.save(os.path.join(output_directory, "g_recurrent_recurrent_%u.npy" % epoch), recurrent_alif_recurrent_alif_g_view)
            if args.num_recurrent_lif > 0:
                recurrent_lif_recurrent_lif.pull_var_from_device("g")
                np.save(os.path.join(output_directory, "g_recurrent_lif_recurrent_lif_%u.npy" % epoch), recurrent_lif_recurrent_lif_g_view)

        output.pull_var_from_device("B")
        np.save(os.path.join(output_directory, "b_output_%u.npy" % epoch), output_b_view)

end_time = perf_counter()
if first_rank:
    print("Time:%f ms" % ((end_time - start_time) * 1000.0))

    performance_file.close()
    if args.timing:
        print("Init: %f" % model.init_time)
        print("Init sparse: %f" % model.init_sparse_time)
        print("Neuron update: %f" % model.neuron_update_time)
        print("Presynaptic update: %f" % model.presynaptic_update_time)
        print("Synapse dynamics: %f" % model.synapse_dynamics_time)
        print("Gradient batch reduction custom update: %f" % model.get_custom_update_time("GradientBatchReduce"))
        print("Gradient learning custom update: %f" % model.get_custom_update_time("GradientLearn"))
        print("Gradient learning custom update transpose: %f" % model.get_custom_update_transpose_time("GradientLearn"))
   
