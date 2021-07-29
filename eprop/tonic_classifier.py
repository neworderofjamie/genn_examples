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
from pygenn.genn_wrapper.Models import (VarAccess_READ_ONLY,
                                        VarAccessMode_READ_ONLY, 
                                        VarAccess_REDUCE_BATCH_SUM)

from tonic_classifier_parser import parse_arguments
import dataloader

# Eprop imports
#import eprop

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

name_suffix, output_directory, args = parse_arguments(parser, description="Train eProp classifier")
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

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

# ----------------------------------------------------------------------------
# Custom models
# ----------------------------------------------------------------------------
adam_optimizer_model = genn_model.create_custom_custom_update_class(
    "adam_optimizer",
    param_names=["beta1", "beta2", "epsilon"],
    var_name_types=[("m", "scalar"), ("v", "scalar")],
    extra_global_params=[("alpha", "scalar"), ("firstMomentScale", "scalar"),
                         ("secondMomentScale", "scalar")],
    var_refs=[("gradient", "scalar", VarAccessMode_READ_ONLY), ("variable", "scalar")],
    update_code="""
    // Update biased first moment estimate
    $(m) = ($(beta1) * $(m)) + ((1.0 - $(beta1)) * $(gradient));
    // Update biased second moment estimate
    $(v) = ($(beta2) * $(v)) + ((1.0 - $(beta2)) * $(gradient) * $(gradient));
    // Add gradient to variable, scaled by learning rate
    $(variable) -= ($(alpha) * $(m) * $(firstMomentScale)) / (sqrt($(v) * $(secondMomentScale)) + $(epsilon));
    """)

gradient_batch_reduce_model = genn_model.create_custom_custom_update_class(
    "gradient_batch_reduce",
    var_name_types=[("reducedGradient", "scalar", VarAccess_REDUCE_BATCH_SUM)],
    var_refs=[("gradient", "scalar")],
    update_code="""
    $(reducedGradient) = $(gradient);
    $(gradient) = 0;
    """)

#----------------------------------------------------------------------------
# Neuron models
#----------------------------------------------------------------------------
recurrent_alif_model = genn_model.create_custom_neuron_class(
    "recurrent_alif",
    param_names=["TauM", "TauAdap", "Vthresh", "TauRefrac", "Beta"],
    var_name_types=[("V", "scalar"), ("A", "scalar"), ("RefracTime", "scalar"), ("E", "scalar")],
    additional_input_vars=[("ISynFeedback", "scalar", 0.0)],
    derived_params=[("Alpha", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))()),
                    ("Rho", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[1]))())],

    sim_code="""
    $(E) = $(ISynFeedback);
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

recurrent_lif_model = genn_model.create_custom_neuron_class(
    "recurrent_lif",
    param_names=["TauM", "Vthresh", "TauRefrac"],
    var_name_types=[("V", "scalar"), ("RefracTime", "scalar"), ("E", "scalar")],
    additional_input_vars=[("ISynFeedback", "scalar", 0.0)],
    derived_params=[("Alpha", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())],
   
    sim_code="""
    $(E) = $(ISynFeedback);
    $(V) = ($(Alpha) * $(V)) + $(Isyn);
    if ($(RefracTime) > 0.0) {
      $(RefracTime) -= DT;
    }
    """,
    reset_code="""
    $(RefracTime) = $(TauRefrac);
    $(V) -= $(Vthresh);
    """,
    threshold_condition_code="""
    $(RefracTime) <= 0.0 && $(V) >= $(Vthresh)
    """,
    is_auto_refractory_required=False)

# **TODO** helper function to generate these models for arbitrary number of output neurons
output_classification_model_16 = genn_model.create_custom_neuron_class(
    "output_classification_16",
    param_names=["TauOut", "TrialTime"],
    var_name_types=[("Y", "scalar"), ("Pi", "scalar"), ("E", "scalar"), ("B", "scalar", VarAccess_READ_ONLY), ("DeltaB", "scalar")],
    derived_params=[("Kappa", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())],
    extra_global_params=[("labels", "uint8_t*")],

    sim_code="""
    $(Y) = ($(Kappa) * $(Y)) + $(Isyn) + $(B);
    scalar m = $(Y);
    m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x1));
    m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x2));
    m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x4));
    m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x8));
    const scalar expPi = exp($(Y) - m);
    scalar sumExpPi = expPi;
    sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x1);
    sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x2);
    sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x4);
    sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x8);
    $(Pi) = expPi / sumExpPi;

    const scalar piStar = ($(id) == $(labels)[$(batch)]) ? 1.0 : 0.0;
    $(E) = $(Pi) - piStar;

    $(DeltaB) += $(E);
    """,
    is_auto_refractory_required=False)

output_classification_model_32 = genn_model.create_custom_neuron_class(
    "output_classification_32",
    param_names=["TauOut", "TrialTime"],
    var_name_types=[("Y", "scalar"), ("Pi", "scalar"), ("E", "scalar"), ("B", "scalar", VarAccess_READ_ONLY), ("DeltaB", "scalar")],
    derived_params=[("Kappa", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())],
    extra_global_params=[("labels", "uint8_t*")],

    sim_code="""
    $(Y) = ($(Kappa) * $(Y)) + $(Isyn) + $(B);
    scalar m = $(Y);
    m = fmax(m, __shfl_xor_sync(0xFFFFFFFF, m, 0x1));
    m = fmax(m, __shfl_xor_sync(0xFFFFFFFF, m, 0x2));
    m = fmax(m, __shfl_xor_sync(0xFFFFFFFF, m, 0x4));
    m = fmax(m, __shfl_xor_sync(0xFFFFFFFF, m, 0x8));
    m = fmax(m, __shfl_xor_sync(0xFFFFFFFF, m, 0x10));
    const scalar expPi = exp($(Y) - m);
    scalar sumExpPi = expPi;
    sumExpPi +=  __shfl_xor_sync(0xFFFFFFFF, sumExpPi, 0x1);
    sumExpPi +=  __shfl_xor_sync(0xFFFFFFFF, sumExpPi, 0x2);
    sumExpPi +=  __shfl_xor_sync(0xFFFFFFFF, sumExpPi, 0x4);
    sumExpPi +=  __shfl_xor_sync(0xFFFFFFFF, sumExpPi, 0x8);
    sumExpPi +=  __shfl_xor_sync(0xFFFFFFFF, sumExpPi, 0x10);
    $(Pi) = expPi / sumExpPi;

    const scalar piStar = ($(id) == $(labels)[$(batch)]) ? 1.0 : 0.0;
    $(E) = $(Pi) - piStar;

    $(DeltaB) += $(E);
    """,
    is_auto_refractory_required=False)

#----------------------------------------------------------------------------
# Weight update models
#----------------------------------------------------------------------------
feedback_model = genn_model.create_custom_weight_update_class(
    "feedback",
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY)],
    synapse_dynamics_code="""
    $(addToInSyn, $(g) * $(E_pre));
    """)

eprop_alif_model = genn_model.create_custom_weight_update_class(
    "eprop_alif",
    param_names=["TauE", "TauA", "CReg", "FTarget", "TauFAvg", "Beta"],
    derived_params=[("Alpha", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))()),
                    ("Rho", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[1]))()),
                    ("FTargetTimestep", genn_model.create_dpf_class(lambda pars, dt: (pars[3] * dt) / 1000.0)()),
                    ("AlphaFAv", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[4]))())],
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY), ("eFiltered", "scalar"), ("epsilonA", "scalar"), ("DeltaG", "scalar")],
    pre_var_name_types=[("ZFilter", "scalar")],
    post_var_name_types=[("Psi", "scalar"), ("FAvg", "scalar")],
    
    sim_code="""
    $(addToInSyn, $(g));
    """,

    pre_spike_code="""
    $(ZFilter) += 1.0;
    """,
    pre_dynamics_code="""
    $(ZFilter) *= $(Alpha);
    """,

    post_spike_code="""
    $(FAvg) += (1.0 - $(AlphaFAv));
    """,
    post_dynamics_code="""
    $(FAvg) *= $(AlphaFAv);
    if ($(RefracTime_post) > 0.0) {
      $(Psi) = 0.0;
    }
    else {
      $(Psi) = (1.0 / $(Vthresh_post)) * 0.3 * fmax(0.0, 1.0 - fabs(($(V_post) - ($(Vthresh_post) + ($(Beta_post) * $(A_post)))) / $(Vthresh_post)));
    }
    """,

    synapse_dynamics_code="""
    // Calculate some common factors in e and epsilon update
    scalar epsilonA = $(epsilonA);
    const scalar psiZFilter = $(Psi) * $(ZFilter);
    const scalar psiBetaEpsilonA = $(Psi) * $(Beta) * epsilonA;
    
    // Calculate e and episilonA
    const scalar e = psiZFilter  - psiBetaEpsilonA;
    $(epsilonA) = psiZFilter + (($(Rho) * epsilonA) - psiBetaEpsilonA);
    
    // Calculate filtered version of eligibility trace
    scalar eFiltered = $(eFiltered);
    eFiltered = (eFiltered * $(Alpha)) + e;
    
    // Apply weight update
    $(DeltaG) += (eFiltered * $(E_post)) + (($(FAvg) - $(FTargetTimestep)) * $(CReg) * e);
    $(eFiltered) = eFiltered;
    """)

eprop_lif_model = genn_model.create_custom_weight_update_class(
    "eprop_lif",
    param_names=["TauE", "CReg", "FTarget", "TauFAvg"],
    derived_params=[("Alpha", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))()),
                    ("FTargetTimestep", genn_model.create_dpf_class(lambda pars, dt: (pars[2] * dt) / 1000.0)()),
                    ("AlphaFAv", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[3]))())],
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY), ("eFiltered", "scalar"), ("DeltaG", "scalar")],
    pre_var_name_types=[("ZFilter", "scalar")],
    post_var_name_types=[("Psi", "scalar"), ("FAvg", "scalar")],
    
    sim_code="""
    $(addToInSyn, $(g));
    """,

    pre_spike_code="""
    $(ZFilter) += 1.0;
    """,
    pre_dynamics_code="""
    $(ZFilter) *= $(Alpha);
    """,

    post_spike_code="""
    $(FAvg) += (1.0 - $(AlphaFAv));
    """,
    post_dynamics_code="""
    $(FAvg) *= $(AlphaFAv);
    if ($(RefracTime_post) > 0.0) {
      $(Psi) = 0.0;
    }
    else {
      $(Psi) = (1.0 / $(Vthresh_post)) * 0.3 * fmax(0.0, 1.0 - fabs(($(V_post) - $(Vthresh_post)) / $(Vthresh_post)));
    }
    """,

    synapse_dynamics_code="""
    const scalar e = $(ZFilter) * $(Psi);
    scalar eFiltered = $(eFiltered);
    eFiltered = (eFiltered * $(Alpha)) + e;
    $(DeltaG) += (eFiltered * $(E_post)) + (($(FAvg) - $(FTargetTimestep)) * $(CReg) * e);
    $(eFiltered) = eFiltered;
    """)

output_learning_model = genn_model.create_custom_weight_update_class(
    "output_learning",
    param_names=["TauE"],
    derived_params=[("Alpha", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())],
    var_name_types=[("g", "scalar", VarAccess_READ_ONLY), ("DeltaG", "scalar")],
    pre_var_name_types=[("ZFilter", "scalar")],

    sim_code="""
    $(addToInSyn, $(g));
    """,

    pre_spike_code="""
    $(ZFilter) += 1.0;
    """,
    pre_dynamics_code="""
    $(ZFilter) *= $(Alpha);
    """,

    synapse_dynamics_code="""
    $(DeltaG) += $(ZFilter) * $(E_post);
    """)

#----------------------------------------------------------------------------
# Postsynaptic models
#----------------------------------------------------------------------------
feedback_psm_model = genn_model.create_custom_postsynaptic_class(
    "feedback_psm",
    apply_input_code="""
    $(ISynFeedback) += $(inSyn);
    $(inSyn) = 0;
    """)

# Create dataset
sensor_size = None
polarity = False
if args.dataset == "shd":
    dataset = tonic.datasets.SHD(save_to='./data', train=True)
elif args.dataset == "smnist":
    dataset = tonic.datasets.SMNIST(save_to='./data', train=True, num_neurons=79)
elif args.dataset == "dvs_gesture":
    transform = tonic.transforms.Compose([
        tonic.transforms.Downsample(spatial_factor=0.25)])
    dataset = tonic.datasets.DVSGesture(save_to='./data', train=True, 
                                        transform=transform)
    sensor_size = (32, 32)
    polarity = True
else:
    raise RuntimeError("Unknown dataset '%s'" % args.dataset)

# Create loader
start_processing_time = perf_counter()
data_loader = dataloader.DataLoader(dataset, shuffle=True, batch_size=args.batch_size,
                                    sensor_size=sensor_size, polarity=polarity)
end_process_time = perf_counter()
print("Data processing time:%f ms" % ((end_process_time - start_processing_time) * 1000.0))

# Calculate number of input neurons from sensor size
num_input_neurons = np.product(dataset.sensor_size) 

# Calculate number of valid outputs from classes
num_outputs = len(dataset.classes)

# Round up to power-of-two
num_output_neurons = int(2**(np.ceil(np.log2(num_outputs))))

# Mapping of output neuron numbers to models
output_neuron_models = {16: output_classification_model_16,
                        32: output_classification_model_32}

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
eprop_lif_params = {"TauE": 20.0, "CReg": 1.0 / 1000.0, "FTarget": 10.0, "TauFAvg": 500.0}
eprop_alif_params = {"TauE": 20.0, "TauA": 2000.0, "CReg": 1.0 / 1000.0,
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
kwargs = {"selectGPUByDeviceID": True, "deviceSelectMethod": DeviceSelect_MANUAL} if args.cuda_visible_devices else {}
model = genn_model.GeNNModel("float", "%s_tonic_classifier_%s" % (args.dataset, name_suffix),
                             **kwargs)
model.dT = args.dt
model.timing_enabled = args.timing
model.batch_size = args.batch_size

# Add neuron populations
input = model.add_neuron_population("Input", num_input_neurons, "SpikeSourceArray",
                                    {}, {"startSpike": None, "endSpike": None})
if args.num_recurrent_alif > 0:
    recurrent_alif = model.add_neuron_population("RecurrentALIF", args.num_recurrent_alif, recurrent_alif_model,
                                                 recurrent_alif_params, recurrent_alif_vars)
    recurrent_alif.spike_recording_enabled = args.record
if args.num_recurrent_lif > 0:
    recurrent_lif = model.add_neuron_population("RecurrentLIF", args.num_recurrent_lif, recurrent_lif_model,
                                                recurrent_lif_params, recurrent_lif_vars)
    recurrent_lif.spike_recording_enabled = args.record

output = model.add_neuron_population("Output", num_output_neurons, output_neuron_models[num_output_neurons],
                                     output_params, output_vars)

# Allocate memory for input spikes and labels
input.set_extra_global_param("spikeTimes", np.zeros(args.batch_size * data_loader.max_spikes_per_stimuli, dtype=np.float32))
output.set_extra_global_param("labels", np.zeros(args.batch_size, dtype=np.uint8))

# Turn on recording
input.spike_recording_enabled = args.record

# (For now) check that there aren't both LIF and ALIF recurrent neurons
assert not (args.num_recurrent_alif > 0 and args.num_recurrent_lif > 0)

# Add synapse populations
if args.num_recurrent_alif > 0:
    input_recurrent_alif = model.add_synapse_population(
        "InputRecurrentALIF", "DENSE_INDIVIDUALG", NO_DELAY,
        input, recurrent_alif,
        eprop_alif_model, eprop_alif_params, input_recurrent_alif_vars, eprop_pre_vars, eprop_post_vars,
        "DeltaCurr", {}, {})
    recurrent_alif_output = model.add_synapse_population(
        "RecurrentALIFOutput", "DENSE_INDIVIDUALG", NO_DELAY,
        recurrent_alif, output,
        output_learning_model, recurrent_output_params, recurrent_alif_output_vars, recurrent_output_pre_vars, {},
        "DeltaCurr", {}, {})
    output_recurrent_alif = model.add_synapse_population(
        "OutputRecurrentALIF", "DENSE_INDIVIDUALG", NO_DELAY,
        output, recurrent_alif,
        feedback_model, {}, {"g": 0.0}, {}, {},
        feedback_psm_model, {}, {})

if args.num_recurrent_lif > 0:
    input_recurrent_lif = model.add_synapse_population(
        "InputRecurrentLIF", "DENSE_INDIVIDUALG", NO_DELAY,
        input, recurrent_lif,
        eprop_lif_model, eprop_lif_params, input_recurrent_lif_vars, eprop_pre_vars, eprop_post_vars,
        "DeltaCurr", {}, {})
    recurrent_lif_output = model.add_synapse_population(
        "RecurrentLIFOutput", "DENSE_INDIVIDUALG", NO_DELAY,
        recurrent_lif, output,
        output_learning_model, recurrent_output_params, recurrent_lif_output_vars, recurrent_output_pre_vars, {},
        "DeltaCurr", {}, {})
    output_recurrent_lif = model.add_synapse_population(
        "OutputRecurrentLIF", "DENSE_INDIVIDUALG", NO_DELAY,
        output, recurrent_lif,
        feedback_model, {}, {"g": 0.0}, {}, {},
        feedback_psm_model, {}, {})

if not args.feedforward:
    if args.num_recurrent_alif > 0:
        recurrent_alif_recurrent_alif = model.add_synapse_population(
            "RecurrentALIFRecurrentALIF", "DENSE_INDIVIDUALG", NO_DELAY,
            recurrent_alif, recurrent_alif,
            eprop_alif_model, eprop_alif_params, recurrent_alif_recurrent_alif_vars, eprop_pre_vars, eprop_post_vars,
            "DeltaCurr", {}, {})
    if args.num_recurrent_lif > 0:
        recurrent_lif_recurrent_lif = model.add_synapse_population(
            "RecurrentLIFRecurrentLIF", "DENSE_INDIVIDUALG", NO_DELAY,
            recurrent_lif, recurrent_lif,
            eprop_lif_model, eprop_lif_params, recurrent_lif_recurrent_lif_vars, eprop_pre_vars, eprop_post_vars,
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
    input_recurrent_alif_reduction = model.add_custom_update("input_recurrent_alif_reduction", "GradientBatchReduce", gradient_batch_reduce_model, 
                                                             {}, gradient_batch_reduce_vars, input_recurrent_alif_reduction_var_refs)

    recurrent_alif_output_reduction_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_alif_output, "DeltaG")}
    recurrent_alif_output_reduction = model.add_custom_update("recurrent_alif_output_reduction", "GradientBatchReduce", gradient_batch_reduce_model, 
                                                              {}, gradient_batch_reduce_vars, recurrent_alif_output_reduction_var_refs)
if args.num_recurrent_lif > 0:
    input_recurrent_lif_reduction_var_refs = {"gradient": genn_model.create_wu_var_ref(input_recurrent_lif, "DeltaG")}
    input_recurrent_lif_reduction = model.add_custom_update("input_recurrent_lif_reduction", "GradientBatchReduce", gradient_batch_reduce_model, 
                                                            {}, gradient_batch_reduce_vars, input_recurrent_lif_reduction_var_refs)
    recurrent_lif_output_reduction_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_lif_output, "DeltaG")}
    recurrent_lif_output_reduction = model.add_custom_update("recurrent_lif_output_reduction", "GradientBatchReduce", gradient_batch_reduce_model, 
                                                             {}, gradient_batch_reduce_vars, recurrent_lif_output_reduction_var_refs)

if not args.feedforward:
    if args.num_recurrent_alif > 0:
        recurrent_alif_recurrent_alif_reduction_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_alif_recurrent_alif, "DeltaG")}
        recurrent_alif_recurrent_alif_reduction = model.add_custom_update("recurrent_alif_recurrent_alif_reduction", "GradientBatchReduce", gradient_batch_reduce_model,
                                                                          {}, gradient_batch_reduce_vars, recurrent_alif_recurrent_alif_reduction_var_refs)
    if args.num_recurrent_lif > 0:
        recurrent_lif_recurrent_lif_reduction_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_lif_recurrent_lif, "DeltaG")}
        recurrent_lif_recurrent_lif_reduction = model.add_custom_update("recurrent_lif_recurrent_lif_reduction", "GradientBatchReduce", gradient_batch_reduce_model,
                                                                          {}, gradient_batch_reduce_vars, recurrent_lif_recurrent_lif_reduction_var_refs)

output_bias_reduction_var_refs = {"gradient": genn_model.create_var_ref(output, "DeltaB")}
output_bias_reduction = model.add_custom_update("output_bias_reduction", "GradientBatchReduce", gradient_batch_reduce_model, 
                                                {}, gradient_batch_reduce_vars, output_bias_reduction_var_refs)
                        
# Add custom updates for updating reduced weights using Adam optimiser
optimisers = []
if args.num_recurrent_alif > 0:
    input_recurrent_alif_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(input_recurrent_alif_reduction, "reducedGradient"),
                                               "variable": genn_model.create_wu_var_ref(input_recurrent_alif, "g")}
    input_recurrent_alif_optimiser = model.add_custom_update("input_recurrent_alif_optimiser", "GradientLearn", adam_optimizer_model,
                                                             adam_params, adam_vars, input_recurrent_alif_optimiser_var_refs)
    recurrent_alif_output_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_alif_output_reduction, "reducedGradient"),
                                                "variable": genn_model.create_wu_var_ref(recurrent_alif_output, "g" , output_recurrent_alif, "g")}
    recurrent_alif_output_optimiser = model.add_custom_update("recurrent_alif_output_optimiser", "GradientLearn", adam_optimizer_model,
                                                              adam_params, adam_vars, recurrent_alif_output_optimiser_var_refs)
    optimisers.extend([recurrent_alif_output_optimiser, input_recurrent_alif_optimiser])

if args.num_recurrent_lif > 0:
    input_recurrent_lif_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(input_recurrent_lif_reduction, "reducedGradient"),
                                              "variable": genn_model.create_wu_var_ref(input_recurrent_lif, "g")}
    input_recurrent_lif_optimiser = model.add_custom_update("input_recurrent_lif_optimiser", "GradientLearn", adam_optimizer_model,
                                                            adam_params, adam_vars, input_recurrent_lif_optimiser_var_refs)
    recurrent_lif_output_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_lif_output_reduction, "reducedGradient"),
                                               "variable": genn_model.create_wu_var_ref(recurrent_lif_output, "g" , output_recurrent_lif, "g")}
    recurrent_lif_output_optimiser = model.add_custom_update("recurrent_lif_output_optimiser", "GradientLearn", adam_optimizer_model,
                                                             adam_params, adam_vars, recurrent_lif_output_optimiser_var_refs)
    optimisers.extend([recurrent_lif_output_optimiser, input_recurrent_lif_optimiser])

if not args.feedforward:
    if args.num_recurrent_alif > 0:
        recurrent_alif_recurrent_alif_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_alif_recurrent_alif_reduction, "reducedGradient"),
                                                            "variable": genn_model.create_wu_var_ref(recurrent_alif_recurrent_alif, "g")}
        recurrent_alif_recurrent_alif_optimiser = model.add_custom_update("recurrent_alif_recurrent_alif_optimiser", "GradientLearn", adam_optimizer_model,
                                                                          adam_params, adam_vars, recurrent_alif_recurrent_alif_optimiser_var_refs)
        optimisers.append(recurrent_alif_recurrent_alif_optimiser)
    if args.num_recurrent_lif > 0:
        recurrent_lif_recurrent_lif_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_lif_recurrent_lif_reduction, "reducedGradient"),
                                                          "variable": genn_model.create_wu_var_ref(recurrent_lif_recurrent_lif, "g")}
        recurrent_lif_recurrent_lif_optimiser = model.add_custom_update("recurrent_lif_recurrent_alif_optimiser", "GradientLearn", adam_optimizer_model,
                                                                        adam_params, adam_vars, recurrent_lif_recurrent_lif_optimiser_var_refs)
        optimisers.append(recurrent_lif_recurrent_lif_optimiser)
output_bias_optimiser_var_refs = {"gradient": genn_model.create_var_ref(output_bias_reduction, "reducedGradient"),
                                  "variable": genn_model.create_var_ref(output, "B")}
output_bias_optimiser = model.add_custom_update("output_bias_optimiser", "GradientLearn", adam_optimizer_model,
                                                adam_params, adam_vars, output_bias_optimiser_var_refs)
optimisers.append(output_bias_optimiser)

# Build and load model
stimuli_timesteps = int(np.ceil(data_loader.max_stimuli_time / args.dt))
model.build()
model.load(num_recording_timesteps=stimuli_timesteps)

# Calculate initial transpose feedback weights
model.custom_update("CalculateTranspose")

learning_rate = 0.001

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
    print("Epoch %u" % epoch)

    # Loop through batches of (preprocessed) data
    data_iter = iter(data_loader)
    for batch_idx, (events, labels) in enumerate(data_iter):
        print("\tBatch %u" % batch_idx)
        batch_start_time = perf_counter()

        # Reset time
        model.timestep = 0
        model.t = 0.0

        # Transform data into batch
        batched_data = dataloader.batch_events(events, args.batch_size)

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
            if args.batch_size == 1:
                classification_output += output_pi_view[:num_outputs]
            else:
                classification_output += output_pi_view[:len(labels), :num_outputs]

        # Calculate number of outputs which match label
        num_correct = np.sum(np.argmax(classification_output[:len(labels),:], axis=1) == labels)

        print("\t\t%u / %u correct = %f %%" % (num_correct, len(labels), 100.0 * num_correct / len(labels)))
        performance_csv.writerow((epoch, batch_idx, len(labels), num_correct))
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

            # Write spikes
            for i, s in enumerate(input.spike_recording_data):
                write_spike_file(os.path.join(output_directory, "input_spikes_%u_%u_%u.csv" % (epoch, batch_idx, i)), s)
            if args.num_recurrent_alif > 0:
                for i, s in enumerate(recurrent_alif.spike_recording_data):
                    write_spike_file(os.path.join(output_directory, "recurrent_spikes_%u_%u_%u.csv" % (epoch, batch_idx, i)), s)
            if args.num_recurrent_lif > 0:
                for i, s in enumerate(recurrent_lif.spike_recording_data):
                    write_spike_file(os.path.join(output_directory, "recurrent_lif_spikes_%u_%u_%u.csv" % (epoch, batch_idx, i)), s)

        batch_end_time = perf_counter()
        print("\t\tTime:%f ms" % ((batch_end_time - batch_start_time) * 1000.0))

    # Save weights and biases to disk
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
   
