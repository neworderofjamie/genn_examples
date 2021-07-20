import csv
import numpy as np
import tonic
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from time import perf_counter
from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY

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
parser.add_argument("--num-epochs", type=int, default=50)
parser.add_argument("--resume-epoch", type=int, default=None)
parser.add_argument("--dataset", choices=["smnist", "shd"], required=True)
args = parser.parse_args()

MAX_STIMULI_TIMES = {"smnist": 1568.0, "shd": 1369.140625}
MAX_SPIKES_PER_STIMULI = {"smnist": 10000, "shd": 14917}

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999

WEIGHT_0 = 1.0


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
    var_refs=[("gradient", "scalar"), ("variable", "scalar")],
    update_code="""
    // Update biased first moment estimate
    $(m) = ($(beta1) * $(m)) + ((1.0 - $(beta1)) * $(gradient));
    // Update biased second moment estimate
    $(v) = ($(beta2) * $(v)) + ((1.0 - $(beta2)) * $(gradient) * $(gradient));
    // Add gradient to variable, scaled by learning rate
    $(variable) -= ($(alpha) * $(m) * $(firstMomentScale)) / (sqrt($(v) * $(secondMomentScale)) + $(epsilon));
    // Zero gradient
    $(gradient) = 0.0;
    """)

input_reset_model = genn_model.create_custom_custom_update_class(
    "input_reset",
    var_refs=[("ZFilter", "scalar")],
    update_code="""
    $(ZFilter) = 0.0;
    """)

recurrent_reset_model = genn_model.create_custom_custom_update_class(
    "recurrent_reset",
    var_refs=[("V", "scalar"), ("A", "scalar"), ("RefracTime", "scalar"), 
             ("ZFilter1", "scalar"), ("ZFilter2", "scalar")],
    update_code="""
    $(V) = 0.0;
    $(A) = 0.0;
    $(RefracTime) = 0.0;
    $(ZFilter1) = 0.0;
    $(ZFilter2) = 0.0;
    """)

output_reset_model = genn_model.create_custom_custom_update_class(
    "output_reset",
    var_refs=[("Y", "scalar")],
    update_code="""
    $(Y) = 0.0;
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

# **TODO** helper function to generate these models for arbitrary number of output neurons
output_classification_model_16 = genn_model.create_custom_neuron_class(
    "output_classification_16",
    param_names=["TauOut", "TrialTime"],
    var_name_types=[("Y", "scalar"), ("Pi", "scalar"), ("E", "scalar"), ("B", "scalar"), ("DeltaB", "scalar")],
    derived_params=[("Kappa", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())],
    extra_global_params=[("labels", "uint8_t*")],

    sim_code="""
    // Calculate trial index from time
    const int trial = (int)floor($(t) / $(TrialTime));

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

    const scalar piStar = ($(id) == $(labels)[trial]) ? 1.0 : 0.0;
    $(E) = $(Pi) - piStar;

    $(DeltaB) += $(E);
    """,
    is_auto_refractory_required=False)

output_classification_model_32 = genn_model.create_custom_neuron_class(
    "output_classification_32",
    param_names=["TauOut", "TrialTime"],
    var_name_types=[("Y", "scalar"), ("Pi", "scalar"), ("E", "scalar"), ("B", "scalar"), ("DeltaB", "scalar")],
    derived_params=[("Kappa", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())],
    extra_global_params=[("labels", "uint8_t*")],

    sim_code="""
    // Calculate trial index from time
    const int trial = (int)floor($(t) / $(TrialTime));

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

    const scalar piStar = ($(id) == $(labels)[trial]) ? 1.0 : 0.0;
    $(E) = $(Pi) - piStar;

    $(DeltaB) += $(E);
    """,
    is_auto_refractory_required=False)

#----------------------------------------------------------------------------
# Weight update models
#----------------------------------------------------------------------------
feedback_model = genn_model.create_custom_weight_update_class(
    "feedback",
    var_name_types=[("g", "scalar")],
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
    var_name_types=[("g", "scalar"), ("eFiltered", "scalar"), ("epsilonA", "scalar"), ("DeltaG", "scalar")],
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

output_learning_model = genn_model.create_custom_weight_update_class(
    "output_learning",
    param_names=["TauE"],
    derived_params=[("Alpha", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())],
    var_name_types=[("g", "scalar"), ("DeltaG", "scalar")],
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
if args.dataset == "shd":
    dataset = tonic.datasets.SHD(save_to='./data', train=True)
elif args.dataset == "smnist":
    dataset = tonic.datasets.SMNIST(save_to='./data', train=True)
else:
    raise RuntimeError("Unknown dataset '%s'" % args.dataset)

# Build file suffix
name_suffix = "%u" % (args.num_recurrent_alif)

# Create loader
data_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)

# Calculate number of input neurons from sensor size
# **NOTE** we add one as we use an additional neuron to 
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
recurrent_params = {"TauM": 20.0, "TauAdap": 2000.0, "Vthresh": 0.6, "TauRefrac": 5.0, "Beta": 0.0174}
recurrent_vars = {"V": 0.0, "A": 0.0, "RefracTime": 0.0, "E": 0.0}

# Output population
output_params = {"TauOut": 20.0, "TrialTime": MAX_STIMULI_TIMES[args.dataset]}
output_vars = {"Y": 0.0, "Pi": 0.0, "E": 0.0, "DeltaB": 0.0}
if args.resume_epoch is None:
    output_vars["B"] = 0.0
else:
    output_vars["B"] = np.load("b_%s_output_%s_%u.npy" % (args.dataset, args.resume_epoch, name_suffix))

# ----------------------------------------------------------------------------
# Synapse initialisation
# ----------------------------------------------------------------------------
# eProp parameters common across all populations
eprop_params = {"TauE": 20.0, "TauA": 2000.0, "CReg": 1.0 / (args.batch_size * 1000.0),
                "FTarget": 10.0, "TauFAvg": 500.0, "Beta": 0.0174}
eprop_pre_vars = {"ZFilter": 0.0}
eprop_post_vars = {"Psi": 0.0, "FAvg": 0.0}

# Input->recurrent synapse parameters
input_recurrent_vars = {"eFiltered": 0.0, "epsilonA": 0.0, "DeltaG": 0.0}
if args.resume_epoch is None:
    input_recurrent_vars["g"] = genn_model.init_var("Normal", {"mean": 0.0, "sd": WEIGHT_0 / np.sqrt(num_input_neurons)})
else:
    input_recurrent_vars["g"] = np.load("g_%s_input_recurrent_%s_%u.npy" % (args.dataset, args.resume_epoch, name_suffix))

# Recurrent->recurrent synapse parameters
recurrent_recurrent_vars = {"eFiltered": 0.0, "epsilonA": 0.0, "DeltaG": 0.0}
if args.resume_epoch is None:
    recurrent_recurrent_vars["g"] = genn_model.init_var("Normal", {"mean": 0.0, "sd": WEIGHT_0 / np.sqrt(args.num_recurrent_alif)})
else:
    recurrent_recurrent_vars["g"] = np.load("g_%s_recurrent_recurrent_%s_%u.npy" % (args.dataset, args.resume_epoch, name_suffix))

# Recurrent->output synapse parameters
recurrent_output_params = {"TauE": 20.0}
recurrent_output_pre_vars = {"ZFilter": 0.0}
recurrent_output_vars = {"DeltaG": 0.0}
if args.resume_epoch is None:
    recurrent_output_vars["g"] = genn_model.init_var("Normal", {"mean": 0.0, "sd": WEIGHT_0 / np.sqrt(args.num_recurrent_alif)})
else:
    recurrent_output_vars["g"] = np.load("g_%s_recurrent_output_%s_%u.npy" % (args.dataset, args.resume_epoch, name_suffix))

# Optimiser initialisation
adam_params = {"beta1": ADAM_BETA1, "beta2": ADAM_BETA2, "epsilon": 1E-8}
adam_vars = {"m": 0.0, "v": 0.0}

# ----------------------------------------------------------------------------
# Model description
# ----------------------------------------------------------------------------
model = genn_model.GeNNModel("float", "%s_tonic_classifier_%s" % (args.dataset, name_suffix))
model.dT = args.dt
model.timing_enabled = args.timing

# Add neuron populations
input = model.add_neuron_population("Input", num_input_neurons, "SpikeSourceArray",
                                    {}, {"startSpike": None, "endSpike": None})
recurrent = model.add_neuron_population("Recurrent", args.num_recurrent_alif, recurrent_alif_model,
                                        recurrent_params, recurrent_vars)
output = model.add_neuron_population("Output", num_output_neurons, output_neuron_models[num_output_neurons],
                                     output_params, output_vars)

# Allocate memory for input spikes and labels
input.set_extra_global_param("spikeTimes", np.zeros(args.batch_size * MAX_SPIKES_PER_STIMULI[args.dataset], dtype=np.float32))
output.set_extra_global_param("labels", np.zeros(args.batch_size, dtype=np.uint8))

# Turn on recording
input.spike_recording_enabled = args.record
recurrent.spike_recording_enabled = args.record

# Add synapse populations
input_recurrent = model.add_synapse_population(
    "InputRecurrent", "DENSE_INDIVIDUALG", NO_DELAY,
    input, recurrent,
    eprop_alif_model, eprop_params, input_recurrent_vars, eprop_pre_vars, eprop_post_vars,
    "DeltaCurr", {}, {})

recurrent_recurrent = model.add_synapse_population(
    "RecurrentRecurrent", "DENSE_INDIVIDUALG", NO_DELAY,
    recurrent, recurrent,
    eprop_alif_model, eprop_params, recurrent_recurrent_vars, eprop_pre_vars, eprop_post_vars,
    "DeltaCurr", {}, {})

recurrent_output = model.add_synapse_population(
    "RecurrentOutput", "DENSE_INDIVIDUALG", NO_DELAY,
    recurrent, output,
    output_learning_model, recurrent_output_params, recurrent_output_vars, recurrent_output_pre_vars, {},
    "DeltaCurr", {}, {})

output_recurrent = model.add_synapse_population(
    "OutputRecurrent", "DENSE_INDIVIDUALG", NO_DELAY,
    output, recurrent,
    feedback_model, {}, {"g": 0.0}, {}, {},
    feedback_psm_model, {}, {})

# Add custom update for calculating initial tranpose weights
model.add_custom_update("recurrent_hidden_transpose", "CalculateTranspose", "Transpose",
                        {}, {}, {"variable": genn_model.create_wu_var_ref(recurrent_output, "g", output_recurrent, "g")})

# Add custom updates for updating weights using Adam optimiser
input_recurrent_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(input_recurrent, "DeltaG"),
                                      "variable": genn_model.create_wu_var_ref(input_recurrent, "g")}
input_recurrent_optimiser = model.add_custom_update("input_recurrent_optimiser", "GradientLearn", adam_optimizer_model,
                                                    adam_params, adam_vars, input_recurrent_optimiser_var_refs)

recurrent_recurrent_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_recurrent, "DeltaG"),
                                          "variable": genn_model.create_wu_var_ref(recurrent_recurrent, "g")}
recurrent_recurrent_optimiser = model.add_custom_update("recurrent_recurrent_optimiser", "GradientLearn", adam_optimizer_model,
                                                        adam_params, adam_vars, recurrent_recurrent_optimiser_var_refs)

recurrent_output_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_output, "DeltaG"),
                                       "variable": genn_model.create_wu_var_ref(recurrent_output, "g" , output_recurrent, "g")}
recurrent_output_optimiser = model.add_custom_update("recurrent_output_optimiser", "GradientLearn", adam_optimizer_model,
                                                     adam_params, adam_vars, recurrent_output_optimiser_var_refs)

output_bias_optimiser_var_refs = {"gradient": genn_model.create_var_ref(output, "DeltaB"),
                                  "variable": genn_model.create_var_ref(output, "B")}
output_bias_optimiser = model.add_custom_update("output_bias_optimiser", "GradientLearn", adam_optimizer_model,
                                                adam_params, adam_vars, output_bias_optimiser_var_refs)

# Add custom updates for resetting model between trials
input_reset_var_refs = {"ZFilter": genn_model.create_wu_pre_var_ref(input_recurrent, "ZFilter")}
model.add_custom_update("input_reset", "Reset", input_reset_model,
                        {}, {}, input_reset_var_refs)

recurrent_reset_var_refs = {"V": genn_model.create_var_ref(recurrent, "V"),
                            "A": genn_model.create_var_ref(recurrent, "A"),
                            "RefracTime": genn_model.create_var_ref(recurrent, "RefracTime"),
                            "ZFilter1": genn_model.create_wu_pre_var_ref(recurrent_recurrent, "ZFilter"),
                            "ZFilter2": genn_model.create_wu_pre_var_ref(recurrent_output, "ZFilter")}
model.add_custom_update("recurrent_reset", "Reset", recurrent_reset_model,
                        {}, {}, recurrent_reset_var_refs)

output_reset_var_refs = {"Y": genn_model.create_var_ref(output, "Y")}
model.add_custom_update("output_reset", "Reset", output_reset_model,
                        {}, {}, output_reset_var_refs)

# Build and load model
stimuli_timesteps = int(np.ceil(MAX_STIMULI_TIMES[args.dataset] / args.dt))
model.build()
model.load(num_recording_timesteps=stimuli_timesteps * args.batch_size)

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
input_recurrent_g_view = input_recurrent.vars["g"].view
recurrent_recurrent_g_view = recurrent_recurrent.vars["g"].view
recurrent_output_g_view = recurrent_output.vars["g"].view

# Open file
if args.resume_epoch is None:
    performance_file = open("%s_performance_%s.csv" % (args.dataset, name_suffix), "w")
    performance_csv = csv.writer(performance_file, delimiter=",")
    performance_csv.writerow(("Epoch", "Batch", "Num trials", "Number correct"))
else:
    performance_file = open("%s_performance_%s.csv" % (args.dataset, name_suffix), "a")
    performance_csv = csv.writer(performance_file, delimiter=",")

# Loop through epochs
epoch_start = 0 if args.resume_epoch is None else (args.resume_epoch + 1)
adam_step = 1
start_time = perf_counter()
for epoch in range(epoch_start, args.num_epochs):
    print("Epoch %u" % epoch)

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

        # Concatenate together all spike times
        spike_times = np.concatenate([(i * 1000.0 * MAX_STIMULI_TIMES[args.dataset]) + e[:,0]
                                      for i, e in enumerate(batch_events)])

        spike_ids = np.concatenate([e[:,1] for e in batch_events]).astype(int)

        # Indirectly sort spikes, first by neuron id and then by time
        spike_order = np.lexsort((spike_times, spike_ids))

        # Check that spike times will fit in view, copy them and push them
        assert len(spike_times) <= len(input_spike_times_view)
        input_spike_times_view[0:len(spike_ids)] = spike_times[spike_order] / 1000.0
        input.push_extra_global_param_to_device("spikeTimes")

        # Calculate start and end spike indices
        input_neuron_end_spike[:] = np.cumsum(np.bincount(spike_ids, minlength=num_input_neurons))
        input_neuron_start_spike[:] = np.concatenate(([0], input_neuron_end_spike[:-1]))
        input.push_var_to_device("startSpike")
        input.push_var_to_device("endSpike")

        # Copy labels into output
        output_labels_view[0:len(batch_labels)] = batch_labels
        output.push_extra_global_param_to_device("labels")

        # Loop through trials in batch
        num_correct = 0
        for trial, label in enumerate(batch_labels):
            # Loop through timesteps in each trial
            classification_output = np.zeros(num_outputs)
            for i in range(stimuli_timesteps):
                model.step_time()

                # Pull Pi from device and add to total
                output.pull_var_from_device("Pi")
                classification_output += output_pi_view[:num_outputs]
            
            # Run custom update to reset model state
            model.custom_update("Reset")
            
            # If maximum output matches label, increment counter
            if np.argmax(classification_output) == label:
                num_correct += 1

        print("\t\t%u / %u correct = %f %%" % (num_correct, len(batch_events), 100.0 * num_correct / len(batch_events)))
        performance_csv.writerow((epoch, batch_idx, len(batch_events), num_correct))
        performance_file.flush()

        # Calculate the correct scaling for adam optimiser
        update_adam(learning_rate, adam_step, [input_recurrent_optimiser, recurrent_recurrent_optimiser,
                                               recurrent_output_optimiser, output_bias_optimiser])
        adam_step += 1

        # Now batch is complete, apply gradients
        model.custom_update("GradientLearn")

        if args.record:
            # Download recording data
            model.pull_recording_buffers_from_device()

            # Write spikes
            write_spike_file("%s_input_spikes_%s_%u_%u.csv" % (args.dataset, name_suffix, epoch, batch_idx), input.spike_recording_data)
            write_spike_file("%s_recurrent_spikes_%s_%u_%u.csv" % (args.dataset, name_suffix, epoch, batch_idx), recurrent.spike_recording_data)

        batch_end_time = perf_counter()
        print("\t\tTime:%f ms" % ((batch_end_time - batch_start_time) * 1000.0))

    # Pull weights and biases from device
    input_recurrent.pull_var_from_device("g")
    recurrent_recurrent.pull_var_from_device("g")
    recurrent_output.pull_var_from_device("g")
    output.pull_var_from_device("B")

    # Save weights and biases to disk
    np.save("g_%s_input_recurrent_%s_%u.npy" % (args.dataset, name_suffix, epoch), input_recurrent_g_view)
    np.save("g_%s_recurrent_recurrent_%s_%u.npy" % (args.dataset, name_suffix, epoch), recurrent_recurrent_g_view)
    np.save("g_%s_recurrent_output_%s_%u.npy" % (args.dataset, name_suffix, epoch), recurrent_output_g_view)
    np.save("b_%s_output_%s_%u.npy" % (args.dataset, name_suffix, epoch), output_b_view)
end_time = perf_counter()
print("Time:%f ms" % ((end_time - start_time) * 1000.0))

performance_file.close()
if args.timing:
    print("Init: %f" % model.init_time)
    print("Init sparse: %f" % model.init_sparse_time)
    print("Neuron update: %f" % model.neuron_update_time)
    print("Presynaptic update: %f" % model.presynaptic_update_time)
    print("Synapse dynamics: %f" % model.synapse_dynamics_time)
    print("Reset custom update: %f" % model.get_custom_update_time("Reset"))
    print("Gradient learning custom update: %f" % model.get_custom_update_time("GradientLearn"))
    print("Gradient learning custom update transpose: %f" % model.get_custom_update_transpose_time("GradientLearn"))
   
