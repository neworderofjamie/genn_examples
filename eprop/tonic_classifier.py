import numpy as np
import tonic
import matplotlib.pyplot as plt

from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY

# Eprop imports
import eprop

TIMESTEP_MS = 1.0
TIMING_ENABLED = True

CUE_TIME = 20.0

BATCH_SIZE = 512

NUM_RECURRENT_NEURONS = 800
NUM_OUTPUT_NEURONS = 32

WEIGHT_0 = 1.0

RESUME_EPOCH = None

#----------------------------------------------------------------------------
# Neuron models
#----------------------------------------------------------------------------
# **TODO** helper function to generate these models for arbitrary number of output neurons
output_classification_model = genn_model.create_custom_neuron_class(
    "output_classification",
    param_names=["TauOut", "TrialTime", "StimuliTime"],
    var_name_types=[("Y", "scalar"), ("Pi", "scalar"), ("E", "scalar"), ("B", "scalar"), ("DeltaB", "scalar")],
    derived_params=[("Kappa", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))())],
    extra_global_params=[("indices", "unsigned int*"), ("labels", "uint8_t*")],

    sim_code="""
    // Split timestep into trial index and time
    const int trial = (int)floor($(t) / $(TrialTime));
    const float trialTime = $(t) - (trial * $(TrialTime));

    $(Y) = ($(Kappa) * $(Y)) + $(Isyn) + $(B);
    scalar m = $(Y);
    m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x1));
    m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x2));
    m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x4));
    m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x8));
    m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x10));
    const scalar expPi = exp($(Y) - m);
    scalar sumExpPi = expPi;
    sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x1);
    sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x2);
    sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x4);
    sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x8);
    sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x10);
    $(Pi) = expPi / sumExpPi;

    // If we should be presenting stimuli
    if(trialTime < $(StimuliTime)) {
       $(E) = 0.0;
    }
    else {
       const scalar piStar = ($(id) == $(labels)[$(indices)[trial]]) ? 1.0 : 0.0;
       $(E) = $(Pi) - piStar;
    }
    $(DeltaB) += $(E);
    """,
    is_auto_refractory_required=False)

# Create dataset
dataset = tonic.datasets.SHD(save_to='./data', train=True)

# Calculate number of input neurons from sensor sizeu
num_input_neurons = np.product(dataset.sensor_size)

# Create dataset loader
dataset_loader = tonic.datasets.DataLoader(dataset, shuffle=True)

print("Reading...")

# Read events and targets
# **NOTE** dataset iterators tend to be slow so run before simulation
data = [d for d in dataset_loader]

# Convert all spike times from microseconds to milliseconds
for d in data:
    d[0][:,0] /= 1000.0

num_stimuli = len(data)
print("%u Stimuli" % num_stimuli)

# Calculate max stimuli duration
stimuli_durations = [np.amax(events[:,0])  for events, _ in data]
max_stimuli_duration = max(stimuli_durations)
print("Max stimuli duration = %f ms" % max_stimuli_duration)
"""
# Concatenate together all spike times, offsetting so each stimuli ends at the start of the cue time of each trial
spike_times = np.concatenate([(i * (max_stimuli_duration + CUE_TIME)) + events[:,0] + (max_stimuli_duration - d)
                              for i, (d, (events, _)) in enumerate(zip(stimuli_durations, data))])
spike_ids = np.concatenate([events[:,1] for events, _ in data])

# Indirectly sort spikes, first by neuron id and then by time
spike_order = np.lexsort((spike_times, spike_ids))
spike_times = spike_times[spike_order]
spike_ids = spike_ids[spike_order]

# Count number of spikes
input_neuron_end_times = np.cumsum(np.bincount(spike_ids, minlength=num_input_neurons))
input_neuron_start_times = np.concatenate(([0], input_neuron_end_times[:-1]))
"""
# ----------------------------------------------------------------------------
# Neuron initialisation
# ----------------------------------------------------------------------------
# Recurrent population
recurrent_params = {"TauM": 20.0, "TauAdap": 2000.0, "Vthresh": 0.6, "TauRefrac": 5.0, "Beta": 0.0174}
recurrent_vars = {"V": 0.0, "A": 0.0, "RefracTime": 0.0, "E": 0.0}

# Output population
output_params = {"TauOut": 20.0, "TrialTime": max_stimuli_duration + CUE_TIME, "StimuliTime": max_stimuli_duration}
output_vars = {"Y": 0.0, "Pi": 0.0, "E": 0.0, "B": 0.0, "DeltaB": 0.0}

# ----------------------------------------------------------------------------
# Synapse initialisation
# ----------------------------------------------------------------------------
# eProp parameters common across all populations
eprop_params = {"TauE": 20.0, "TauA": 2000.0, "CReg": 1.0 / (BATCH_SIZE * 1000.0),
                "FTarget": 10.0, "TauFAvg": 500.0, "Beta": 0.0174}
eprop_pre_vars = {"ZFilter": 0.0}
eprop_post_vars = {"Psi": 0.0, "FAvg": 0.0}

# Input->recurrent synapse parameters
input_recurrent_vars = {"eFiltered": 0.0, "epsilonA": 0.0, "DeltaG": 0.0}
if RESUME_EPOCH is None:
    input_recurrent_vars["g"] = genn_model.init_var("Normal", {"mean": 0.0, "sd": WEIGHT_0 / np.sqrt(num_input_neurons)})
else:
    input_recurrent_vars["g"] = None

# Recurrent->recurrent synapse parameters
recurrent_recurrent_vars = {"eFiltered": 0.0, "epsilonA": 0.0, "DeltaG": 0.0}
if RESUME_EPOCH is None:
    recurrent_recurrent_vars["g"] = genn_model.init_var("Normal", {"mean": 0.0, "sd": WEIGHT_0 / np.sqrt(NUM_RECURRENT_NEURONS)})
else:
    recurrent_recurrent_vars["g"] = None

# Recurrent->output synapse parameters
recurrent_output_params = {"TauE": 20.0}
recurrent_output_pre_vars = {"ZFilter": 0.0}
recurrent_output_vars = {"DeltaG": 0.0}
if RESUME_EPOCH is None:
    recurrent_output_vars["g"] = genn_model.init_var("Normal", {"mean": 0.0, "sd": WEIGHT_0 / np.sqrt(NUM_RECURRENT_NEURONS)})
else:
    recurrent_output_vars["g"] = None

# Optimiser initialisation
adam_params = {"beta1": 0.9, "beta2": 0.999, "epsilon": 1E-8}
adam_vars = {"m": 0.0, "v": 0.0}

# ----------------------------------------------------------------------------
# Model description
# ----------------------------------------------------------------------------
model = genn_model.GeNNModel("float", "tonic_classifier")
model.dT = TIMESTEP_MS
model.timing_enabled = TIMING_ENABLED


# Add neuron populations
input = model.add_neuron_population("Input", num_input_neurons, "SpikeSourceArray",
                                    {}, {"startSpike": None, "endSpike": None})
recurrent = model.add_neuron_population("Recurrent", NUM_RECURRENT_NEURONS, eprop.recurrent_alif_model,
                                        recurrent_params, recurrent_vars)
output = model.add_neuron_population("Output", NUM_OUTPUT_NEURONS, output_classification_model,
                                     output_params, output_vars)

#input.set_extra_global_param("spikeTimes", input_spikes)
#output.set_extra_global_param("spikeTimes", target_spikes["time"])

# Turn on recording
input.spike_recording_enabled = True
recurrent.spike_recording_enabled = True

# Add synapse populations
input_recurrent = model.add_synapse_population(
    "InputRecurrent", "DENSE_INDIVIDUALG", NO_DELAY,
    input, recurrent,
    eprop.eprop_alif_model, eprop_params, input_recurrent_vars, eprop_pre_vars, eprop_post_vars,
    "DeltaCurr", {}, {})

recurrent_recurrent = model.add_synapse_population(
    "RecurrentRecurrent", "DENSE_INDIVIDUALG", NO_DELAY,
    recurrent, recurrent,
    eprop.eprop_alif_model, eprop_params, recurrent_recurrent_vars, eprop_pre_vars, eprop_post_vars,
    "DeltaCurr", {}, {})

recurrent_output = model.add_synapse_population(
    "RecurrentOutput", "DENSE_INDIVIDUALG", NO_DELAY,
    recurrent, output,
    eprop.output_learning_model, recurrent_output_params, recurrent_output_vars, recurrent_output_pre_vars, {},
    "DeltaCurr", {}, {})

output_recurrent = model.add_synapse_population(
    "OutputRecurrent", "DENSE_INDIVIDUALG", NO_DELAY,
    output, recurrent,
    eprop.feedback_model, {}, {"g": 0.0}, {}, {},
    eprop.feedback_psm_model, {}, {})

# Add custom update for calculating initial tranpose weights
model.add_custom_update("recurrent_hidden_transpose", "CalculateTranspose", "Transpose",
                        {}, {}, {"variable": genn_model.create_wu_var_ref(recurrent_output, "g", output_recurrent, "g")})

# Add custom updates for updating weights using Adam optimiser
input_recurrent_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(input_recurrent, "DeltaG"),
                                      "variable": genn_model.create_wu_var_ref(input_recurrent, "g")}
input_recurrent_optimiser = model.add_custom_update("input_recurrent_optimiser", "GradientLearn", eprop.adam_optimizer_model,
                                                    adam_params, adam_vars, input_recurrent_optimiser_var_refs)

recurrent_recurrent_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_recurrent, "DeltaG"),
                                          "variable": genn_model.create_wu_var_ref(recurrent_recurrent, "g")}
recurrent_recurrent_optimiser = model.add_custom_update("recurrent_recurrent_optimiser", "GradientLearn", eprop.adam_optimizer_model,
                                                        adam_params, adam_vars, recurrent_recurrent_optimiser_var_refs)

recurrent_output_optimiser_var_refs = {"gradient": genn_model.create_wu_var_ref(recurrent_output, "DeltaG"),
                                       "variable": genn_model.create_wu_var_ref(recurrent_output, "g" , output_recurrent, "g")}
recurrent_output_optimiser = model.add_custom_update("recurrent_output_optimiser", "GradientLearn", eprop.adam_optimizer_model,
                                                     adam_params, adam_vars, recurrent_output_optimiser_var_refs)

output_bias_optimiser_var_refs = {"gradient": genn_model.create_var_ref(output, "DeltaB"),
                                  "variable": genn_model.create_var_ref(output, "B")}
output_bias_optimiser = model.add_custom_update("output_bias_optimiser", "GradientLearn", eprop.adam_optimizer_model,
                                                adam_params, adam_vars, output_bias_optimiser_var_refs)

batch_timesteps = int(np.ceil(((max_stimuli_duration + CUE_TIME) * BATCH_SIZE) / TIMESTEP_MS))

# Build and load model
model.build()
model.load(num_recording_timesteps=batch_timesteps)

# Calculate initial transpose feedback weights
model.custom_update("CalculateTranspose")
