import numpy as np

from pygenn import genn_model
from pygenn.genn_wrapper import NO_DELAY

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def calc_t_peak(tau_rise, tau_decay):
    return ((tau_decay * tau_rise) / (tau_decay - tau_rise)) * np.log(tau_decay / tau_rise)

# ----------------------------------------------------------------------------
# Custom models
# ----------------------------------------------------------------------------
r_max_prop_model = genn_model.create_custom_custom_update_class(
    "r_max_prop",
    param_names=["updateTime", "tauRMS", "epsilon", "wMin", "wMax"],
    var_name_types=[("upsilon", "scalar")],
    extra_global_params=[("r0", "scalar")],
    derived_params=[("updateTimesteps", genn_model.create_dpf_class(lambda pars, dt: pars[0] / dt)()),
                    ("expRMS", genn_model.create_dpf_class(lambda pars, dt: np.exp(-pars[0] / pars[1]))())],
    var_refs=[("m", "scalar"), ("variable", "scalar")],
    update_code="""
    // Get gradients
    const scalar gradient = $(m) / $(updateTimesteps);
    // Calculate learning rate r
    $(upsilon) = fmax($(upsilon) * $(expRMS), gradient * gradient);
    const scalar r = $(r0) / (sqrt($(upsilon)) + $(epsilon));
    // Update synaptic parameter
    $(variable) += r * gradient;
    $(variable) = fmin($(wMax), fmax($(wMin), $(variable)));
    $(m) = 0.0;
    """)
    
superspike_model = genn_model.create_custom_weight_update_class(
    "superspike",
    param_names=["tauRise", "tauDecay", "beta"],
    var_name_types=[("w", "scalar"), ("e", "scalar"), 
                    ("lambda", "scalar"), ("m", "scalar")],
    pre_var_name_types=[("z", "scalar"), ("zTilda", "scalar")],
    post_var_name_types=[("sigmaPrime", "scalar")],
    
    sim_code="""
    $(addToInSyn, $(w));
    """,
    
    pre_spike_code="""
    $(z) += 1.0;
    """,
    pre_dynamics_code="""
    // filtered presynaptic trace
    $(z) += (-$(z) / $(tauRise)) * DT;
    $(zTilda) += ((-$(zTilda) + $(z)) / $(tauDecay)) * DT;
    """,
    
    post_dynamics_code="""
    // filtered partial derivative
    if($(V_post) < -80.0) {
       $(sigmaPrime) = 0.0;
    }
    else {
       const scalar onePlusHi = 1.0 + fabs($(beta) * 0.001 * ($(V_post) - $(Vthresh_post)));
       $(sigmaPrime) = $(beta) / (onePlusHi * onePlusHi);
    }
    """,
    
    synapse_dynamics_code="""
    // Filtered eligibility trace
    $(e) += ($(zTilda) * $(sigmaPrime) - $(e) / $(tauRise))*DT;
    $(lambda) += ((-$(lambda) + $(e)) / $(tauDecay)) * DT;
    // Get error from neuron model and compute full
    // expression under integral and calculate m
    $(m) += $(lambda) * $(errTilda_post);
    """)

feedback_model = genn_model.create_custom_weight_update_class(
    "feedback",
    var_name_types=[("w", "scalar")],
    synapse_dynamics_code="""
    $(addToInSyn, $(w) * $(errTilda_pre));
    """)
    
feedback_psm_model = genn_model.create_custom_postsynaptic_class(
    "feedback_psm",
    apply_input_code="""
    $(ISynFeedback) += $(inSyn);
    $(inSyn) = 0;
    """)

hidden_neuron_model = genn_model.create_custom_neuron_class(
    "hidden",
    param_names=["C", "tauMem", "Vrest", "Vthresh", "tauRefrac"],
    var_name_types=[("V", "scalar"), ("refracTime", "scalar"), ("errTilda", "scalar")],
    additional_input_vars=[("ISynFeedback", "scalar", 0.0)],
    derived_params=[("ExpTC", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[1]))()),
                    ("Rmembrane", genn_model.create_dpf_class(lambda pars, dt: pars[1] / pars[0])())],
   
    sim_code="""
    // membrane potential dynamics
    if ($(refracTime) == $(tauRefrac)) {
        $(V) = $(Vrest);
    }
    if ($(refracTime) <= 0.0) {
        scalar alpha = ($(Isyn) * $(Rmembrane)) + $(Vrest);
        $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
    }
    else {
        $(refracTime) -= DT;
    }
    // error
    $(errTilda) = $(ISynFeedback);
    """,
    reset_code="""
    $(refracTime) = $(tauRefrac);
    """,
    threshold_condition_code="""
    $(refracTime) <= 0.0 && $(V) >= $(Vthresh)
    """,
    is_auto_refractory_required=False)
    
    
output_neuron_model = genn_model.create_custom_neuron_class(
    "output",
    param_names=["C", "tauMem", "Vrest", "Vthresh", "tauRefrac",
                 "tauRise", "tauDecay", "tauAvgErr"],
    var_name_types=[("V", "scalar"), ("refracTime", "scalar"), ("errRise", "scalar"),
                    ("errTilda", "scalar"), ("avgSqrErr", "scalar"), ("errDecay", "scalar"),
                    ("startSpike", "unsigned int"), ("endSpike", "unsigned int")],
    extra_global_params=[("spikeTimes", "scalar*")],
    derived_params=[("ExpTC", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[1]))()),
                    ("Rmembrane", genn_model.create_dpf_class(lambda pars, dt: pars[1] / pars[0])()),
                    ("normFactor", genn_model.create_dpf_class(lambda pars, dt: 1.0 / (-np.exp(-calc_t_peak(pars[5], pars[6]) / pars[5]) + np.exp(-calc_t_peak(pars[5], pars[6]) / pars[6])))()),
                    ("tRiseMult", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[5]))()),
                    ("tDecayMult", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[6]))()),
                    ("tPeak", genn_model.create_dpf_class(lambda pars, dt: calc_t_peak(pars[5], pars[6]))()),
                    ("mulAvgErr", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[7]))())],
   
    sim_code="""
    // membrane potential dynamics
    if ($(refracTime) == $(tauRefrac)) {
        $(V) = $(Vrest);
    }
    if ($(refracTime) <= 0.0) {
        scalar alpha = ($(Isyn) * $(Rmembrane)) + $(Vrest);
        $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
    }
    else {
        $(refracTime) -= DT;
    }
    // error
    scalar sPred = 0.0;
    if ($(startSpike) != $(endSpike) && $(t) >= $(spikeTimes)[$(startSpike)]) {
        $(startSpike)++;
        sPred = 1.0;
    }
    const scalar sReal = ($(refracTime) <= 0.0 && $(V) >= $(Vthresh)) ? 1.0 : 0.0;
    const scalar mismatch = sPred - sReal;
    $(errRise) = ($(errRise) * $(tRiseMult)) + mismatch;
    $(errDecay) = ($(errDecay) * $(tDecayMult)) + mismatch;
    $(errTilda) = ($(errDecay) - $(errRise)) * $(normFactor);
    // calculate average error trace
    const scalar temp = $(errTilda) * $(errTilda) * DT * 0.001;
    $(avgSqrErr) *= $(mulAvgErr);
    $(avgSqrErr) += temp;
    """,
    reset_code="""
    $(refracTime) = $(tauRefrac);
    """,
    threshold_condition_code="""
    $(refracTime) <= 0.0 && $(V) >= $(Vthresh)
    """,
    is_auto_refractory_required=False)

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
TIMESTEP_MS = 0.1
BUILD = True

# Network structure
NUM_INPUT = 200
NUM_OUTPUT = 200
NUM_HIDDEN = 256

# Model parameters
TAU_RISE_MS = 5.0
TAU_DECAY_MS = 10.0
TAU_RMS_MS = 30000.0
TAU_AVG_ERR_MS = 10000.0
R0 = 0.001 * 1000.0
EPSILON = 1E-32

# Weights
# **NOTE** Auryn units are volts, seconds etc so essentially 1000x GeNN parameters
W_MIN = -0.1 * 1000.0
W_MAX = 0.1 * 1000.0
W0 = 0.05 * 1000.0

# Experiment parameters
INPUT_FREQ_HZ = 5.0
NUM_TRIALS = 600
UPDATE_TIME_MS = 500.0
TRIAL_MS = 1890.0

# Convert parameters to timesteps
UPDATE_TIMESTEPS = int(UPDATE_TIME_MS / TIMESTEP_MS)
TRIAL_TIMESTEPS = int(TRIAL_MS / TIMESTEP_MS)

# ----------------------------------------------------------------------------
# Load target data
# ----------------------------------------------------------------------------
# Load target data
target_spikes = np.loadtxt("oxford-target.ras",
                           dtype={"names": ("time", "neuron_id"),
                                  "formats": (np.float, np.int)})

# Make neuron IDs zero-based
target_spikes["neuron_id"] -= 1

# Convert times to milliseconds
target_spikes["time"] *= 1000.0

# Sort first by neuron id and then by time
target_spikes = np.sort(target_spikes, order=["neuron_id", "time"])

# Count number of spikes
target_neuron_end_times = np.cumsum(np.bincount(target_spikes["neuron_id"], minlength=NUM_OUTPUT))
target_neuron_start_times = np.concatenate(([0], target_neuron_end_times[:-1]))

# ----------------------------------------------------------------------------
# Generate frozen poisson input
# ----------------------------------------------------------------------------
input_isi_ms = 1000.0 / INPUT_FREQ_HZ

# Generate time of first spike for each neuron
input_spike_times = input_isi_ms * np.random.exponential(size=NUM_INPUT)
input_spike_times = np.reshape(input_spike_times, (1, NUM_INPUT))

while True:
    # Generate vector of spike times
    s = input_isi_ms * np.random.exponential(size=NUM_INPUT)
    
    # Add previous times
    s += input_spike_times[-1,:]
    
    # If all neurons have reached end of trial
    if np.all(s >= TRIAL_MS):
        break
    # Otherwise stack
    else:        
        input_spike_times = np.vstack((input_spike_times, s))

# Count spikes per input neuron
input_spikes_per_neuron = np.sum(input_spike_times < TRIAL_MS, axis=0)

# Concatenate spikes within trial together
input_spikes = np.concatenate([input_spike_times[:input_spikes_per_neuron[i],i] 
                               for i in range(NUM_INPUT)])

# Calculate indices
input_neuron_end_times = np.cumsum(input_spikes_per_neuron)
input_neuron_start_times = np.concatenate(([0], input_neuron_end_times[:-1]))

# ----------------------------------------------------------------------------
# Neuron initialisation
# ----------------------------------------------------------------------------
input_init_vars = {"startSpike": input_neuron_start_times, "endSpike": input_neuron_end_times}

hidden_params = {"C" : 10.0, "tauMem": 10.0, "Vrest": -60.0, 
                 "Vthresh": -50.0 , "tauRefrac": 5.0}
hidden_init_vars = {"V": -60.0, "refracTime": 0.0, "errTilda": 0.0}

output_params = {"C": 10.0, "tauMem": 10.0, "Vrest": -60.0, 
                 "Vthresh": -50.0, "tauRefrac": 5.0, "tauRise": TAU_RISE_MS, 
                 "tauDecay": TAU_DECAY_MS, "tauAvgErr": TAU_AVG_ERR_MS}
output_init_vars = {"V": -60.0, "refracTime": 0.0, "errRise": 0.0, 
                    "errTilda": 0.0, "errDecay": 0.0, "avgSqrErr": 0.0,
                    "startSpike": target_neuron_start_times, "endSpike": target_neuron_end_times}

# ----------------------------------------------------------------------------
# Synapse initialisation
# ----------------------------------------------------------------------------
superspike_params = {"tauRise": TAU_RISE_MS, "tauDecay": TAU_DECAY_MS, "beta": 1000.0}
superspike_pre_init_vars = {"z": 0.0, "zTilda": 0.0}
superspike_post_init_vars = {"sigmaPrime": 0.0}

input_hidden_weight_dist_params = {"mean": 0.0, "sd": W0 / np.sqrt(float(NUM_INPUT)),
                                   "min": W_MIN, "max": W_MAX}
input_hidden_init_vars = {"w": genn_model.init_var("NormalClipped", input_hidden_weight_dist_params),
                          "e": 0.0, "lambda": 0.0, "m": 0.0}

hidden_output_weight_dist_params = {"mean": 0.0, "sd": W0 / np.sqrt(float(NUM_HIDDEN)),
                                    "min": W_MIN, "max": W_MAX}
hidden_output_init_vars = {"w": genn_model.init_var("NormalClipped", hidden_output_weight_dist_params),
                           "e": 0.0, "lambda": 0.0, "m": 0.0}     
       
# ----------------------------------------------------------------------------
# Custom update initialisation
# ----------------------------------------------------------------------------
r_max_prop_params = {"updateTime": UPDATE_TIME_MS, "tauRMS": TAU_RMS_MS, 
                     "epsilon": EPSILON, "wMin": W_MIN, "wMax": W_MAX}
       
# ----------------------------------------------------------------------------
# Model description
# ----------------------------------------------------------------------------
model = genn_model.GeNNModel("float", "superspike_demo", generateLineInfo=True)
model.dT = TIMESTEP_MS

# Add neuron populations
input = model.add_neuron_population("Input", NUM_INPUT, "SpikeSourceArray", 
                                    {}, input_init_vars)
hidden = model.add_neuron_population("Hidden", NUM_HIDDEN, hidden_neuron_model, 
                                     hidden_params, hidden_init_vars)
output = model.add_neuron_population("Output", NUM_OUTPUT, output_neuron_model, 
                                     output_params, output_init_vars)

input.set_extra_global_param("spikeTimes", input_spikes)
output.set_extra_global_param("spikeTimes", target_spikes["time"])

# Turn on recording
input.spike_recording_enabled = True
hidden.spike_recording_enabled = True
output.spike_recording_enabled = True

# Add synapse populations
input_hidden = model.add_synapse_population(
    "InputHidden", "DENSE_INDIVIDUALG", NO_DELAY,
    input, hidden,
    superspike_model, superspike_params, input_hidden_init_vars, superspike_pre_init_vars, superspike_post_init_vars,
    "ExpCurr", {"tau": 5.0}, {})

hidden_output = model.add_synapse_population(
    "HiddenOutput", "DENSE_INDIVIDUALG", NO_DELAY,
    hidden, output,
    superspike_model, superspike_params, hidden_output_init_vars, superspike_pre_init_vars, superspike_post_init_vars,
    "ExpCurr", {"tau": 5.0}, {})

output_hidden = model.add_synapse_population(
    "OutputHidden", "DENSE_INDIVIDUALG", NO_DELAY,
    output, hidden,
    feedback_model, {}, {"w": 0.0}, {}, {},
    feedback_psm_model, {}, {})

# Add custom update for calculating initial tranpose weights
model.add_custom_update("input_hidden_transpose", "CalculateTranspose", "Transpose",
                        {}, {}, {"variable": genn_model.create_wu_var_ref(hidden_output, "w", output_hidden, "w")})

# Add custom updates for gradient update
input_hidden_optimiser_var_refs = {"m": genn_model.create_wu_var_ref(input_hidden, "m"), 
                                   "variable": genn_model.create_wu_var_ref(input_hidden, "w")}
input_hidden_optimiser = model.add_custom_update("input_hidden_optimiser", "GradientLearn", r_max_prop_model,
                                                 r_max_prop_params, {"upsilon": 0.0}, input_hidden_optimiser_var_refs)

hidden_output_optimiser_var_refs = {"m": genn_model.create_wu_var_ref(hidden_output, "m"), 
                                   "variable": genn_model.create_wu_var_ref(hidden_output, "w", output_hidden, "w")}
hidden_output_optimiser = model.add_custom_update("hidden_output_optimiser", "GradientLearn", r_max_prop_model,
                                                  r_max_prop_params, {"upsilon": 0.0}, hidden_output_optimiser_var_refs)

# Set initial learning rate
input_hidden_optimiser.set_extra_global_param("r0", R0)
hidden_output_optimiser.set_extra_global_param("r0", R0)

# Build and load model
if BUILD:
    model.build()
model.load(num_recording_timesteps=TRIAL_TIMESTEPS)

# Calculate initial transpose feedback weights
model.custom_update("CalculateTranspose")

input_hidden_r0_view = input_hidden_optimiser.extra_global_params["r0"].view
hidden_output_r0_view = hidden_output_optimiser.extra_global_params["r0"].view

# Loop through trials
timestep = 0
for trial in range(NUM_TRIALS):
    # Reduce learning rate every 400 trials
    if trial != 0 and (trial % 400) == 0:
        input_hidden_r0_view[:] *= 0.1
        hidden_output_r0_view[:] *= 0.1
    
    # Display trial number peridically
    if trial != 0 and (trial % 10) == 0:
        # Get average square error
        output.pull_var_from_device("avgSqrErr")
        
        print("Trial %u (r0 = %f)" % (trial, input_hidden_r0_view[0]))
        #std::cout << "Trial " << trial << " (r0 = " << r0HiddenOutputWeightOptimiser << ", error = " << calculateError(timestep) << ")" << std::endl;

    # Reset model timestep
    # **NOTE** this a bit gross but means we can simplify a lot of logic
    model.t = 0.0
    model.timestep = 0

    # Loop through timesteps within trial
    for i in range(TRIAL_TIMESTEPS):
        model.step_time()

        # If it's time to update weights
        if timestep != 0 and (timestep % UPDATE_TIMESTEPS) == 0:
            model.custom_update("GradientLearn");
        timestep+=1;


    # Reset spike sources by re-uploading starting spike indices
    # **TODO** build repeating spike source array
    input.push_var_to_device("startSpike")
    output.push_var_to_device("startSpike")
    
    if (trial % 100) == 0:
        model.pull_recording_buffers_from_device();
        
        np.savetxt("input_spikes_%u.csv" % trial, input.spike_recording_data)
        np.savetxt("hidden_spikes_%u.csv" % trial, hidden.spike_recording_data)
        np.savetxt("output_spikes_%u.csv" % trial, output.spike_recording_data)
