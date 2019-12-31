import numpy as np 
import matplotlib.pyplot as plt 

from pygenn import genn_model, genn_wrapper
from scipy.stats import binom, norm
from six import iteritems, itervalues
from time import perf_counter

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
# Layer names
LAYER_NAMES = ["23", "4", "5", "6"]

MEASURE_TIMING = True

BUILD_MODEL = True

# Population names
POPULATION_NAMES = ["E", "I"]

# Simulation timestep [ms]
DT_MS = 0.1

# Simulation duration [ms]
DURATION_MS = 1000.0

# Scaling factors for number of neurons and synapses
NEURON_SCALING_FACTOR = 1.0
CONNECTIVITY_SCALING_FACTOR = 1.0

# Background rate per synapse
BACKGROUND_RATE = 8.0  # spikes/s

# Relative inhibitory synaptic weight
G = -4.0

# Mean synaptic weight for all excitatory projections except L4e->L2/3e
MEAN_W = 87.8e-3  # nA
EXTERNAL_W = 87.8e-3   # nA

# Mean synaptic weight for L4e->L2/3e connections
# See p. 801 of the paper, second paragraph under 'Model Parameterization',
# and the caption to Supplementary Fig. 7
LAYER_23_4_W = 2.0 * MEAN_W   # nA

# Standard deviation of weight distribution relative to mean for
# all projections except L4e->L2/3e
REL_W = 0.1

# Standard deviation of weight distribution relative to mean for L4e->L2/3e
# This value is not mentioned in the paper, but is chosen to match the
# original code by Tobias Potjans
LAYER_23_4_RELW = 0.05

# Numbers of neurons in full-scale model
NUM_NEURONS = {
    "23":   {"E":20683, "I": 5834},
    "4":    {"E":21915, "I": 5479},
    "5":    {"E":4850,  "I": 1065},
    "6":    {"E":14395, "I": 2948}}

# Probabilities for >=1 connection between neurons in the given populations.
# The first index is for the target population; the second for the source population
CONNECTION_PROBABILTIES = {
    "23E":  {"23E": 0.1009, "23I": 0.1689,  "4E": 0.0437,   "4I": 0.0818,   "5E": 0.0323,   "5I": 0.0,      "6E": 0.0076,   "6I": 0.0},
    "23I":  {"23E": 0.1346, "23I": 0.1371,  "4E": 0.0316,   "4I": 0.0515,   "5E": 0.0755,   "5I": 0.0,      "6E": 0.0042,   "6I": 0.0},
    "4E":   {"23E": 0.0077, "23I": 0.0059,  "4E": 0.0497,   "4I": 0.135,    "5E": 0.0067,   "5I": 0.0003,   "6E": 0.0453,   "6I": 0.0},
    "4I":   {"23E": 0.0691, "23I": 0.0029,  "4E": 0.0794,   "4I": 0.1597,   "5E": 0.0033,   "5I": 0.0,      "6E": 0.1057,   "6I": 0.0},
    "5E":   {"23E": 0.1004, "23I": 0.0622,  "4E": 0.0505,   "4I": 0.0057,   "5E": 0.0831,   "5I": 0.3726,   "6E": 0.0204,   "6I": 0.0},
    "5I":   {"23E": 0.0548, "23I": 0.0269,  "4E": 0.0257,   "4I": 0.0022,   "5E": 0.06,     "5I": 0.3158,   "6E": 0.0086,   "6I": 0.0},
    "6E":   {"23E": 0.0156, "23I": 0.0066,  "4E": 0.0211,   "4I": 0.0166,   "5E": 0.0572,   "5I": 0.0197,   "6E": 0.0396,   "6I": 0.2252},
    "6I":   {"23E": 0.0364, "23I": 0.001,   "4E": 0.0034,   "4I": 0.0005,   "5E": 0.0277,   "5I": 0.008,    "6E": 0.0658,   "6I": 0.1443}}
    

# In-degrees for external inputs
NUM_EXTERNAL_INPUTS = {
    "23":   {"E": 1600, "I": 1500},
    "4":    {"E": 2100, "I": 1900},
    "5":    {"E": 2000, "I": 1900},
    "6":    {"E": 2900, "I": 2100}}

# Mean rates in the full-scale model, necessary for scaling
# Precise values differ somewhat between network realizations
MEAN_FIRING_RATES = {
    "23":   {"E": 0.971,    "I": 2.868},
    "4":    {"E": 4.746,    "I": 5.396},
    "5":    {"E": 8.142,    "I": 9.078},
    "6":    {"E": 0.991,    "I": 7.523}}

# Means and standard deviations of delays from given source populations (ms)
MEAN_DELAY = {"E": 1.5, "I": 0.75}

DELAY_SD = {"E": 0.75, "I": 0.375}

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def get_scaled_num_neurons(layer, pop):
    return int(round(NEURON_SCALING_FACTOR * NUM_NEURONS[layer][pop]))

def get_full_num_inputs(src_layer, src_pop, trg_layer, trg_pop):
    num_src = NUM_NEURONS[src_layer][src_pop]
    num_trg = NUM_NEURONS[trg_layer][trg_pop]
    connection_prob = CONNECTION_PROBABILTIES[trg_layer + trg_pop][src_layer + src_pop]

    return int(round(np.log(1.0 - connection_prob) / np.log(float(num_trg * num_src - 1) / float(num_trg * num_src))) / num_trg)

def get_mean_weight(src_layer, src_pop, trg_layer, trg_pop):
    # Determine mean weight
    if src_pop == "E":
        if src_layer == "4" and trg_layer == "23" and trg_pop == "E":
            return LAYER_23_4_W
        else:
            return MEAN_W
    else:
        return G * MEAN_W

def get_scaled_num_connections(src_layer, src_pop, trg_layer, trg_pop):
    # Scale full number of inputs by scaling factor
    num_inputs = get_full_num_inputs(src_layer, src_pop, trg_layer, trg_pop) * CONNECTIVITY_SCALING_FACTOR
    assert num_inputs >= 0.0

    # Multiply this by number of postsynaptic neurons
    return int(round(num_inputs * float(get_scaled_num_neurons(trg_layer, trg_pop))))

def get_full_mean_input_current(layer, pop):
    # Loop through source populations
    mean_input_current = 0.0
    for src_layer in LAYER_NAMES:
        for src_pop in POPULATION_NAMES:
            mean_input_current += (get_mean_weight(src_layer, src_pop, layer, pop) *
                                   get_full_num_inputs(src_layer, src_pop, layer, pop) *
                                   MEAN_FIRING_RATES[src_layer][src_pop])

    # Add mean external input current
    mean_input_current += EXTERNAL_W * NUM_EXTERNAL_INPUTS[layer][pop] * BACKGROUND_RATE
    assert mean_input_current >= 0.0
    return mean_input_current

def build_row_lengths(num_pre, num_post, num_connections):
    remaining_connections = num_connections
    matrix_size = num_pre * num_post
    
    row_lengths = np.empty(num_pre, dtype=np.uint32)
    for i in range(num_pre - 1):
        probability = float(num_post) / float(matrix_size)
        
        # Sample row length;
        row_lengths[i] = binom.rvs(remaining_connections, probability)
        
        # Update counters
        remaining_connections -= row_lengths[i]
        matrix_size -= num_post
        
    # Insert remaining connections into last row
    row_lengths[num_pre - 1] = remaining_connections
    return row_lengths

# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
# LIF neuron model
lif_model = genn_model.create_custom_neuron_class(
    "lif",
    param_names=[
        "C",                # Membrane capacitance
        "TauM",             # Membrane time constant [ms]
        "Vrest",            # Resting membrane potential [mV]
        "Vreset",           # Reset voltage [mV]
        "Vthresh",          # Spiking threshold [mV]
        "Ioffset",          # Offset current
        "TauRefrac",        # Refractory time [ms]
        "PoissonRate",      # Poisson input rate [Hz]
        "PoissonWeight",    # How much current each poisson spike adds [nA]
        "IpoissonTau"],     # Time constant of poisson spike integration [ms]],
        
    var_name_types=[("V","scalar"), ("RefracTime", "scalar"), ("Ipoisson", "scalar")],
    derived_params=[
        ("ExpTC",                   genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[1]))()),
        ("Rmembrane",               genn_model.create_dpf_class(lambda pars, dt: pars[1] / pars[0])()),
        ("PoissonExpMinusLambda",   genn_model.create_dpf_class(lambda pars, dt: np.exp(-(pars[7] / 1000.0) * dt))()),
        ("IpoissonExpDecay",        genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[9]))()),
        ("IpoissonInit",            genn_model.create_dpf_class(lambda pars, dt: pars[8] * (1.0 - np.exp(-dt / pars[9])) * (pars[9] / dt))()),
    ],
    
    sim_code="""
    scalar p = 1.0f;
    unsigned int numPoissonSpikes = 0;
    do
    {
        numPoissonSpikes++;
        p *= $(gennrand_uniform);
    } while (p > $(PoissonExpMinusLambda));
    $(Ipoisson) += $(IpoissonInit) * (scalar)(numPoissonSpikes - 1);
    if ($(RefracTime) <= 0.0)
    {
      scalar alpha = (($(Isyn) + $(Ioffset) + $(Ipoisson)) * $(Rmembrane)) + $(Vrest);
      $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
    }
    else
    {
      $(RefracTime) -= DT;
    }
    $(Ipoisson) *= $(IpoissonExpDecay);
    """,
    
    reset_code="""
    $(V) = $(Vreset);
    $(RefracTime) = $(TauRefrac);
    """,
    threshold_condition_code="$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)")

normal_clipped_model = genn_model.create_custom_init_var_snippet_class(
    "normal_clipped",
    param_names=["mean", "sd", "min", "max"],
    var_init_code="""
    scalar normal;
    do
    {
       normal = $(mean) + ($(gennrand_normal) * $(sd));
    } while (normal > $(max) || normal < $(min));
    $(value) = normal;
    """)

normal_clipped_delay_model = genn_model.create_custom_init_var_snippet_class(
    "normal_clipped_delay",
    param_names=["mean", "sd", "min", "max"],
    var_init_code="""
    scalar normal;
    do
    {
       normal = $(mean) + ($(gennrand_normal) * $(sd));
    } while (normal > $(max) || normal < $(min));
    $(value) = rint(normal / DT);
    """)
    
fixed_num_total_with_replacement_model = genn_model.create_custom_sparse_connect_init_snippet_class(
    "fixed_num_total_with_replacement",
    param_names=["total"],
    row_build_state_vars=[("x", "scalar", 0.0), ("c", "unsigned int", 0)],
    extra_global_params=[("preCalcRowLength", "unsigned int*")],
    calc_max_row_len_func=genn_model.create_cmlf_class(
        lambda num_pre, num_post, pars: int(binom.ppf(0.9999**(1.0 / num_pre), n=pars[0], p=float(num_post) / (num_pre * num_post))))(),
    calc_max_col_len_func=genn_model.create_cmlf_class(
        lambda num_pre, num_post, pars: int(binom.ppf(0.9999**(1.0 / num_post), n=pars[0], p=float(num_pre) / (num_pre * num_post))))(),
    row_build_code="""
    const unsigned int rowLength = $(preCalcRowLength)[($(id_pre) * $(num_threads)) + $(id_thread)];
    if(c >= rowLength) {
       $(endRow);
    }
    const scalar u = $(gennrand_uniform);
    x += (1.0 - x) * (1.0 - pow(u, 1.0 / (scalar)(rowLength - c)));
    unsigned int postIdx = (unsigned int)(x * $(num_post));
    postIdx = (postIdx < $(num_post)) ? postIdx : ($(num_post) - 1);
    $(addSynapse, postIdx + $(id_post_begin));
    c++;
    """)

# ----------------------------------------------------------------------------
# Network creation
# ----------------------------------------------------------------------------
model = genn_model.GeNNModel("float", "potjans_microcircuit")
model.dT = DT_MS
model._model.set_merge_postsynaptic_models(True)
model.timing_enabled = MEASURE_TIMING
model.default_var_location = genn_wrapper.VarLocation_DEVICE
model.default_sparse_connectivity_location = genn_wrapper.VarLocation_DEVICE

lif_init = {"V": genn_model.init_var("Normal", {"mean": -58.0, "sd": 5.0}),
            "RefracTime": 0.0, "Ipoisson": 0.0}

exp_curr_params = {"tau": 0.5}

quantile = 0.9999
normal_quantile_cdf = norm.ppf(quantile)
max_delay = {pop: MEAN_DELAY[pop] + (DELAY_SD[pop] * normal_quantile_cdf)
             for pop in POPULATION_NAMES}
print("Max excitatory delay:%fms , max inhibitory delay:%fms" % (max_delay["E"], max_delay["I"]))

# Calculate maximum dendritic delay slots
# **NOTE** it seems inefficient using maximum for all but this allows more aggressive merging of postsynaptic models
max_dendritic_delay_slots = int(round(max(itervalues(max_delay)) / DT_MS))
print("Max dendritic delay slots:%d" % max_dendritic_delay_slots)

print("Creating neuron populations:")
total_neurons = 0
neuron_populations = {}
for layer in LAYER_NAMES:
    for pop in POPULATION_NAMES:
        pop_name = layer + pop
        
        # Calculate external input rate, weight and current
        ext_input_rate = NUM_EXTERNAL_INPUTS[layer][pop] * CONNECTIVITY_SCALING_FACTOR * BACKGROUND_RATE
        ext_weight = EXTERNAL_W / np.sqrt(CONNECTIVITY_SCALING_FACTOR)
        ext_input_current = 0.001 * 0.5 * (1.0 - np.sqrt(CONNECTIVITY_SCALING_FACTOR)) * get_full_mean_input_current(layer, pop)
        assert ext_input_current >= 0.0
            
        lif_params = {"C": 0.25, "TauM": 10.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh" : -50.0,
                      "Ioffset": ext_input_current, "TauRefrac": 2.0, "PoissonRate": ext_input_rate, 
                      "PoissonWeight": ext_weight, "IpoissonTau": 0.5}
        
        pop_size = get_scaled_num_neurons(layer, pop)
        neuron_pop = model.add_neuron_population(pop_name, pop_size, lif_model, lif_params, lif_init)
        
        neuron_pop.pop.set_spike_location(genn_wrapper.VarLocation_HOST_DEVICE)
        
        print("\tPopulation %s: num neurons:%u, external input rate:%f, external weight:%f, external DC offset:%f" % (pop_name, pop_size, ext_input_rate, ext_weight, ext_input_current))
        
        # Add number of neurons to total
        total_neurons += pop_size
        
        # Add neuron population to dictionary
        neuron_populations[pop_name] = neuron_pop
 
# Loop through target populations and layers
print("Creating synapse populations:")
total_synapses = 0
for trg_layer in LAYER_NAMES:
    for trg_pop in POPULATION_NAMES:
        trg_name = trg_layer + trg_pop
        
        # Loop through source populations and layers
        for src_layer in LAYER_NAMES:
            for src_pop in POPULATION_NAMES:
                src_name = src_layer + src_pop

                # Determine mean weight
                mean_weight = get_mean_weight(src_layer, src_pop, trg_layer, trg_pop) / np.sqrt(CONNECTIVITY_SCALING_FACTOR)

                # Determine weight standard deviation
                if src_pop == "E" and src_layer == "4" and trg_layer == "23" and trg_pop == "E":
                    weight_sd = mean_weight * LAYER_23_4_RELW
                else:
                    weight_sd = abs(mean_weight * REL_W)
                
                # Calculate number of connections
                num_connections = get_scaled_num_connections(src_layer, src_pop, trg_layer, trg_pop)

                if num_connections > 0:
                    num_src_neurons = get_scaled_num_neurons(src_layer, src_pop)
                    num_trg_neurons = get_scaled_num_neurons(trg_layer, trg_pop)
               
                    print("\tConnection between '%s' and '%s': numConnections=%u, meanWeight=%f, weightSD=%f, meanDelay=%f, delaySD=%f" 
                          % (src_name, trg_name, num_connections, mean_weight, weight_sd, MEAN_DELAY[src_pop], DELAY_SD[src_pop]))

                    # Build parameters for fixed number total connector
                    connect_params = {"total": num_connections}
                    
                    # Build distribution for delay parameters
                    d_dist = {"mean": MEAN_DELAY[src_pop], "sd": DELAY_SD[src_pop], "min": 0.0, "max": max_delay[src_pop]}

                    total_synapses += num_connections
                    
                    # Build unique synapse name
                    synapse_name = src_name + "_" + trg_name

                    # Excitatory
                    if src_pop == "E":
                        # Build distribution for weight parameters
                        # **HACK** np.float32 doesn't seem to automatically cast 
                        w_dist = {"mean": mean_weight, "sd": weight_sd, "min": 0.0, "max": float(np.finfo(np.float32).max)}
                        
                        # Create weight parameters
                        static_synapse_init = {"g": genn_model.init_var(normal_clipped_model, w_dist),
                                               "d": genn_model.init_var(normal_clipped_delay_model, d_dist)}

                        # Add synapse population
                        syn_pop = model.add_synapse_population(synapse_name, "SPARSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                            neuron_populations[src_name], neuron_populations[trg_name],
                            "StaticPulseDendriticDelay", {}, static_synapse_init, {}, {},
                            "ExpCurr", exp_curr_params, {},
                            genn_model.init_connectivity(fixed_num_total_with_replacement_model, connect_params))
                        
                        # Add extra global parameter with row lengths
                        syn_pop.add_connectivity_extra_global_param(
                            "preCalcRowLength", build_row_lengths(num_src_neurons, num_trg_neurons, num_connections))
                                                       
                        # Set max dendritic delay and span type
                        syn_pop.pop.set_max_dendritic_delay_timesteps(max_dendritic_delay_slots)

                        #synPop->setSpanType(Parameters::presynapticParallelism ? SynapseGroup::SpanType::PRESYNAPTIC : SynapseGroup::SpanType::POSTSYNAPTIC)
                        #if(Parameters::presynapticParallelism) {
                        #    synPop->setNumThreadsPerSpike(4)
                        #}
                    # Inhibitory
                    else:
                        # Build distribution for weight parameters
                        # **HACK** np.float32 doesn't seem to automatically cast 
                        w_dist = {"mean": mean_weight, "sd": weight_sd, "min": float(-np.finfo(np.float32).max), "max": 0.0}
                        
                        # Create weight parameters
                        static_synapse_init = {"g": genn_model.init_var(normal_clipped_model, w_dist),
                                               "d": genn_model.init_var(normal_clipped_delay_model, d_dist)}
                        
                        # Add synapse population
                        syn_pop = model.add_synapse_population(synapse_name, "SPARSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                            neuron_populations[src_name], neuron_populations[trg_name],
                            "StaticPulseDendriticDelay", {}, static_synapse_init, {}, {},
                            "ExpCurr", exp_curr_params, {},
                            genn_model.init_connectivity(fixed_num_total_with_replacement_model, connect_params))
                        
                        # Add extra global parameter with row lengths
                        syn_pop.add_connectivity_extra_global_param(
                            "preCalcRowLength", build_row_lengths(num_src_neurons, num_trg_neurons, num_connections))

                        # Set max dendritic delay and span type
                        syn_pop.pop.set_max_dendritic_delay_timesteps(max_dendritic_delay_slots)
                        #synPop->setSpanType(Parameters::presynapticParallelism ? SynapseGroup::SpanType::PRESYNAPTIC : SynapseGroup::SpanType::POSTSYNAPTIC)
                        #if(Parameters::presynapticParallelism) {
                        #    synPop->setNumThreadsPerSpike(4)
                        #}
print("Total neurons=%u, total synapses=%u" % (total_neurons, total_synapses))

if BUILD_MODEL:
    print("Building Model")
    model.build()
print("Loading Model")
model.load()

print("Simulating")
duration_timesteps = int(round(DURATION_MS / DT_MS))
ten_percent_timestep = duration_timesteps // 10

# Create dictionary to hold spikes for each population
pop_spikes = [[pop, np.empty(0), np.empty(0)] 
              for pop in itervalues(neuron_populations)]

# Loop through timesteps
sim_start_time = perf_counter()
while model.t < DURATION_MS:
    # Advance simulation
    model.step_time()
    
    # Indicate every 10%
    if (model.timestep % ten_percent_timestep) == 0:
        print("%u%%" % (model.timestep / 100))
    
    for i, spikes in enumerate(pop_spikes):
        # Download spikes
        model.pull_current_spikes_from_device(spikes[0].name)

        # Add to data structure
        spike_times = np.ones_like(spikes[0].current_spikes) * model.t
        pop_spikes[i][1] = np.hstack((pop_spikes[i][1], spikes[0].current_spikes))
        pop_spikes[i][2] = np.hstack((pop_spikes[i][2], spike_times))

sim_end_time =  perf_counter()

print("Timing:")
print("\tSimulation:%f" % ((sim_end_time - sim_start_time) * 1000.0))
    
if MEASURE_TIMING:
    print("\tInit:%f" % (1000.0 * model.init_time))
    print("\tSparse init:%f" % (1000.0 * model.init_sparse_time))
    print("\tNeuron simulation:%f" % (1000.0 * model.neuron_update_time))
    print("\tSynapse simulation:%f" % (1000.0 * model.presynaptic_update_time))

# Create plot
figure, axes = plt.subplots(1, 2)

start_id = 0
bar_y = 0.0
for pop, i, t in reversed(pop_spikes):
    # Plot spikes
    actor = axes[0].scatter(t, i + start_id, s=2, edgecolors="none")

    # Plot bar showing rate in matching colour
    axes[1].barh(bar_y, len(t) / (float(pop.size) * DURATION_MS / 1000.0), 
                 align="center", color=actor.get_facecolor(), ecolor="black")

    # Update offset
    start_id += pop.size

    # Update bar pos
    bar_y += 1.0


axes[0].set_xlabel("Time [ms]")
axes[0].set_ylabel("Neuron number")

axes[1].set_xlabel("Mean firing rate [Hz]")
axes[1].set_yticks(np.arange(0.0, len(pop_spikes) * 1.0, 1.0))
axes[1].set_yticklabels([s[0].name for s in pop_spikes])

# Show plot
plt.show()