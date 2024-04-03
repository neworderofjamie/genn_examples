import numpy as np
import matplotlib.pyplot as plt
from time import time

from pygenn import GeNNModel
from pygenn import (create_weight_update_model, init_postsynaptic,
                    init_sparse_connectivity, init_weight_update)

# Additive STDP model with nearest neighbour spike pairing
hebbian_stdp_model = create_weight_update_model(
    "hebbian_stdp",
    params=["tauPlus", "tauMinus", "aPlus", "aMinus", "wMin", "wMax"],
    vars=[("g", "scalar")],
    
    # Code that gets called whenever a presynaptic spike arrives at the synapse
    pre_spike_syn_code=
        """
        addToPost(g);
        const scalar dt = t - st_post;
        if(dt > 0) {
            const scalar newWeight = g - (aMinus * exp(-dt / tauMinus));
            g = fmin(wMax, fmax(wMin, newWeight));
        }
        """,

    # Code that gets called whenever a back-propagated postsynaptic spike arrives at the synapse
    post_spike_syn_code=
        """
        const scalar dt = t - st_pre;
        if(dt > 0) {
            const scalar newWeight = g + (aPlus * exp(-dt / tauPlus));
            g = fmin(wMax, fmax(wMin, newWeight));
        }
        """)

# Model parameters
NUM_NEURONS = 14
NUM_SPIKES = 60
START_TIME = 200.0
TIME_BETWEEN_PAIRS = 1000.0
DELTA_T = [-100.0, -60.0, -40.0, -30.0, -20.0, -10.0, -1.0,
           1.0, 10.0, 20.0, 30.0, 40.0, 60.0, 100.0]

# LIF neuron parameters
lif_params = {"C": 1.0, "TauM": 20.0, "Vrest": -70.0, "Vreset": -70.0,
              "Vthresh": -51.0, "Ioffset": 0.0, "TauRefrac": 2.0}
 
# LIF neuron initial state
lif_init = {"V": -70.0, "RefracTime": 0.0}
 
# STDP parameters
stdp_params =  {"tauPlus": 16.7, "tauMinus": 33.7, "aPlus": 0.005,
                "aMinus": 0.005,  "wMin": 0.0, "wMax": 1.0}

# Initial state for spike sources - each one emits NUM_SPIKES spikes
stim_init = {"startSpike": np.arange(0, NUM_NEURONS * NUM_SPIKES, NUM_SPIKES, dtype=int),
             "endSpike": np.arange(NUM_SPIKES, (NUM_NEURONS + 1) * NUM_SPIKES, NUM_SPIKES, dtype=int)}

# Calculate spike times
pre_phase = [START_TIME + d + 1.0 if d > 0 else START_TIME + 1.0 for d in DELTA_T]
post_phase = [START_TIME if d > 0 else START_TIME - d for d in DELTA_T]
pre_stim_spike_times = np.concatenate([p + np.arange(0, TIME_BETWEEN_PAIRS * NUM_SPIKES, TIME_BETWEEN_PAIRS) 
                                       for p in pre_phase])
post_stim_spike_times = np.concatenate([p + np.arange(0, TIME_BETWEEN_PAIRS * NUM_SPIKES, TIME_BETWEEN_PAIRS) 
                                       for p in post_phase])

# Create model using single-precion and 1ms timesteps
model = GeNNModel("float", "stdp_curve_pygenn")
model.dt = 1.0

# Add a neuron population and two spike sources to provide pre and postsynaptic stimuli
neuron_pop = model.add_neuron_population("Pop", NUM_NEURONS, "LIF", lif_params, lif_init)
pre_stim_pop = model.add_neuron_population("PreStim", NUM_NEURONS, "SpikeSourceArray", {}, stim_init)
post_stim_pop = model.add_neuron_population("PostStim", NUM_NEURONS, "SpikeSourceArray", {}, stim_init)

# Set spike source spike times
pre_stim_pop.extra_global_params["spikeTimes"].set_init_values(pre_stim_spike_times)
post_stim_pop.extra_global_params["spikeTimes"].set_init_values(post_stim_spike_times)

# Add STDP connection between presynaptic spike source and neurons
# Uses build in one-to-one connectivity and initialises all weights to 0.5 (midway between wMin and wMax)
pre_stim_to_pop = model.add_synapse_population("PreStimToPop", "SPARSE",
    pre_stim_pop, neuron_pop,
    init_weight_update(hebbian_stdp_model, stdp_params, {"g": 0.5}),
    init_postsynaptic("DeltaCurr"),
    init_sparse_connectivity("OneToOne"))

# Add static connection between postsynaptic spike source and neurons
# Uses built in one-to-one connectivity and initialises all weights to large value to cause immediate spikes
model.add_synapse_population("PostStimToPop", "SPARSE",
    post_stim_pop, neuron_pop,
    init_weight_update("StaticPulseConstantWeight", {"g": 8.0}),
    init_postsynaptic("ExpCurr", {"tau": 5.0}),
    init_sparse_connectivity("OneToOne", {}))

# Build and load model
model.build()
model.load()

# Simulate model
while model.t < 60200.0:
    model.step_time()

# Download weight and connectivity from GPU and access via synapse group
# **NOTE** because connectivity is initialised on device it also needs downloading
pre_stim_to_pop.pull_connectivity_from_device()
g = pre_stim_to_pop.vars["g"]
g.pull_from_device()

# Scale weights relative to initial value
weight = (g.current_values - 0.5) / 0.5

# Create plot
figure, axis = plt.subplots()

# Add axis lines
axis.axhline(0.0, color="black")
axis.axvline(0.0, color="black")

# Plot voltages
axis.plot(DELTA_T, weight)

# Show plot
plt.show()