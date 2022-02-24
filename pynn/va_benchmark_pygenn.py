import numpy as np
import matplotlib.pyplot as plt
from time import time

from pygenn import (GeNNModel, VarLocation, SpanType, init_var, 
                    init_sparse_connectivity)

# Parameters
TIMESTEP = 1.0
NUM_NEURONS = 4000

RESET_VOLTAGE = -60.0
THRESHOHOLD_VOLTAGE = -50.0

PROBABILITY_CONNECTION = 0.1

EXCITATORY_INHIBITORY_RATIO = 4.0

NUM_EXCITATORY = int(round((NUM_NEURONS * EXCITATORY_INHIBITORY_RATIO) / (1.0 + EXCITATORY_INHIBITORY_RATIO)))
NUM_INHIBITORY = NUM_NEURONS - NUM_EXCITATORY

SCALE = (4000.0 / NUM_NEURONS) * (0.02 / PROBABILITY_CONNECTION)

EXCITATORY_WEIGHT = 4.0E-3 * SCALE
INHIBITORY_WEIGHT = -51.0E-3 * SCALE

model = GeNNModel("float", "va_benchmark")
model.dt = TIMESTEP
model.default_narrow_sparse_ind_enabled = True
fixed_prob = {"prob": PROBABILITY_CONNECTION}

lif_params = {"C": 1.0, "TauM": 20.0, "Vrest": -49.0, "Vreset": RESET_VOLTAGE,
              "Vthresh": THRESHOHOLD_VOLTAGE, "Ioffset": 0.0, "TauRefrac": 5.0}

lif_init = {"V": init_var("Uniform", {"min": RESET_VOLTAGE, "max": THRESHOHOLD_VOLTAGE}),
            "RefracTime": 0.0}

excitatory_synapse_init = {"g": EXCITATORY_WEIGHT}
inhibitory_synapse_init = {"g": INHIBITORY_WEIGHT}

excitatory_post_syn_params = {"tau": 5.0}
inhibitory_post_syn_params = {"tau": 10.0}

excitatory_pop = model.add_neuron_population("E", NUM_EXCITATORY, "LIF", lif_params, lif_init)
inhibitory_pop = model.add_neuron_population("I", NUM_INHIBITORY, "LIF", lif_params, lif_init)

excitatory_pop.spike_recording_enabled = True

model.add_synapse_population("EE", "SPARSE_GLOBALG", 0,
    excitatory_pop, excitatory_pop,
    "StaticPulse", {}, excitatory_synapse_init, {}, {},
    "ExpCurr", excitatory_post_syn_params, {},
    init_sparse_connectivity("FixedProbabilityNoAutapse", fixed_prob))

model.add_synapse_population("EI", "SPARSE_GLOBALG", 0,
    excitatory_pop, inhibitory_pop,
    "StaticPulse", {}, excitatory_synapse_init, {}, {},
    "ExpCurr", excitatory_post_syn_params, {},
    init_sparse_connectivity("FixedProbability", fixed_prob))

model.add_synapse_population("II", "SPARSE_GLOBALG", 0,
    inhibitory_pop, inhibitory_pop,
    "StaticPulse", {}, inhibitory_synapse_init, {}, {},
    "ExpCurr", inhibitory_post_syn_params, {},
    init_sparse_connectivity("FixedProbabilityNoAutapse", fixed_prob))

model.add_synapse_population("IE", "SPARSE_GLOBALG", 0,
    inhibitory_pop, excitatory_pop,
    "StaticPulse", {}, inhibitory_synapse_init, {}, {},
    "ExpCurr", inhibitory_post_syn_params, {},
    init_sparse_connectivity("FixedProbability", fixed_prob))

print("Building Model")
model.build()
print("Loading Model")
model.load(num_recording_timesteps=10000)

sim_start_time = time()
while model.timestep < 10000:
    model.step_time()

sim_end_time = time()
print("Simulation time:%fs" % (sim_end_time - sim_start_time))

# Download recording data
model.pull_recording_buffers_from_device()

# Get recording data
spike_times, spike_ids = excitatory_pop.spike_recording_data

fig, axes = plt.subplots(2)

bin_size = 10.0
rate_bins = np.arange(0, 10000.0, bin_size)
rate = np.histogram(spike_times, bins=rate_bins)[0]
rate_bin_centres = rate_bins[:-1] + (bin_size / 2.0)

axes[0].scatter(spike_times, spike_ids, s=1)
axes[1].plot(rate_bin_centres, rate * (1000.0 / bin_size) * (1.0 / float(NUM_EXCITATORY)))

plt.show()
