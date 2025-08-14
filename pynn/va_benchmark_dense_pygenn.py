import numpy as np
import matplotlib.pyplot as plt
from time import time

from pygenn import (GeNNModel, VarLocation, init_var,
                    init_postsynaptic, init_weight_update,
                    init_sparse_connectivity)

def build_dense(num_pre: int, num_post: int, weight: float, prob):
    weights = np.zeros(num_pre * num_post, dtype=np.float32)
    
    weights[np.random.rand(num_pre * num_post) < prob] = weight
    #weights = np.reshape(weights, (num_pre, num_post))
    
    return weights
    
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

lif_params = {"C": 1.0, "TauM": 20.0, "Vrest": -49.0, "Vreset": RESET_VOLTAGE,
              "Vthresh": THRESHOHOLD_VOLTAGE, "Ioffset": 0.0, "TauRefrac": 5.0}

lif_init = {"V": init_var("Uniform", {"min": RESET_VOLTAGE, "max": THRESHOHOLD_VOLTAGE}),
            "RefracTime": 0.0}

excitatory_pop = model.add_neuron_population("E", NUM_EXCITATORY, "LIF", lif_params, lif_init)
inhibitory_pop = model.add_neuron_population("I", NUM_INHIBITORY, "LIF", lif_params, lif_init)

excitatory_pop.spike_recording_enabled = True

excitatory_postsynaptic_init = init_postsynaptic("ExpCurr", {"tau": 5.0})
inhibitory_postsynaptic_init = init_postsynaptic("ExpCurr", {"tau": 10.0})

model.add_synapse_population("EE", "DENSE",
    excitatory_pop, excitatory_pop,
    init_weight_update("StaticPulse", {}, {"g": build_dense(NUM_EXCITATORY,
                                                            NUM_EXCITATORY,
                                                            EXCITATORY_WEIGHT,
                                                            PROBABILITY_CONNECTION)}),
    excitatory_postsynaptic_init)

model.add_synapse_population("EI", "DENSE",
    excitatory_pop, inhibitory_pop,
    init_weight_update("StaticPulse", {}, {"g": build_dense(NUM_EXCITATORY,
                                                            NUM_INHIBITORY,
                                                            EXCITATORY_WEIGHT,
                                                            PROBABILITY_CONNECTION)}),
    excitatory_postsynaptic_init)

model.add_synapse_population("II", "DENSE",
    inhibitory_pop, inhibitory_pop,
    init_weight_update("StaticPulse", {}, {"g": build_dense(NUM_INHIBITORY,
                                                            NUM_INHIBITORY,
                                                            INHIBITORY_WEIGHT,
                                                            PROBABILITY_CONNECTION)}),
    inhibitory_postsynaptic_init)

model.add_synapse_population("IE", "DENSE",
    inhibitory_pop, excitatory_pop,
    init_weight_update("StaticPulse", {}, {"g": build_dense(NUM_INHIBITORY,
                                                            NUM_EXCITATORY,
                                                            INHIBITORY_WEIGHT,
                                                            PROBABILITY_CONNECTION)}),
    inhibitory_postsynaptic_init)

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
spike_times, spike_ids = excitatory_pop.spike_recording_data[0]

fig, axes = plt.subplots(2)

bin_size = 10.0
rate_bins = np.arange(0, 10000.0, bin_size)
rate = np.histogram(spike_times, bins=rate_bins)[0]
rate_bin_centres = rate_bins[:-1] + (bin_size / 2.0)

axes[0].scatter(spike_times, spike_ids, s=1)
axes[1].plot(rate_bin_centres, rate * (1000.0 / bin_size) * (1.0 / float(NUM_EXCITATORY)))

plt.show()
