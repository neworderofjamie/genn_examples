import numpy as np
import matplotlib.pyplot as plt
from time import time

from pygenn import genn_wrapper
from pygenn import genn_model

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

model = genn_model.GeNNModel("float", "va_benchmark")
model.dT = TIMESTEP

fixed_prob = {"prob": PROBABILITY_CONNECTION}

lif_params = {"C": 1.0, "TauM": 20.0, "Vrest": -49.0, "Vreset": RESET_VOLTAGE,
              "Vthresh": THRESHOHOLD_VOLTAGE, "Ioffset": 0.0, "TauRefrac": 5.0}

lif_init = {"V": genn_model.init_var("Uniform", {"min": RESET_VOLTAGE, "max": THRESHOHOLD_VOLTAGE}),
            "RefracTime": 0.0}

excitatory_synapse_init = {"g": EXCITATORY_WEIGHT}
inhibitory_synapse_init = {"g": INHIBITORY_WEIGHT}

excitatory_post_syn_params = {"tau": 5.0}
inhibitory_post_syn_params = {"tau": 10.0}

excitatory_pop = model.add_neuron_population("E", NUM_EXCITATORY, "LIF", lif_params, lif_init)
inhibitory_pop = model.add_neuron_population("I", NUM_INHIBITORY, "LIF", lif_params, lif_init)

model.add_synapse_population("EE", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
    excitatory_pop, inhibitory_pop,
    "StaticPulse", {}, excitatory_synapse_init, {}, {},
    "ExpCurr", excitatory_post_syn_params, {},
    genn_model.init_connectivity("FixedProbabilityNoAutapse", fixed_prob))

model.add_synapse_population("EI", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
    excitatory_pop, inhibitory_pop,
    "StaticPulse", {}, excitatory_synapse_init, {}, {},
    "ExpCurr", excitatory_post_syn_params, {},
    genn_model.init_connectivity("FixedProbability", fixed_prob))

model.add_synapse_population("II", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
    inhibitory_pop, inhibitory_pop,
    "StaticPulse", {}, inhibitory_synapse_init, {}, {},
    "ExpCurr", inhibitory_post_syn_params, {},
    genn_model.init_connectivity("FixedProbabilityNoAutapse", fixed_prob))

model.add_synapse_population("IE", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
    inhibitory_pop, excitatory_pop,
    "StaticPulse", {}, inhibitory_synapse_init, {}, {},
    "ExpCurr", inhibitory_post_syn_params, {},
    genn_model.init_connectivity("FixedProbability", fixed_prob))

print("Building Model")
model.build()
print("Loading Model")
model.load()

spike_ids = None
spike_times = None

sim_start_time = time()
while model.t < 10000.0:
    model.step_time()

    model.pull_current_spikes_from_device("E")

    i = excitatory_pop.current_spikes
    t = np.ones(i.shape) * model.t

    if spike_ids is None:
        spike_ids = np.copy(i)
        spike_times = t
    else:
        spike_ids = np.hstack((spike_ids, i))
        spike_times = np.hstack((spike_times, t))

sim_end_time = time()
print("Simulation time:%fs" % (sim_end_time - sim_start_time))

fig, axes = plt.subplots(2)

bin_size = 10.0
rate_bins = np.arange(0, 10000.0, bin_size)
rate = np.histogram(spike_times, bins=rate_bins)[0]
rate_bin_centres = rate_bins[:-1] + (bin_size / 2.0)

axes[0].scatter(spike_times, spike_ids, s=1)
axes[1].plot(rate_bin_centres, rate * (1000.0 / bin_size) * (1.0 / float(NUM_EXCITATORY)))

plt.show()
