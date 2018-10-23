# --------------------------------------------------------------------------
# Song, S., Miller, K. D., & Abbott, L. F. (2000).
# Competitive Hebbian learning through spike-timing-dependent synaptic plasticity.
# Nature Neuroscience, 3(9), 919-26. http://doi.org/10.1038/78829
# --------------------------------------------------------------------------

import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
from pyNN.random import NumpyRNG, RandomDistribution
from six import iteritems, iterkeys, itervalues

dt = 1.0
num_ex_synapses = 1000
num_neurons = 1
g_max = 0.01
duration = 300000

def simulate(sim, rng, setup_kwargs):
    sim.setup(timestep=dt, **setup_kwargs)

    # Brian was performing synaptic input with ge*(Ee-vr)
    # Ee = 0 so infact Vr is being treated as an input resistance and therefore C = tau_m / v_rest = 10*10^-3 / 60*10^6 = 0.17*10^-9

    # Weight dependences to test
    weight_dependences = [
        sim.AdditiveWeightDependence(w_min=0.0, w_max=g_max),
        sim.MultiplicativeWeightDependence(w_min=0.0, w_max=g_max)]

    # Create a neural population to stimulate using each weight dependence
    neural_pops = [sim.Population(num_neurons,
                                  sim.IF_curr_exp(v_rest=-74.0, v_reset=-60.0, v_thresh=-54.0,
                                                  tau_syn_E=5.0, tau_syn_I=5.0, tau_m=10.0, cm=0.17))
                                  for _ in weight_dependences]

    # Create poisson source to stimulate both populations
    ex_poisson = sim.Population(num_ex_synapses, sim.SpikeSourcePoisson(rate=15.0))

    # Record spikes from each population
    for n in neural_pops:
        n.record("spikes")

    a_plus = 0.01
    a_minus = 1.05 * a_plus

    # Create weight distribution
    weight_dist = RandomDistribution("uniform", low=0, high=g_max, rng=rng)

    # Create STDP projections with correct weight dependence between poisson source and each neural population
    projections = [sim.Projection(ex_poisson, n,
                                  sim.AllToAllConnector(),
                                  sim.STDPMechanism(
                                      timing_dependence=sim.SpikePairRule(tau_plus=20.0, tau_minus=20.00,
                                                                          A_plus=a_plus, A_minus=a_minus),
                                      weight_dependence=w,
                                      weight=weight_dist, delay=dt, dendritic_delay_fraction=1.0),
                                  receptor_type="excitatory")
                                  for n, w in zip(neural_pops, weight_dependences)]
    # Simulate
    sim.run(duration)

    # Download weights
    weights = [np.asarray(p.get("weight", format="list", with_address=False))
               for p in projections]

    # Download spikes
    data = [n.get_data() for n in neural_pops]

    # End simulation
    sim.end()

    # Return learned weights and data
    return weights, data

def simulate_genn():
    import pynn_genn as sim

    rng = NumpyRNG()

    return simulate(sim, rng, {})

def simulate_nest():
    import pyNN.nest as sim
    rng = NumpyRNG()
    return simulate(sim, rng, {"spike_precision": "on_grid"})

# Simulate network
weights, data = simulate_genn()


figure, axes = plt.subplots(1, len(weights), sharey=True)

axes[0].set_xlabel("Normalised weight")

for i, (w, d, l) in enumerate(zip(weights, data,
                                  ["Additive", "Multiplicative"])):
    print("%s mean firing rate %fHz" %
          (l, float(len(d.segments[0].spiketrains[0])) / float(duration / 1000)))

    axes[i].hist(w / g_max, np.linspace(0.0, 1.0, 20))
    axes[i].set_title("%s weight dependence" % l)
    axes[i].set_ylabel("Number of synapses")
    axes[i].set_xlim((0.0, 1.0))

plt.show()