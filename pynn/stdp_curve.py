import logging, math, numpy, pylab, random, sys
from pyNN.parameters import Sequence

#-------------------------------------------------------------------
# This example uses the sPyNNaker implementation of pair-based STDP
# To reproduce the eponymous STDP curve first
# Plotted by Bi and Poo (1998)
#-------------------------------------------------------------------

#-------------------------------------------------------------------
# Common parameters
#-------------------------------------------------------------------
time_between_pairs = 1000
num_pairs = 60
start_w = 0.5

delta_t = [-100, -60, -40, -30, -20, -10, -1, 1, 10, 20, 30, 40, 60, 100]

start_time = 200
genn = True
spinnaker = False
nest = False

def simulate(sim, setup_kwargs):
    # Population parameters
    model = sim.IF_curr_exp
    cell_params = {'cm'        : 0.25, # nF
                'i_offset'  : 0.0,
                'tau_m'     : 10.0,
                'tau_refrac': 2.0,
                'tau_syn_E' : 2.5,
                'tau_syn_I' : 2.5,
                'v_reset'   : -70.0,
                'v_rest'    : -65.0,
                'v_thresh'  : -55.4}

    # SpiNNaker setup
    sim.setup(timestep=1.0, **setup_kwargs)
    sim_time = (num_pairs * time_between_pairs) + max(delta_t)

    # Neuron populations
    pre_pop = sim.Population(len(delta_t), model(**cell_params), label="pre")
    post_pop = sim.Population(len(delta_t), model(**cell_params), label="post")

    # Stimulating populations
    pre_stim = sim.Population(len(delta_t), sim.SpikeSourceArray(spike_times=[10]), label="pre_stim")
    post_stim = sim.Population(len(delta_t), sim.SpikeSourceArray(spike_times=[10]), label="post_stim")

    # Build stimulus spike times
    for i, t in enumerate(delta_t):
        # Pre after post
        if t > 0:
            post_phase = start_time
            pre_phase = start_time + t + 1
        # Post after pre
        else:
            post_phase = start_time - t
            pre_phase = start_time + 1

        pre_stim[i].spike_times = Sequence(range(pre_phase, sim_time, time_between_pairs))
        post_stim[i].spike_times = Sequence(range(post_phase, sim_time, time_between_pairs))

    # Connections between spike sources and neuron populations
    ee_connector = sim.OneToOneConnector()
    ee_synapse = sim.StaticSynapse(weight=2.0, delay=1.0)
    sim.Projection(pre_stim, pre_pop, ee_connector, ee_synapse)
    sim.Projection(post_stim, post_pop, ee_connector, ee_synapse)

    # Plastic Connection between pre_pop and post_pop
    stdp_model = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(tau_plus=16.7, tau_minus=33.7, A_plus=0.005, A_minus=0.005),
        weight_dependence=sim.AdditiveWeightDependence(w_min=0.0, w_max=start_w * 2.0),
        weight=start_w, delay=1.0, dendritic_delay_fraction=1.0,
    )

    plastic_proj = sim.Projection(pre_pop, post_pop, sim.OneToOneConnector(), stdp_model)

    print("Simulating for %us" % (sim_time / 1000))

    # Run simulation
    sim.run(sim_time)

    # Get weight from each projection
    end_w = plastic_proj.get("weight", format="list", with_address=False)

    # End simulation on SpiNNaker
    sim.end()

    return end_w

def simulate_spinnaker():
    import pynn_spinnaker as sim

    logger = logging.getLogger("pynn_spinnaker")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    setup_kwargs = {}

    return simulate(sim, setup_kwargs)

def simulate_nest():
    import pyNN.nest as sim
    return simulate(sim, {"spike_precision": "on_grid"})

def simulate_genn():
    import pynn_genn as sim
    return simulate(sim, {})

#-------------------------------------------------------------------
# Plot curve
#-------------------------------------------------------------------
# Plot STDP curve
figure, axis = pylab.subplots()
axis.set_xlabel(r"$(t_{j} - t_{i}/ms)$")
axis.set_ylabel(r"$(\frac{\Delta w_{ij}}{w_{ij}})$", rotation = "horizontal", size = "xx-large")

if spinnaker:
    end_w = simulate_spinnaker()
    delta_w = [(w - start_w) / start_w for w in end_w]
    axis.plot(delta_t, delta_w, color="red", label="SpiNNaker")

if genn:
    end_w = simulate_genn()
    delta_w = [(w - start_w) / start_w for w in end_w]
    axis.plot(delta_t, delta_w, color="green", label="GeNN")

if nest:
    end_w = simulate_nest()
    delta_w = [(w - start_w) / start_w for w in end_w]
    axis.plot(delta_t, delta_w, color="blue", label="NEST")

axis.axhline(color = "grey", linestyle = "--")
axis.axvline(color = "grey", linestyle = "--")
axis.legend()
pylab.show()
