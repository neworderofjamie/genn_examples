
import logging
import numpy
import pylab
import pynn_genn as sim
from pyNN.random import NumpyRNG, RandomDistribution


#-------------------------------------------------------------------
# This example uses the sPyNNaker implementation of the inhibitory
# Plasticity rule developed by Vogels, Sprekeler, Zenke et al (2011)
# To reproduce the experiment from their paper
#-------------------------------------------------------------------
# Population parameters
model = sim.IF_curr_exp
cell_params = {
    'cm'        : 0.2, # nF
    'i_offset'  : 0.2,
    'tau_m'     : 20.0,
    'tau_refrac': 5.0,
    'tau_syn_E' : 5.0,
    'tau_syn_I' : 10.0,
    'v_reset'   : -60.0,
    'v_rest'    : -60.0,
    'v_thresh'  : -50.0
    }


# How large should the population of excitatory neurons be?
# (Number of inhibitory neurons is proportional to this)
NUM_EXCITATORY = 2000

# Function to build the basic network - dynamics should be a PyNN synapse dynamics object
def build_network(ie_synapse, e_mean_firing_rate):
    # Create excitatory and inhibitory populations of neurons
    ex_pop = sim.Population(NUM_EXCITATORY, model(**cell_params), label="E")
    in_pop = sim.Population(NUM_EXCITATORY / 4, model(**cell_params), label="I")
    
    # Randomize initial membrane voltages
    #rng = sim.NativeRNG(host_rng=NumpyRNG())
    rng = sim.NumpyRNG()
    uniformDistr = RandomDistribution('uniform', low=-60.0,
                                      high=-50.0, rng=rng)
    ex_pop.initialize(v=uniformDistr)
    in_pop.initialize(v=uniformDistr)


    # Record excitatory spikes
    ex_pop.record("spikes")
    
    # Make excitatory->inhibitory projections
    static_synapse = sim.StaticSynapse(weight=0.03)
    connector = sim.FixedProbabilityConnector(0.02, rng=rng)
    sim.Projection(ex_pop, in_pop, connector, static_synapse,
                   receptor_type='excitatory')
    sim.Projection(ex_pop, ex_pop, connector, static_synapse,
                   receptor_type='excitatory')

    # Make inhibitory->inhibitory projections
    sim.Projection(in_pop, in_pop, connector, static_synapse,
                   receptor_type='inhibitory')
    
    # Make inhibitory->excitatory projections
    ie_projection = sim.Projection(in_pop, ex_pop, connector, ie_synapse,
                                   receptor_type='inhibitory')

    return ex_pop, ie_projection

'''
# Build static network
sim.setup(timestep=1.0)
static_ex_pop,_ = build_network(sim.StaticSynapse(weight=0.0, delay=1.0), 300.0)

# Run for 1s
sim.run(1000)

# Get static spikes and save to disk
static_data = static_ex_pop.get_data()

# Clear simulator state
sim.end()
'''
# Clear simulation state
sim.setup(min_delay=1.0, max_delay=7.0, timestep=1.0)

# Build inhibitory plasticity  model
stdp_model = sim.STDPMechanism(
    timing_dependence = sim.Vogels2011Rule(rho=0.12, tau=20.0, eta=0.005),
    weight_dependence = sim.AdditiveWeightDependence(w_min=-1.0, w_max=0.0),
    weight=0.0, delay=1.0,
)

# Build plastic network
plastic_ex_pop, plastic_ie_projection = build_network(stdp_model, 10.0)

# Run simulation
sim.run(10000)

# Get plastic spikes and save to disk
plastic_data = plastic_ex_pop.get_data()

plastic_weights = plastic_ie_projection.get("weight", format="list", with_address=False)
mean_weight = numpy.average(plastic_weights)
print "Mean learnt ie weight:%f" % mean_weight

def plot_spiketrains(axis, segment, offset, **kwargs):
    for spiketrain in segment.spiketrains:
        y = numpy.ones_like(spiketrain) * (offset + spiketrain.annotations["source_index"])
        axis.scatter(spiketrain, y, **kwargs)

def calculate_rate(segment, rate_bins, population_size):
    population_histogram = numpy.zeros(len(rate_bins) - 1)
    for spiketrain in segment.spiketrains:
        population_histogram += numpy.histogram(spiketrain, bins=rate_bins)[0]

    return population_histogram * (1000.0 / 10.0) * (1.0 / float(population_size))

# Create plot
fig, axes = pylab.subplots(4)
binsize = 10

# Plot last 200ms of static spikes (to match Brian script)
'''
axes[0].set_title("Excitatory raster without inhibitory plasticity")
plot_spiketrains(axes[0], static_data.segments[0], 0, color="blue", s=2)
axes[0].set_xlim(800, 1000)
axes[0].set_ylim(0, NUM_EXCITATORY)

# Plot last 200ms of static spike rates
static_bins = numpy.arange(0, 1000 + 1, binsize)
static_rate = calculate_rate(static_data.segments[0], static_bins, NUM_EXCITATORY)
axes[1].set_title("Excitatory rates without inhibitory plasticity")
axes[1].plot(static_bins[0:-1], static_rate, color="red")
axes[1].set_xlim(800, 1000)
'''
# Plot last 200ms of plastic spikes (to match Brian script)
axes[2].set_title("Excitatory raster with inhibitory plasticity")
plot_spiketrains(axes[2], plastic_data.segments[0], 0, color="blue", s=2)
axes[2].set_xlim(9800, 10000)
axes[2].set_ylim(0, NUM_EXCITATORY)

# Plot last 200ms of plastic spike rates
plastic_bins = numpy.arange(0, 10000 + 1, binsize)
plastic_rate = calculate_rate(plastic_data.segments[0], plastic_bins, NUM_EXCITATORY)
axes[3].set_title("Excitatory rates with inhibitory plasticity")
axes[3].plot(plastic_bins[0:-1], plastic_rate, color="red")
#axes[3].set_xlim(9800, 10000)
#axes[3].set_ylim(0, 20)

# Show figures
pylab.show()

# End simulation on SpiNNaker
sim.end()
