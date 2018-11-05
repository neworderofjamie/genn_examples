# coding: utf-8
"""
Balanced network of excitatory and inhibitory neurons.

An implementation of benchmarks 1 and 2 from

    Brette et al. (2007) Journal of Computational Neuroscience 23: 349-398

The network is based on the CUBA and COBA models of Vogels & Abbott
(J. Neurosci, 2005).  The model consists of a network of excitatory and
inhibitory neurons, connected via current-based "exponential"
synapses (instantaneous rise, exponential decay).


Usage: python VAbenchmarks.py [-h] [--plot-figure] [--use-views] [--use-assembly]
                              [--use-csa] [--debug DEBUG]
                              simulator benchmark

positional arguments:
  simulator       neuron, nest, brian or another backend simulator
  benchmark       either CUBA or COBA

optional arguments:
  -h, --help      show this help message and exit
  --plot-figure   plot the simulation results to a file
  --use-views     use population views in creating the network
  --use-assembly  use assemblies in creating the network
  --use-csa       use the Connection Set Algebra to define the connectivity
  --debug DEBUG   print debugging information


Andrew Davison, UNIC, CNRS
August 2006

"""
import logging
import numpy
import socket
import pylab
from math import *
from six import iteritems
from pyNN.utility import get_simulator, Timer, ProgressBar, init_logging, normalized_filename
from pyNN.random import NumpyRNG, RandomDistribution
from collections import defaultdict

# === Configure the simulator ================================================

genn = True
use_views = True
use_assembly = False
benchmark = "CUBA"
use_csa = False

rate_estimate = 5.0

rngseed  = 98765
rng = NumpyRNG(seed=rngseed)

if genn:
    import pynn_genn as sim

    setup_kwargs = {}

    #rng = sim.NativeRNG(host_rng=rng)
else:
    import pyNN.nest as sim

    setup_kwargs = {"spike_precision": "on_grid"}

timer = Timer()

# === Define parameters ========================================================


n        = 4000  # number of cells
r_ei     = 4.0   # number of excitatory cells:number of inhibitory cells
pconn    = 0.1  # connection probability
stim_dur = 50.   # (ms) duration of random stimulation
rate     = 100.  # (Hz) frequency of the random stimulation

dt       = 1.0   # (ms) simulation timestep
tstop    = 500  # (ms) simulaton duration
delay    = 2.0 * dt

# Cell parameters
area     = 20000. # (µm²)
tau_m    = 20.    # (ms)
cm       = 1.     # (µF/cm²)plot_spiketrains
g_leak   = 5e-5   # (S/cm²)
if benchmark == "COBA":
    E_leak   = -60.  # (mV)
elif benchmark == "CUBA":
    E_leak   = -49.  # (mV)
v_thresh = -50.   # (mV)
v_reset  = -60.   # (mV)
t_refrac = 5.     # (ms) (clamped at v_reset)
v_mean   = -60.   # (mV) 'mean' membrane potential, for calculating CUBA weights
tau_exc  = 5.     # (ms)
tau_inh  = 10.    # (ms)

# Synapse parameters
scale = (4000. / float(n)) * (0.02 / pconn)
if benchmark == "COBA":
    Gexc = 4. * scale   # (nS)
    Ginh = 51. * scale    # (nS)
elif benchmark == "CUBA":
    Gexc = 0.27 * scale   # (nS) #Those weights should be similar to the COBA weights
    Ginh = 4.5 * scale    # (nS) # but the delpolarising drift should be taken into account
Erev_exc = 0.     # (mV)
Erev_inh = -80.   # (mV)

### what is the synaptic delay???

# === Calculate derived parameters =============================================

area  = area*1e-8                     # convert to cm²
cm    = cm*area*1000                  # convert to nF
Rm    = 1e-6/(g_leak*area)            # membrane resistance in MΩ
assert tau_m == cm*Rm                 # just to check
n_exc = int(round((n*r_ei/(1+r_ei)))) # number of excitatory cells
n_inh = n - n_exc                     # number of inhibitory cells
if benchmark == "COBA":
    celltype = sim.IF_cond_exp
    w_exc    = Gexc*1e-3              # We convert conductances to uS
    w_inh    = Ginh*1e-3
elif benchmark == "CUBA":
    celltype = sim.IF_curr_exp
    w_exc = 1e-3*Gexc*(Erev_exc - v_mean) # (nA) weight of excitatory synapses
    w_inh = 1e-3*Ginh*(Erev_inh - v_mean) # (nA)
    assert w_exc > 0; assert w_inh < 0

# === Build the network ========================================================

node_id = sim.setup(timestep=dt, min_delay=dt, max_delay=2.0, **setup_kwargs)

cell_params = {
    'tau_m'      : tau_m,    'tau_syn_E'  : tau_exc,  'tau_syn_I'  : tau_inh,
    'v_rest'     : E_leak,   'v_reset'    : v_reset,  'v_thresh'   : v_thresh,
    'cm'         : cm,       'tau_refrac' : t_refrac}

if (benchmark == "COBA"):
    cell_params['e_rev_E'] = Erev_exc
    cell_params['e_rev_I'] = Erev_inh

timer.start()

print("%s Creating cell populations..." % node_id)
if use_views:
    # create a single population of neurons, and then use population views to define
    # excitatory and inhibitory sub-populations
    all_cells = sim.Population(n_exc + n_inh, celltype(**cell_params), label="All Cells")

    exc_cells = all_cells[:n_exc]
    exc_cells.label = "Excitatory cells"
    inh_cells = all_cells[n_exc:]
    inh_cells.label = "Inhibitory cells"
else:
    # create separate populations for excitatory and inhibitory neurons
    exc_cells = sim.Population(n_exc, celltype(**cell_params), label="Excitatory_Cells")
    inh_cells = sim.Population(n_inh, celltype(**cell_params), label="Inhibitory_Cells")


    if use_assembly:
        # group the populations into an assembly
        all_cells = exc_cells + inh_cells

if benchmark == "COBA":
    ext_stim = sim.Population(20, sim.SpikeSourcePoisson(rate=rate, duration=stim_dur), label="expoisson")
    rconn = 0.01
    ext_conn = sim.FixedProbabilityConnector(rconn)
    ext_syn = sim.StaticSynapse(weight=0.1)

print("%s Initialising membrane potential to random values..." % node_id)

uniformDistr = RandomDistribution('uniform', low=v_reset, high=v_thresh, rng=rng)
if use_views:
    all_cells.initialize(v=uniformDistr)
else:
    exc_cells.initialize(v=uniformDistr)
    inh_cells.initialize(v=uniformDistr)

print("%s Connecting populations..." % node_id)
progress_bar = ProgressBar(width=20)
if use_csa:
    connector = sim.CSAConnector(csa.cset(csa.random(pconn)))
else:
    connector = sim.FixedProbabilityConnector(pconn, rng=rng, callback=progress_bar)
exc_syn = sim.StaticSynapse(weight=w_exc, delay=delay)
inh_syn = sim.StaticSynapse(weight=w_inh, delay=delay)


connections = {}
if use_views or use_assembly:
    connections['exc'] = sim.Projection(exc_cells, all_cells, connector, exc_syn, receptor_type='excitatory')
    connections['inh'] = sim.Projection(inh_cells, all_cells, connector, inh_syn, receptor_type='inhibitory')
    if (benchmark == "COBA"):
        connections['ext'] = sim.Projection(ext_stim, all_cells, ext_conn, ext_syn, receptor_type='excitatory')
else:
    connections['e2e'] = sim.Projection(exc_cells, exc_cells, connector, exc_syn, receptor_type='excitatory')
    connections['e2i'] = sim.Projection(exc_cells, inh_cells, connector, exc_syn, receptor_type='excitatory')
    connections['i2e'] = sim.Projection(inh_cells, exc_cells, connector, inh_syn, receptor_type='inhibitory')
    connections['i2i'] = sim.Projection(inh_cells, inh_cells, connector, inh_syn, receptor_type='inhibitory')
    if (benchmark == "COBA"):
        connections['ext2e'] = sim.Projection(ext_stim, exc_cells, ext_conn, ext_syn, receptor_type='excitatory')
        connections['ext2i'] = sim.Projection(ext_stim, inh_cells, ext_conn, ext_syn, receptor_type='excitatory')

# === Setup recording ==========================================================
print("%s Setting up recording..." % node_id)
if use_views or use_assembly:
    all_cells.record('spikes')
    #exc_cells[[0, 1]].record('v')
else:
    exc_cells.record('spikes')
    inh_cells.record('spikes')
    #exc_cells[0, 1].record('v')

buildCPUTime = timer.diff()

# === Save connections to file =================================================

#for prj in connections.keys():
    #connections[prj].saveConnections('Results/VAbenchmark_%s_%s_%s_np%d.conn' % (benchmark, prj, options.simulator, np))
saveCPUTime = timer.diff()

# === Run simulation ===========================================================

print("%d Running simulation..." % node_id)

sim.run(tstop)

simCPUTime = timer.diff()

#------------------------------------------------------------------------------
# Plot output
#------------------------------------------------------------------------------
def plot_spiketrains(axis, segment, offset, **kwargs):
    for spiketrain in segment.spiketrains:
        y = numpy.ones_like(spiketrain) * (offset + spiketrain.annotations["source_index"])
        axis.scatter(spiketrain, y, **kwargs)

def calculate_rate(segment, rate_bins, population_size):
    population_histogram = numpy.zeros(len(rate_bins) - 1)
    for spiketrain in segment.spiketrains:
        population_histogram += numpy.histogram(spiketrain, bins=rate_bins)[0]

    return population_histogram * (1000.0 / 10.0) * (1.0 / float(population_size))

e_data = exc_cells.get_data()
i_data = inh_cells.get_data()

writeCPUTime = timer.diff()

print("Build time             : %g s" % buildCPUTime)
print("Simulation time        : %g s" % simCPUTime)
print("Writing time           : %g s" % writeCPUTime)

# Create plot
figure, axes = pylab.subplots(2, sharex=True)
# Plot spike raster
raster_axis = axes[0]
plot_spiketrains(raster_axis, e_data.segments[0], 0, color="red", s=0.5)
plot_spiketrains(raster_axis, i_data.segments[0], 0 if use_views else n_exc, color="grey", s=0.5)
raster_axis.set_ylabel("Neuron ID")
raster_axis.set_xlim((0, tstop))
raster_axis.set_ylim((0, n))

# Plot excitatory and inhibitory rate
rate_bins = numpy.arange(0, tstop + 1, 10)

# Calculate rates
e_rates = calculate_rate(e_data.segments[0], rate_bins, n_exc)
i_rates = calculate_rate(i_data.segments[0], rate_bins, n_inh)

axes[1].plot(rate_bins[:-1], e_rates, color="red")
axes[1].plot(rate_bins[:-1], i_rates, color="grey")
axes[1].set_ylabel("Rate [Hz]")
axes[1].set_xlabel("Time [ms]")
axes[1].set_ylim((0, 10.0))

pylab.show()

# === Finished with simulator ==================================================

sim.end()
