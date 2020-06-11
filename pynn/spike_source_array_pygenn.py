import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt

from pygenn import genn_wrapper
from pygenn import genn_model

# Generate poisson spike trains
poisson_spikes = []
dt = 1.0
rate = 10.0
isi = 1000.0 / (rate * dt)
for p in range(100):
    time = 0.0
    neuron_spikes = []
    while True:
        time += expon.rvs(1) * isi
        if time >= 500.0:
            break
        else:
            neuron_spikes.append(time)
    poisson_spikes.append(neuron_spikes)

model = genn_model.GeNNModel("float", "ssa")
model.dT = dt

# Count spikes each neuron should emit
spike_counts = [len(n) for n in poisson_spikes]

# Get start and end indices of each spike sources section
end_spike = np.cumsum(spike_counts)
start_spike = np.empty_like(end_spike)
start_spike[0] = 0
start_spike[1:] = end_spike[0:-1]

# Build model
model = genn_model.GeNNModel("float", "spike_source_array")
model.dT = dt

ssa = model.add_neuron_population("SSA", 100, "SpikeSourceArray", {}, 
                                  {"startSpike": start_spike, "endSpike": end_spike})
ssa.set_extra_global_param("spikeTimes", np.hstack(poisson_spikes))

model.build()
model.load()

# Simulate
spike_ids = np.empty(0)
spike_times = np.empty(0)
while model.t < 500.0:
    model.step_time()
    ssa.pull_current_spikes_from_device()

    times = np.ones_like(ssa.current_spikes) * model.t
    spike_ids = np.hstack((spike_ids, ssa.current_spikes))
    spike_times = np.hstack((spike_times, times))

# Plot for verification
fig,axis = plt.subplots()
for i, n in enumerate(poisson_spikes):
    axis.scatter(n, [i] * len(n), color="blue", label="Offline")
axis.scatter(spike_times, spike_ids, color="red", label="GeNN")
axis.legend()
plt.show()