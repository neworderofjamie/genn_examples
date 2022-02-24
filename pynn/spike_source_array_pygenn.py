import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt

from pygenn import GeNNModel

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

# Count spikes each neuron should emit
spike_counts = [len(n) for n in poisson_spikes]

# Get start and end indices of each spike sources section
end_spike = np.cumsum(spike_counts)
start_spike = np.empty_like(end_spike)
start_spike[0] = 0
start_spike[1:] = end_spike[0:-1]

# Build model
model = GeNNModel("float", "spike_source_array")
model.dt = dt

ssa = model.add_neuron_population("SSA", 100, "SpikeSourceArray", {}, 
                                  {"startSpike": start_spike, "endSpike": end_spike})
ssa.extra_global_params["spikeTimes"].set_values(np.hstack(poisson_spikes))
ssa.spike_recording_enabled = True

model.build()
model.load(num_recording_timesteps=int(np.ceil(500.0 / dt)))

# Simulate
while model.t < 500.0:
    model.step_time()

# Download recording data
model.pull_recording_buffers_from_device()

# Get recording data
spike_times, spike_ids = ssa.spike_recording_data

# Plot for verification
fig,axis = plt.subplots()
for i, n in enumerate(poisson_spikes):
    axis.scatter(n, [i] * len(n), color="blue", label=("Offline" if i == 0 else None))
axis.scatter(spike_times, spike_ids, color="red", label="GeNN")
axis.legend()
plt.show()