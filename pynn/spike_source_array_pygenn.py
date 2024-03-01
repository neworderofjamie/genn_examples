import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt

from pygenn import GeNNModel

# Generate poisson spike trains
poisson_ids = []
poisson_times = []
dt = 1.0
rate = 10.0
isi = 1000.0 / (rate * dt)
for p in range(100):
    time = 0.0
    while True:
        time += expon.rvs(1) * isi
        if time >= 500.0:
            break
        else:
            poisson_ids.append(p)
            poisson_times.append(time)

poisson_ids = np.asarray(poisson_ids, dtype=int)
poisson_times = np.asarray(poisson_times)

# Calculate end spikes
end_spike = np.cumsum(np.bincount(poisson_ids, minlength=100))

# Sort events first by neuron id and then 
# by time and use to order spike times
poisson_times = poisson_times[np.lexsort((poisson_times, poisson_ids))]

start_spike = np.concatenate(([0], end_spike[0:-1]))

# Build model
model = GeNNModel("float", "spike_source_array")
model.dt = dt

ssa = model.add_neuron_population("SSA", 100, "SpikeSourceArray", {}, 
                                  {"startSpike": start_spike, "endSpike": end_spike})
ssa.extra_global_params["spikeTimes"].set_init_values(poisson_times)
ssa.spike_recording_enabled = True

model.build()
model.load(num_recording_timesteps=int(np.ceil(500.0 / dt)))

# Simulate
while model.t < 500.0:
    model.step_time()

# Download recording data
model.pull_recording_buffers_from_device()

# Get recording data
spike_times, spike_ids = ssa.spike_recording_data[0]

# Plot for verification
fig,axis = plt.subplots()
axis.scatter(poisson_times, poisson_ids, color="blue", label="Offline")
axis.scatter(spike_times, spike_ids, color="red", label="GeNN")
axis.legend()
plt.show()