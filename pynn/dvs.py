from pygenn.genn_model import GeNNModel
from pygenn.dvs import DVS, Polarity
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

PLOT_INTERVAL = 1000 // 60

dvs_device = DVS.create_davis()
num_dvs_pixels = dvs_device.width * dvs_device.height
model = GeNNModel("float", "dvs")
model.dt = 1.0

dvs = model.add_neuron_population("DVS", num_dvs_pixels, "EventCamera")
dvs.spike_recording_enabled = True
dvs.extra_global_params["spikeVector"].set_init_values(np.empty((num_dvs_pixels + 31) // 32, dtype=np.uint32))

model.build()
model.load(num_recording_timesteps=PLOT_INTERVAL)

# Start event streaming
dvs_device.start()

# Create axis
fig, axis = plt.subplots()

# Create empty scatter blog
scatter = axis.scatter([], [], s=1)
axis.set_xlim((0, dvs_device.width))
axis.set_ylim((0, dvs_device.height))

# Cache background
fig.canvas.draw()
ax_background = fig.canvas.copy_from_bbox(axis.bbox)

# Show figure
plt.show(block=False)

spike_vector = dvs.extra_global_params["spikeVector"]
while True:
    # Loop through interval between frames
    for i in range(PLOT_INTERVAL):
        # Zero spike vector, read events into it and push to GPU
        spike_vector.view[:] = 0
        dvs_device.read_events(spike_vector._array, Polarity.ON_ONLY)
        spike_vector.push_to_device()

        # Step time
        model.step_time()
    
    # Download all spikes which have happened in intervals
    model.pull_recording_buffers_from_device()

    # Unravel neuron IDs into x, y coordinates
    spike_times, spike_ids = dvs.spike_recording_data[0]
    spike_coord = np.unravel_index(spike_ids, (dvs_device.height, dvs_device.width))
    
    # Update scatter plot
    scatter.set_offsets(np.c_[spike_coord[1], spike_coord[0]])
    
    # Restore background
    fig.canvas.restore_region(ax_background)
    
    # Redraw just scatter
    axis.draw_artist(scatter)

    fig.canvas.blit(axis.bbox)

    # fill in the axes rectangle
    fig.canvas.flush_events()
