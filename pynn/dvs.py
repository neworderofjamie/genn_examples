from pygenn.genn_model import GeNNModel
from pygenn.dvs import DVS, Polarity
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

PLOT_INTERVAL = 1000 // 60

dvs = DVS.create_davis()
model = GeNNModel("float", "dvs")
model.dt = 1.0

dvs.add_to_model(model)
dvs.pop.spike_recording_enabled = True

model.build()
model.load(num_recording_timesteps=PLOT_INTERVAL)

# Start event streaming
dvs.start()

# Create axis
fig, axis = plt.subplots()

# Create empty scatter blog
on_scatter = axis.scatter([], [], s=1, c="red")
off_scatter = axis.scatter([], [], s=1, c="green")
axis.set_xlim((0, dvs.output_width))
axis.set_ylim((0, dvs.output_height))

# Cache background
fig.canvas.draw()
ax_background = fig.canvas.copy_from_bbox(axis.bbox)

# Show figure
plt.show(block=False)

while True:
    # Loop through interval between frames
    for i in range(PLOT_INTERVAL):
        dvs.copy_spikes()

        # Step time
        model.step_time()
    
    # Download all spikes which have happened in intervals
    model.pull_recording_buffers_from_device()

    # Unravel neuron IDs into x, y coordinates
    spike_times, spike_ids = dvs.pop.spike_recording_data[0]
    spike_coord = np.unravel_index(spike_ids, (dvs.output_height, dvs.output_width, dvs.output_channels))
    
    # Update scatter plot
    on_mask = (spike_coord[2] == 1)
    on_scatter.set_offsets(np.c_[spike_coord[1][on_mask], spike_coord[0][on_mask]])    
    off_mask = (spike_coord[2] == 0)
    off_scatter.set_offsets(np.c_[spike_coord[1][off_mask], spike_coord[0][off_mask]])

    
    # Restore background
    fig.canvas.restore_region(ax_background)
    
    # Redraw just scatter
    axis.draw_artist(on_scatter)
    axis.draw_artist(off_scatter)

    fig.canvas.blit(axis.bbox)

    # fill in the axes rectangle
    fig.canvas.flush_events()
