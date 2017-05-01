import csv
import numpy as np
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import sys

original_resolution = 128
macro_pixel_resolution = 9
detector_resolution = 7

# How many 
timesteps_per_frame = 33

def read_spikes(filename, resolution, timesteps_per_frame):
    with open(filename, "rb") as spike_file:
        # Create CSV reader
        spike_csv_reader = csv.reader(spike_file, delimiter = ",")

        # Skip headers
        spike_csv_reader.next()

        # Read data and zip into columns
        spike_columns = zip(*spike_csv_reader)

        # Convert CSV columns to numpy
        if len(spike_columns) == 0:
            return np.zeros((resolution, resolution, 0))
        else:
            spike_times = np.asarray(spike_columns[0], dtype=float)
            spike_addresses = np.asarray(spike_columns[1], dtype=int)

            spike_x = np.remainder(spike_addresses, resolution)
            spike_y = np.floor_divide(spike_addresses, resolution)

            return np.histogramdd((spike_x, spike_y, spike_times),
                                  (range(resolution + 1), range(resolution + 1), range(0, int(np.ceil(np.amax(spike_times))) + 1, timesteps_per_frame)))[0]

# Build spike histograms
dvs_spikes = read_spikes("dvs_pixel_spikes.csv", 128, timesteps_per_frame)
macro_pixel_spikes = read_spikes("macro_pixel_spikes.csv", 9, timesteps_per_frame)
output_spikes = read_spikes("output_spikes.csv", 7 * 4, timesteps_per_frame)[:,:7,:]

# Extract left, right, up and down channels
direction_output_spikes = [output_spikes[i::4,:,:] for i in range(4)]

duration = max(dvs_spikes.shape[2], macro_pixel_spikes.shape[2], output_spikes.shape[2])

fig, axes = plt.subplots(1, 6)

dvs_image_data = dvs_spikes[:,:,0]
macro_pixel_image_data = macro_pixel_spikes[:,:,0]
direction_output_image_data = [d[:,:,0] for d in direction_output_spikes]

dvs_image = axes[0].imshow(dvs_image_data, interpolation="nearest", cmap="jet", vmin=0.0, vmax=float(timesteps_per_frame))
macro_pixel_image = axes[1].imshow(macro_pixel_image_data, interpolation="nearest", cmap="jet", vmin=0.0, vmax=float(timesteps_per_frame))
direction_output_image = [axes[2 + i].imshow(d, interpolation="nearest", cmap="jet", vmin=0.0, vmax=float(timesteps_per_frame))
                          for i, d in enumerate(direction_output_image_data)]

axes[0].set_title("DVS")
axes[1].set_title("Macro pixels")
axes[2].set_title("Left")
axes[3].set_title("Right")
axes[4].set_title("Up")
axes[5].set_title("Down")

# Add border showing region within which optical flow is calculated
border = axes[0].add_patch(
    patches.Rectangle(
        (41, 41),   # (x,y)
        45,          # width
        45,          # height
        fill=False,
        color="white",
        linewidth=2.0))

def updatefig(frame):
    global dvs_spikes, macro_pixel_spikes, direction_output_spikes
    global dvs_image_data, macro_pixel_image_data, direction_output_image_data
    global dvs_image, macro_pixel_image, direction_output_image
    global border
    
    # Decay image data
    dvs_image_data *= 0.9
    macro_pixel_image_data *= 0.9

    # Add spikes that occur during this frame
    if frame < dvs_spikes.shape[2]:
        dvs_image_data += dvs_spikes[:,:,frame]
    if frame < macro_pixel_spikes.shape[2]:
        macro_pixel_image_data += macro_pixel_spikes[:,:,frame]

    # Update images
    dvs_image.set_array(dvs_image_data)
    macro_pixel_image.set_array(macro_pixel_image_data)

    # Loop through direction outputs
    for s, d, i in zip(direction_output_spikes, direction_output_image_data, direction_output_image):
        d *= 0.9

        if frame < s.shape[2]:
            d += s[:, :, frame]

        i.set_array(d)

    # Return list of artists which we have updated
    # **YUCK** order of these dictates sort order
    # **YUCK** score_text must be returned whether it has
    # been updated or not to prevent overdraw
    return [dvs_image, macro_pixel_image] + direction_output_image + [border]

# Play animation
ani = animation.FuncAnimation(fig, updatefig, range(duration), interval=timesteps_per_frame, blit=True, repeat=True)
plt.show()