import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

original_resolution = 128
output_resolution = 32

# How many 
timesteps_per_frame = 33
duration = 6000
num_frames = duration // timesteps_per_frame


timestep_inputs = np.zeros((output_resolution, output_resolution, num_frames), dtype=float)
s_output = None

with open(sys.argv[1], "r") as input_spike_file, open("s_spikes.csv", "rb") as s_spike_file:
    # Read input spike file
    scale = original_resolution / output_resolution
    for line in input_spike_file:
        # Split lines into time and keys
        time_string, keys_string = line.split(";")
        time = int(time_string)
        frame = time // timesteps_per_frame
        
        # Load spikes into numpy array 
        frame_input_spikes = np.asarray(keys_string.split(","), dtype=int)
        
        # Split into X and Y
        frame_input_x = np.floor_divide(frame_input_spikes, original_resolution)
        frame_input_y = np.remainder(frame_input_spikes, original_resolution)
        
        # Scale to output resolution
        frame_input_x /= scale
        frame_input_y /= scale
        
        # Take histogram so as to assemble frame
        frame_input_image = np.histogram2d(frame_input_x, frame_input_y, 
                                           bins=np.arange(output_resolution + 1))[0]
        
        # Add this to correct frame of image
        timestep_inputs[:,:,frame] += frame_input_image
        
    # Create CSV reader
    s_spike_csv_reader = csv.reader(s_spike_file, delimiter = ",")
    
    # Skip headers
    s_spike_csv_reader.next()

    # Read data and zip into columns
    s_spike_columns = zip(*s_spike_csv_reader)
    
    # Convert CSV columns to numpy
    s_spike_times = np.asarray(s_spike_columns[0], dtype=float)
    s_spike_neuron_id = np.asarray(s_spike_columns[1], dtype=int)
    
    # Scale time inot frames and x and y into output resolution
    s_spike_frames = np.floor_divide(s_spike_times, timesteps_per_frame)
    s_spike_x = np.floor_divide(s_spike_neuron_id, output_resolution)
    s_spike_y = np.remainder(s_spike_neuron_id, output_resolution)
    
    # Build 3D histogram i.e. video frames from this data
    s_output = np.histogramdd((s_spike_x, s_spike_y, s_spike_frames), 
                              bins=(np.arange(output_resolution + 1), 
                                    np.arange(output_resolution + 1),
                                    np.arange(num_frames + 1)))[0]

fig, axes = plt.subplots(1, 2)
input_image_data = timestep_inputs[:,:,0]
s_image_data = s_output[:,:,0]
input_image = axes[0].imshow(input_image_data, interpolation="nearest", vmin=0.0, vmax=float(timesteps_per_frame))
s_spike_image = axes[1].imshow(s_image_data, interpolation="nearest", vmin=0.0, vmax=float(timesteps_per_frame))


def updatefig(frame):
    global input_image_data, s_image_data, input_image, s_spike_image
    
    # Decay image data
    input_image_data *= 0.75
    s_image_data *= 0.75
    
    # Loop through all timesteps that occur within frame
    input_image_data += timestep_inputs[:,:,frame]
    s_image_data += s_output[:,:,frame]
    
    # Set image data
    input_image.set_array(input_image_data)
    s_spike_image.set_array(s_image_data)
    return [input_image, s_spike_image]

# Play animation
ani = animation.FuncAnimation(fig, updatefig, range(timestep_inputs.shape[2]), interval=timesteps_per_frame, blit=True, repeat=False)
plt.show()