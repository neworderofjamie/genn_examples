import csv
import numpy as np
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import sys

original_resolution = 128
output_resolution = 32

# How many 
timesteps_per_frame = 33


def read_input_spikes(filename, num_frames):
    timestep_inputs = np.zeros((output_resolution, output_resolution, num_frames), dtype=float)
    
    with open(filename, "r") as input_spike_file:
        # Read input spike file
        scale = original_resolution / output_resolution
        for line in input_spike_file:
            # Split lines into time and keys
            time_string, keys_string = line.split(";")
            time = int(time_string)
            frame = time // timesteps_per_frame
            
            if frame >= num_frames:
                print("Warning: input overruns %u/%u frames" % (frame, num_frames))
                continue
                
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
        return timestep_inputs

def read_s_voltage():
    with open("s_voltages.csv", "rb") as s_v_file:
        # Create CSV reader
        s_v_csv_reader = csv.reader(s_v_file, delimiter = ",")
        
        # Skip headers
        s_v_csv_reader.next()

        # Read data and zip into columns
        s_v_columns = zip(*s_v_csv_reader)
        
        # Convert CSV columns to numpy
        s_v = np.asarray(s_v_columns[2], dtype=float)

        # Build 3D histogram i.e. video frames from this data
        return  np.reshape(s_v, (-1, output_resolution, output_resolution))
        

def read_lgmd_voltage():
    with open("lgmd_voltages.csv", "rb") as lgmd_v_file:
        # Create CSV reader
        lgmd_v_csv_reader = csv.reader(lgmd_v_file, delimiter = ",")
        
        # Skip headers
        lgmd_v_csv_reader.next()

        # Read data and zip into columns
        lgmd_v_columns = zip(*lgmd_v_csv_reader)
        
         # Convert CSV columns to numpy
        return np.asarray(lgmd_v_columns[2], dtype=float)

def read_lgmd_spikes():
    with open("lgmd_spikes.csv", "rb") as lgmd_spike_file:
        # Create CSV reader
        lgmd_spike_csv_reader = csv.reader(lgmd_spike_file, delimiter = ",")
        
        # Skip headers
        lgmd_spike_csv_reader.next()

        # Read data and zip into columns
        lgmd_spike_columns = zip(*lgmd_spike_csv_reader)
        
         # Convert CSV columns to numpy
        return np.asarray(lgmd_spike_columns[0], dtype=float)

s_output = read_s_voltage()
num_frames = int(np.ceil(float(s_output.shape[0]) / float(timesteps_per_frame)))

timestep_inputs = read_input_spikes(sys.argv[1], num_frames)
lgmd_output = read_lgmd_voltage()
lgmd_spikes = read_lgmd_spikes()

fig = plt.figure()

p_axis = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
p_axis.set_title("Retina spikes")

s_axis = plt.subplot2grid((3, 4), (0, 2), colspan=2, rowspan=2)
s_axis.set_title("Summing population membrane voltage")

lgmd_axis = plt.subplot2grid((3, 4), (2, 0), colspan=4)
lgmd_axis.set_title("LGMD membrane voltage")
lgmd_axis.set_ylim((-10.0, 0.25))
lgmd_axis.plot(lgmd_output, label="LGMD")

# Mark spikes
lgmd_axis.vlines(lgmd_spikes, -10.0, 0.25, color="red")

print("Maximum summing %f" % np.amax(s_output))

input_image_data = timestep_inputs[:,:,0]
input_image = p_axis.imshow(input_image_data, interpolation="nearest", cmap="jet", vmin=0.0, vmax=float(timesteps_per_frame))
s_v_image = s_axis.imshow(s_output[0,:,:], interpolation="nearest", cmap="jet", vmin=0.0, vmax=np.amax(s_output))

border = s_axis.add_patch(
    patches.Rectangle(
        (6, 6),   # (x,y)
        20,          # width
        20,          # height
        fill=False,
        color="white",
        linewidth=2.0))
    
current_time = lgmd_axis.axvline(0, 0, 1, color="black")

def updatefig(frame):
    global input_image_data, input_image, s_v_image, border, timesteps_per_frame, current_time
    
    # Decay image data
    input_image_data *= 0.9
 
    # Loop through all timesteps that occur within frame
    input_image_data += timestep_inputs[:,:,frame]

    # Set image data
    input_image.set_array(input_image_data)
    
    # Get 3D block of voltage timestep data representing to show in this frame and show average
    s_timesteps_in_frame = s_output[frame * timesteps_per_frame:(frame + 1) * timesteps_per_frame,:,:]
    s_v_image.set_array(np.average(s_timesteps_in_frame, axis = 0))
    
    current_time.set_xdata([frame * timesteps_per_frame])
    
    # Return list of artists which we have updated
    # **YUCK** order of these dictates sort order
    # **YUCK** score_text must be returned whether it has
    # been updated or not to prevent overdraw
    return [input_image, s_v_image, border, current_time]

# Play animation
ani = animation.FuncAnimation(fig, updatefig, range(timestep_inputs.shape[2]), interval=timesteps_per_frame, blit=True, repeat=True)
plt.show()