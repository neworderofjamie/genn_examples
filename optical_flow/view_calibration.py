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

def read_spikes(filename):
    with open(filename, "rb") as spike_file:
        # Create CSV reader
        spike_csv_reader = csv.reader(spike_file, delimiter = ",")

        # Skip headers
        spike_csv_reader.next()

        # Read data and zip into columns
        spike_columns = zip(*spike_csv_reader)

        # Convert CSV columns to numpy
        #if len(lgmd_spike_columns) == 0:
        #    return np.asarray([], dtype=float)
        #else:
        return (np.asarray(spike_columns[0], dtype=float),
                np.asarray(spike_columns[1], dtype=int))

dvs_spikes = read_spikes("dvs_pixel_spikes.csv")
macro_pixel_spikes = read_spikes("macro_pixel_spikes.csv")

dvs_x = np.remainder(dvs_spikes[1], 128)
dvs_y = np.floor_divide(dvs_spikes[1], 128)
dvs_histogram = np.histogram2d(dvs_x, dvs_y, (range(41, 91, 5), range(41, 91, 5)))[0]
print dvs_histogram

macro_pixel_x = np.remainder(macro_pixel_spikes[1], 9)
macro_pixel_y = np.floor_divide(macro_pixel_spikes[1], 9)
macro_pixel_histogram = np.histogram2d(macro_pixel_x, macro_pixel_y, (range(10), range(10)))[0]
print macro_pixel_histogram

ratio = dvs_histogram / macro_pixel_histogram
print np.average(ratio[:,1:])