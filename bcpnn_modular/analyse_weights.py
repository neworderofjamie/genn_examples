import argparse
import csv
import copy
import itertools
import numpy
import pylab
import sys

NE = 1000

def load_masked_weights(filename):
    # Initialise combined weights to nan
    combined_masked_weights = numpy.ma.empty((NE, NE), fill_value=0.0)
    combined_masked_weights[:] = numpy.ma.masked
    
    sparse = numpy.loadtxt(filename, skiprows=1, delimiter=",",
                           dtype={'names': ('i', 'j', 'weight'), 'formats': ('i4', 'i4', 'f4')})
    
    combined_masked_weights[sparse["i"], sparse["j"]] = sparse["weight"]
    return combined_masked_weights
    

def display_raw_weights(masked_weights, figure, axis):
    image = axis.imshow(masked_weights, cmap="jet", interpolation="none")
    axis.set_xlabel("Post neuron ID")
    axis.set_ylabel("Pre neuron ID")
    figure.colorbar(image)

def display_mean_weights(masked_weights, figure, axis, num_mcu_neurons, palette):
    # Calculate number of minicolumns
    num_minicolumns = masked_weights.shape[0] // num_mcu_neurons
    mean_weights = numpy.zeros((num_minicolumns, num_minicolumns))
    for mi, mj in itertools.product(range(num_minicolumns), repeat=2):
        slice_i = slice(mi * num_mcu_neurons, (mi + 1) * num_mcu_neurons)
        slice_j = slice(mj * num_mcu_neurons, (mj + 1) * num_mcu_neurons)
        sub_weights = masked_weights[slice_i, slice_j]

        mean_weights[mi, mj] = numpy.ma.mean(sub_weights)

    image = axis.imshow(mean_weights, cmap=palette, interpolation="none")
    return image

def display_single_attractor(masked_weights, mi, axis, num_mcu_neurons):
    # Calculate number of minicolumns
    num_minicolumns = masked_weights.shape[0] / num_mcu_neurons

    slice_i = slice(mi * num_mcu_neurons, (mi + 1) * num_mcu_neurons)

    mean_weights = [numpy.ma.mean(masked_weights[slice_i, slice(mj * num_mcu_neurons, (mj + 1) * num_mcu_neurons)])
                    for mj in range(num_minicolumns)]

    axis.plot(mean_weights, marker="x")
    axis.axhline(linestyle="--", color="gray")
    axis.axvline(mi, linestyle="--", color="gray")
    axis.set_xlabel("Post minicolumn ID")
    axis.set_ylabel("Mean weight")

def combine_connection_weights(filenames):
    # Initialise combined weights to nan
    combined_masked_weights = numpy.ma.empty((NE, NE), fill_value=0.0)
    combined_masked_weights[:] = numpy.ma.masked

    # Loop through all connectors, 256
    for f in filenames:
        # Load weights
        masked_weights = load_masked_weights(f)
        combined_masked_weights = numpy.ma.array(combined_masked_weights.data + masked_weights.data,
                                                 mask=numpy.logical_and(combined_masked_weights.mask, masked_weights.mask),
                                                 fill_value=0.0)
    return combined_masked_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine together weights recorded from multiple HCUs and display")
    parser.add_argument("--num_hcus", type=int, default=9, help="How many HCUs is data for")
    parser.add_argument("--num_mcu_neurons", type=int, default=100, help="How many neurons make up an MCU")
    parser.add_argument("--selected_attractor", type=int, default=7, help="Which attractor to plot")
    parser.add_argument("suffix", nargs=1, help="Suffix for filenames")
    args = parser.parse_args()

    figure, axes = pylab.subplots(1, 2)

    # Build filenames from command line
    filenames = ["E_%u_E_%u_%s.csv" % (i, j, args.suffix[0]) 
                 for i, j in itertools.product(range(args.num_hcus), repeat=2)]
    combined_masked_weights = combine_connection_weights(filenames)

    # Display weights
    display_single_attractor(combined_masked_weights, args.selected_attractor, axes[0], args.num_mcu_neurons)
    mean_image = display_mean_weights(combined_masked_weights, figure, axes[1], args.num_mcu_neurons, "jet")
    axes[1].set_xlabel("Post-synaptic attractor number")
    axes[1].set_ylabel("Pre-synaptic attractor number")

    figure.colorbar(mean_image, shrink=0.75)
    #display_raw_weights(combined_masked_weights, figure, axes[2])
    pylab.show()
