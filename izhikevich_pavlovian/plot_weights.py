import csv
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import binned_statistic

with open("weight_evolution.csv", "rb") as weight_evolution_file:
    # Create reader
    weight_evolution_reader = csv.reader(weight_evolution_file, delimiter=",")

    # Read columns
    weight_evolution_columns = zip(*weight_evolution_reader)

    # Convert CSV columns to numpy
    mean_weights = np.asarray(weight_evolution_columns[0], dtype=float)
    mean_rewarded_weights = np.asarray(weight_evolution_columns[1], dtype=float)

    # Calculate corresponding times in minutes (samples are taken every 10 seconds)
    sample_time_m = 10.0 / 60.0
    time_m = np.arange(0, sample_time_m * len(mean_weights), sample_time_m)

    # Create plot
    figure, axis = plt.subplots()

    axis.plot(time_m, mean_weights, label="Mean weights")
    axis.plot(time_m, mean_rewarded_weights, label="Mean weights from S0")

    axis.set_xlabel("Time [minutes]")
    axis.set_ylabel("Weight [mV]")

    axis.legend()
    plt.show()