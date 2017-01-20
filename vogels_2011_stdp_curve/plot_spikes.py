import csv
import matplotlib.pyplot as plt
import numpy as np

with open("spikes.csv", "rb") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ",")

    # Skip headers
    csv_reader.next()

    # Read data and zip into columns
    data_columns = zip(*csv_reader)

    # Convert times to numpy
    times = np.asarray(data_columns[0], dtype=float)
    neuron_id = np.asarray(data_columns[1], dtype=int)

    # Create plot
    figure, axis = plt.subplots()

    # Plot voltages
    axis.scatter(times, neuron_id)

    # Show plot
    plt.show()
