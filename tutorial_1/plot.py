import csv
import matplotlib.pyplot as plt
import numpy as np

with open("tenHHModel_output.V.dat", "rb") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = " ")

    # Skip headers
    csv_reader.next()

    # Read data and zip into columns
    data_columns = zip(*csv_reader)

    # Convert times to numpy
    times = np.asarray(data_columns[0], dtype=float)
    voltages = [np.asarray(data_columns[i], dtype=float) for i in range(1, 11)]

    # Create plot
    figure, axis = plt.subplots()
    axis.set_xlabel("time [ms]")
    axis.set_ylabel("membrane voltage [mV]")
    # Plot voltages
    for v in voltages:
        axis.plot(times, v)

    # Show plot
    plt.show()
