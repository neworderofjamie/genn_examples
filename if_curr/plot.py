import csv
import matplotlib.pyplot as plt
import numpy as np

with open("voltages.csv", "rb") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ",")

    # Skip headers
    csv_reader.next()

    # Read data and zip into columns
    data_columns = zip(*csv_reader)

    # Convert times to numpy
    times = np.asarray(data_columns[0], dtype=float)
    voltage = np.asarray(data_columns[1], dtype=float)

    # Create plot
    figure, axis = plt.subplots()

    # Plot voltages
    axis.plot(times, voltage)

    # Show plot
    plt.show()
