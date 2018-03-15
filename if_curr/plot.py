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
    current = np.asarray(data_columns[2], dtype=float)

    # Create plot
    figure, axes = plt.subplots(2)

    # Plot voltages
    axes[0].set_title("Voltage")
    axes[0].plot(times, voltage)
    
    axes[1].set_title("Input current")
    axes[1].plot(times, current)

    # Show plot
    plt.show()
