import csv
import matplotlib.pyplot as plt
import numpy as np

with open("weights.csv", "rb") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ",")

    # Skip headers
    csv_reader.next()

    # Read data and zip into columns
    data_columns = zip(*csv_reader)

    # Convert times to numpy
    delta_t = np.asarray(data_columns[0], dtype=float)
    weight = np.asarray(data_columns[1], dtype=float)

    #weight = (weight - -0.5) / 0.5

    # Create plot
    figure, axis = plt.subplots()

    # Add axis lines
    axis.axhline(0.0, color="black")
    axis.axvline(0.0, color="black")


    # Plot voltages
    axis.plot(delta_t, weight)

    # Show plot
    plt.show()
