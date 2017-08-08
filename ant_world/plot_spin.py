import csv
import matplotlib.pyplot as plt
import numpy as np


def get_csv_columns(csv_file, headers=True):
    # Create reader
    reader = csv.reader(csv_file, delimiter=",")

    # Skip headers if required
    if headers:
        reader.next()

    # Read columns and return
    return zip(*reader)

def get_column_safe(data, column, dtype):
    if column < len(data):
        return np.asarray(data[column], dtype=dtype)
    else:
        return []

with open("spin.csv", "rb") as spin_file:

    # Read data and zip into columns
    spin_columns = get_csv_columns(spin_file)

    # Convert CSV columns to numpy
    angle = get_column_safe(spin_columns, 0, float)
    en_spikes = get_column_safe(spin_columns, 1, int)

    # Create plot
    figure, axis = plt.subplots()

    axis.plot(angle, en_spikes)

    # Configure axes
    axis.set_ylabel("Angle [degrees]")
    axis.set_ylabel("Num EN spikes")


    # Show plot
    plt.show()

