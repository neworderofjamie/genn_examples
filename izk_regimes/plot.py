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
    neuron_id = np.asarray(data_columns[1], dtype=int)
    voltage = np.asarray(data_columns[2], dtype=float)

    # Check there are 4 neurons
    neuron_ids = np.unique(neuron_id)
    assert len(neuron_ids) == 4

    # Create plot
    figure, axes = plt.subplots(2, 2)

    # Loop through neurons
    for n in neuron_ids:
        # Convert neuron id to axex coordinate
        x = n // 2
        y = n % 2

        # Pick rows corresponding to this neuron
        mask = neuron_id == n

        # Plot voltages
        axes[x, y].plot(times[mask], voltage[mask])

    # Show plot
    plt.show()
