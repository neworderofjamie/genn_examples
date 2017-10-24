import csv
import matplotlib.pyplot as plt
import numpy as np

def load_csv(filename):
    with open(filename, "rb") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ",")

        # Skip headers
        csv_reader.next()

        # Read data and zip into columns
        data_columns = zip(*csv_reader)

        # Convert CSV columns to numpy
        neuron_id = np.asarray(data_columns[1], dtype=int)
        value = np.asarray(data_columns[2], dtype=float)

        # Reshape value into 2D array and transpose
        value = np.reshape(value, (-1, len(np.unique(neuron_id))))
        return np.transpose(value)


tn2 = load_csv("tn2.csv")
cl1 = load_csv("cl1.csv")
tb1 = load_csv("tb1.csv")
cpu4 = load_csv("cpu4.csv")
cpu1 = load_csv("cpu1.csv")

fig, axes = plt.subplots(5, sharex=True)

# Labels
axes[0].set_ylabel("TN2\n(speed)")
axes[1].set_ylabel("CL1")
axes[1].set_yticks([0, 7, 15])
axes[2].set_ylabel("TB1")
axes[3].set_ylabel("CPU4")
axes[4].set_ylabel("CPU1")
axes[4].set_xlabel("Time [steps]")

# Plot
axes[0].plot(tn2[0,])
axes[0].plot(tn2[1,])
axes[1].imshow(cl1, aspect=32.0)
axes[2].imshow(tb1, aspect=64.0)
axes[3].imshow(cpu4, aspect=32.0)
axes[4].imshow(cpu1, aspect=32.0)

plt.show()