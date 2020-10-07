import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_timeslice(input_spikes, recurrent_spikes, output_data, trial, axes):
    start_time = trial * 1000.0
    end_time = start_time + 1000.0
    
    output_data_mask = (output_data["time"] >= start_time) & (output_data["time"] < end_time)
    output_data = output_data[output_data_mask]
   
    y1_error =  output_data["y1"] - output_data["y_star1"]
    y2_error =  output_data["y2"] - output_data["y_star2"]
    y3_error =  output_data["y3"] - output_data["y_star3"]

    # Plot Y ands Y*
    axes[0].set_title("Y0 (MSE=%f)" % (np.sum(y1_error * y1_error) / float(len(y1_error))))
    axes[0].plot(output_data["time"], output_data["y1"], color="blue")
    axes[0].plot(output_data["time"], output_data["y_star1"], color="blue", linestyle="--", alpha=0.5)
    axes[0].set_xlim((start_time, end_time))
    axes[0].set_ylim((-3.0, 3.0))

    axes[1].set_title("Y1 (MSE=%f)" % (np.sum(y2_error * y2_error) / float(len(y2_error))))
    axes[1].plot(output_data["time"], output_data["y2"], color="blue")
    axes[1].plot(output_data["time"], output_data["y_star2"], color="blue", linestyle="--", alpha=0.5)
    axes[1].set_ylim((-3.0, 3.0))

    axes[2].set_title("Y2 (MSE=%f)" % (np.sum(y3_error * y3_error) / float(len(y3_error))))
    axes[2].plot(output_data["time"], output_data["y3"], color="blue")
    axes[2].plot(output_data["time"], output_data["y_star3"], color="blue", linestyle="--", alpha=0.5)
    axes[2].set_ylim((-3.0, 3.0))

    axes[3].set_title("X")
    axes[3].scatter(input_spikes["time"], input_spikes["neuron_id"], s=2, edgecolors="none")

    axes[4].set_title("Recurrent")
    axes[4].scatter(recurrent_spikes["time"], recurrent_spikes["neuron_id"], s=2, edgecolors="none")
    plt.setp(axes[4].xaxis.get_majorticklabels(), rotation="vertical")

# Read data
input_spikes = np.loadtxt("input_spikes.csv", delimiter=",", skiprows=1,
                          dtype={"names": ("time", "neuron_id"),
                                 "formats": (np.float, np.int)})
recurrent_spikes = np.loadtxt("recurrent_spikes.csv", delimiter=",", skiprows=1,
                              dtype={"names": ("time", "neuron_id"),
                                     "formats": (np.float, np.int)})
output_data = np.loadtxt("output.csv", delimiter=",",
                         dtype={"names": ("time", "y1", "y2", "y3", "y_star1", "y_star2", "y_star3"),
                                "formats": (np.float, np.float, np.float, np.float, np.float, np.float, np.float)})

# Create mask to select the recurrent spikes from 20 random neurons
subset_recurrent_neuron_ids = np.sort(np.random.choice(600, 20, replace=False))
subset_recurrent_spikes = np.isin(recurrent_spikes["neuron_id"], 
                                  subset_recurrent_neuron_ids)


recurrent_spikes = recurrent_spikes[:][subset_recurrent_spikes]
recurrent_spikes["neuron_id"] = np.digitize(recurrent_spikes["neuron_id"], subset_recurrent_neuron_ids, right=True)

# Create plot
figure, axes = plt.subplots(5, 6, sharex="col")

plot_timeslice(input_spikes, recurrent_spikes, output_data, 0, axes[:,0])
plot_timeslice(input_spikes, recurrent_spikes, output_data, 200, axes[:,1])
plot_timeslice(input_spikes, recurrent_spikes, output_data, 400, axes[:,2])
plot_timeslice(input_spikes, recurrent_spikes, output_data, 600, axes[:,3])
plot_timeslice(input_spikes, recurrent_spikes, output_data, 800, axes[:,4])
plot_timeslice(input_spikes, recurrent_spikes, output_data, 1000, axes[:,5])

# Show plot
#figure.tight_layout(pad=0)
plt.show()
