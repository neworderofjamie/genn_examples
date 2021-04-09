import matplotlib.pyplot as plt
import numpy as np
from glob import glob

# Get list of input spike trains
input_spikes = list(sorted(glob("input_spikes_*.csv"), 
                           key=lambda n: int(n.split("_")[2].split(".")[0])))

num_rows = int(np.sqrt(len(input_spikes)))
num_cols = int(np.ceil(len(input_spikes) / float(num_rows)))

fig, axes = plt.subplots(num_rows, num_cols, sharex="col", sharey="row")

for i, c in enumerate(input_spikes):
    # Load spikes
    spikes = np.loadtxt(c, skiprows=1, delimiter=",")
    
    # Count spikes for each pixel
    spike_counts = np.bincount(spikes[:,1].astype(int), minlength=28 * 28)
    spike_counts = np.reshape(spike_counts, (28, 28))
    
    axis = axes[i // num_cols,i % num_cols]
    axis.imshow(spike_counts, cmap="jet")
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
    
plt.show()
    