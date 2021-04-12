import matplotlib.pyplot as plt
import numpy as np


# Load and reshape weights
weights = np.fromfile("input_output_30300.bin", dtype=np.float32)
weights = np.reshape(weights, (784, -1))

grid_size = int(np.ceil(np.sqrt(weights.shape[1])))

fig, axes = plt.subplots(grid_size, grid_size, sharex="col", sharey="row", 
                         gridspec_kw = {"wspace":0, "hspace":0})

for i in range(grid_size):
    for j in range(grid_size):
        c = (i * grid_size) + j
    
        if c < weights.shape[1]:
            axes[i,j].imshow(weights[:,c].reshape((28, 28)), cmap="jet")
            axes[i,j].get_xaxis().set_visible(False)
            axes[i,j].get_yaxis().set_visible(False)
        else:
            axes[i,j].remove()

fig.tight_layout(pad=0, h_pad=0, w_pad=0)
plt.show()
    