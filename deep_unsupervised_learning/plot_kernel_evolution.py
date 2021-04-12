import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def get_time(name):
    return int(name.split("_")[2].split(".")[0])
   
max_kernels = 32

# Get list of kernels
conv1_kernels = list(sorted(glob("conv1_kernel_*.bin"), key=get_time))

fig, axes = plt.subplots(16, len(conv1_kernels[-max_kernels:]), sharex="col", sharey="row",
                         gridspec_kw = {"wspace":0, "hspace":0})

for j, c in enumerate(conv1_kernels[-max_kernels:]):
    # Load and reshape kernel
    kernel = np.fromfile(c, dtype=np.float32)
    kernel = np.reshape(kernel, (5, 5, 16))
    
    # Plot seperate channels
    for i in range(16):
        axes[i,j].imshow(kernel[:,:,i], cmap="jet")
        axes[i,j].get_xaxis().set_visible(False)
        axes[i,j].get_yaxis().set_visible(False)
    
    axes[0,j].set_title(get_time(c))

plt.show()
    