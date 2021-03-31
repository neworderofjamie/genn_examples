import matplotlib.pyplot as plt
import numpy as np
from glob import glob

# Get list of kernels
conv1_kernels = list(sorted(glob("conv1_kernel_*.bin"), 
                            key=lambda n: int(n.split("_")[2].split(".")[0])))

conv1_kernels = conv1_kernels[::10]

fig, axes = plt.subplots(len(conv1_kernels), 16, sharex="col", sharey="row")

for i, c in enumerate(conv1_kernels):
    # Load and reshape kernel
    kernel = np.fromfile(c, dtype=np.float32)
    kernel = np.reshape(kernel, (5, 5, 16))
    
    # Plot seperate channels
    for j in range(16):
        axes[i,j].imshow(kernel[:,:,j], cmap="Greys")

plt.show()
    