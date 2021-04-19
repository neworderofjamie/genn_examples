import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def get_time(name):
    return int(name.split("_")[2].split(".")[0])

   
max_kernels = 32


    
# Get list of kernels
conv1_kernels = list(sorted(glob("conv1_kernel_*.bin"), key=get_time))
conv2_kernels = list(sorted(glob("conv2_kernel_*.bin"), key=get_time))


conv1_fig, conv1_axes = plt.subplots(16, len(conv1_kernels[-max_kernels:]), sharex="col", sharey="row",
                                     gridspec_kw = {"wspace":0, "hspace":0})
conv2_fig, conv2_axes = plt.subplots(32, len(conv2_kernels[-max_kernels:]), sharex="col", sharey="row",
                                     gridspec_kw = {"wspace":0, "hspace":0})

for t, (conv1, conv2) in enumerate(zip(conv1_kernels[-max_kernels:], conv2_kernels[-max_kernels:])):
    # Load and reshape kernels
    conv1_kernel = np.fromfile(conv1, dtype=np.float32)
    conv2_kernel = np.fromfile(conv2, dtype=np.float32)
    conv1_kernel = np.reshape(conv1_kernel, (5, 5, 16))
    conv2_kernel = np.reshape(conv2_kernel, (5, 5, 16, 32))
    
    # Find which input feature is strongest for each conv2 neuron and feature
    max_conv2_features = np.argmax(conv2_kernel, axis=2)
    
    # Plot seperate conv1 features
    for i in range(16):
        conv1_axes[i,t].imshow(conv1_kernel[:,:,i], cmap="jet")
        conv1_axes[i,t].get_xaxis().set_visible(False)
        conv1_axes[i,t].get_yaxis().set_visible(False)
    
    # Loop through conv2 features
    for f in range(32):
        # Loop through x and y dimension of feature
        visualise_feature = np.empty((25, 25), dtype=np.float32)
        for i in range(5):
            for j in range(5):
                # Select conv1 feature corresponding to strongest feature
                strength = conv2_kernel[i, j, max_conv2_features[i, j, f], f]
                best = conv1_kernel[:,:,max_conv2_features[i, j, f]]
                
                # Insert feature into image for visualisation
                visualise_feature[i * 5: (i * 5) + 5, j * 5:(j * 5) + 5] = best * strength
                
        # Plot visualisation image
        conv2_axes[f,t].imshow(visualise_feature, cmap="jet")
        conv2_axes[f,t].get_xaxis().set_visible(False)
        conv2_axes[f,t].get_yaxis().set_visible(False)
    
    conv1_axes[0,t].set_title(get_time(conv1))
    conv2_axes[0,t].set_title(get_time(conv2))


plt.show()
    