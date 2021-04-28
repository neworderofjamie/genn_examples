import matplotlib.pyplot as plt
import numpy as np
from glob import glob
   
MAX_KERNELS = 1

def get_time(name):
    return int(name.split("_")[2].split(".")[0])

def plot(conv1_kernel, conv2_kernel, conv1_axes, conv2_axes):
    # Reshape kernels
    conv1_kernel = np.reshape(conv1_kernel, (5, 5, 16))
    conv2_kernel = np.reshape(conv2_kernel, (5, 5, 16, 32))
    
    # Find which input feature is strongest for each conv2 neuron and feature
    max_conv2_features = np.argmax(conv2_kernel, axis=2)

    # Plot seperate conv1 features
    for f in range(16):
        conv1_axes[f].imshow(conv1_kernel[:,:,f], cmap="gray")
        conv1_axes[f].get_xaxis().set_visible(False)
        conv1_axes[f].get_yaxis().set_visible(False)
    
    # Loop through conv2 features
    difference = 0.0
    for f in range(32):
        # Loop through x and y dimension of feature
        visualise_feature = np.empty((25, 25), dtype=np.float32)
        for i in range(5):
            for j in range(5):
                # Select conv1 feature corresponding to strongest feature
                strength = conv2_kernel[i, j, max_conv2_features[i, j, f], f]
                best = conv1_kernel[:,:,max_conv2_features[i, j, f]]
                
                # Insert feature into image for visualisation
                visualise_feature[i * 5: (i * 5) + 5, 
                                  j * 5: (j * 5) + 5] = best * strength

        # Plot visualisation image
        conv2_axes[f].imshow(visualise_feature, cmap="gray")
        #conv2_axes[f].imshow(max_conv2_features[:,:,f], cmap="jet")
        conv2_axes[f].get_xaxis().set_visible(False)
        conv2_axes[f].get_yaxis().set_visible(False)
        
        for g in range(32):
            if g == f:
                continue
            
            kernel_difference = conv2_kernel[:, :, :, f] - conv2_kernel[:, :, :, g] 
            difference += np.sqrt(np.sum(kernel_difference ** 2.0))
    print(difference)


# Get list of kernels
conv1_kernels = list(sorted(glob("conv1_kernel_*.bin"), key=get_time))
conv2_kernels = list(sorted(glob("conv2_kernel_*.bin"), key=get_time))

if MAX_KERNELS == 1:
    fig, axes = plt.subplots(3, 16)
    
    # Load and reshape kernels
    conv1_kernel = np.fromfile(conv1_kernels[-1], dtype=np.float32)
    conv2_kernel = np.fromfile(conv2_kernels[-1], dtype=np.float32)
    
    # Plot
    plot(conv1_kernel, conv2_kernel, axes[0,:], axes[1:3,:].flatten())
else:
    conv1_fig, conv1_axes = plt.subplots(16, len(conv1_kernels[-MAX_KERNELS:]), sharex="col", sharey="row",
                                         gridspec_kw = {"wspace":0, "hspace":0})
    conv2_fig, conv2_axes = plt.subplots(32, len(conv2_kernels[-MAX_KERNELS:]), sharex="col", sharey="row",
                                         gridspec_kw = {"wspace":0, "hspace":0})

    for t, (conv1, conv2) in enumerate(zip(conv1_kernels[-MAX_KERNELS:], conv2_kernels[-MAX_KERNELS:])):
        # Load and reshape kernels
        conv1_kernel = np.fromfile(conv1, dtype=np.float32)
        conv2_kernel = np.fromfile(conv2, dtype=np.float32)
        
        
        plot(conv1_kernel, conv2_kernel, conv1_axes[:,t], conv2_axes[:,t])
        
        conv1_axes[0,t].set_title(get_time(conv1))
        conv2_axes[0,t].set_title(get_time(conv2))

plt.show()
    