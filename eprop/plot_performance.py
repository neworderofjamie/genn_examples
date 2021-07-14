from glob import glob
import numpy as np
from matplotlib import pyplot as plt

prefix = ""#"hsd_"
files = list(glob(prefix + "performance*.csv"))

fig, axis = plt.subplots()
for f in files:
    print(f)
    data = np.loadtxt(f, delimiter=",", skiprows=1)
    
    # Count epochs
    #num_epochs = len(np.unique(data[:,0]))
    epochs = np.unique(data[:,0])
    
    num_trials = np.empty_like(epochs)
    num_correct = np.empty_like(epochs)
    
    for i, e in enumerate(epochs):
        epoch_mask = (data[:,0] == e)
        
        num_trials[i] = np.sum(data[epoch_mask,2])
        num_correct[i] = np.sum(data[epoch_mask,3])

    axis.plot(num_correct / num_trials, label=f[len(prefix) + 12:-4], marker="x")
axis.set_xlabel("Epoch")
axis.set_ylabel("Training performance")
axis.legend()
plt.show()