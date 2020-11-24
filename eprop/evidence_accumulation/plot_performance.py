import matplotlib.pyplot as plt
import numpy as np

# Load performance data
performance = np.genfromtxt("performance.csv", delimiter=",", skip_header=1)
assert performance.shape[1] == 3

# Determine where number of cues changes
edges = np.where(np.roll(performance[:,1],1) != performance[:,1])[0]

# Crate figure
fig, axis = plt.subplots()

# Plot decision error against epoch
axis.plot(performance[:,0], 1.0 - (performance[:,2] / 64.0))

# Plot locations o 
axis.vlines(performance[edges[1:],0], ymin=0.0, ymax=0.6, linestyle="--", color="gray")

# Plot stopping criteria
axis.axhline(1.0 - (58.0 / 64.0), linestyle="--", color="red")

axis.set_ylabel("Decision error")
axis.set_xlabel("Epoch")

axis.set_ylim((0.0, 0.6))
plt.show()