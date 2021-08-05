import math
import sys
from matplotlib import pyplot as plt
from plot_performance import plot

directories = sys.argv[1:]

num_axis = int(math.ceil(math.sqrt(len(directories))))

fig, axes = plt.subplots(num_axis, num_axis, sharex="col", sharey="row")

for i, (a, d) in enumerate(zip(axes.flatten(), directories)):
    a.set_title(d)
    plot(d, a)
    if i == 0:
        a.legend()

for i in range(num_axis):
    axes[-1, i].set_xlabel("Epoch")
    axes[i, 0].set_ylabel("Performance [%]")
plt.show()
