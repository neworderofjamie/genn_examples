import math
import sys
from matplotlib import pyplot as plt
from plot_performance import plot

directories = sys.argv[1:]

num_results = len(directories)
num_cols = int(math.ceil(math.sqrt(num_results)))
num_rows = int(math.ceil(float(num_results) / num_cols))

print(num_cols, num_rows, num_results)
fig, axes = plt.subplots(num_rows, num_cols, sharex="col", sharey="row")

for i, (a, d) in enumerate(zip(axes.flatten(), directories)):
    a.set_title(d)
    plot(d, a)
    if i == 0:
        a.legend()

for i in range(num_rows):
    axes[i, 0].set_ylabel("Performance [%]")

for i in range(num_cols):
    axes[-1, i].set_xlabel("Epoch")

plt.show()
