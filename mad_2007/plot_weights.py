import matplotlib.pyplot as plt
import numpy as np

weights = np.fromfile("weights.bin", dtype=np.float32)

fig, axis = plt.subplots()
axis.hist(weights, 30, density=True)
plt.show()
