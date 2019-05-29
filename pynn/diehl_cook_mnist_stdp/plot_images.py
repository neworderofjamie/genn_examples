import numpy as np
import matplotlib.pyplot as plt
from sys import argv

assert len(argv) == 2

data = np.load(argv[1])

image_indices = np.random.choice(data.shape[0], 100)

fig, axes = plt.subplots(10, 10)
for i, idx in enumerate(image_indices):
    image = np.reshape(data[idx,:], (28,28))
    axes[i // 10,i % 10].imshow(image)
plt.show()
