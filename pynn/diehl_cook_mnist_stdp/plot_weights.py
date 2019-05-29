import numpy as np
import matplotlib.pyplot as plt

weights = np.load("weights.npy")

reshaped_weights = weights.reshape((100, 784))

fig,axes = plt.subplots(10, 10)
for i in range(100):
    x = i % 10
    y = i // 10
    a = axes[x,y]
    rf = reshaped_weights[i,:]
    rf_image = rf.reshape((28,28))
    a.imshow(rf_image)
plt.show()

