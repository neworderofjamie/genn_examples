import matplotlib.pyplot as plt
import numpy as np

num_batches_per_epoch = 118

test_accuracy = np.asarray([48.94, 50.66, 55.97, 55.73, 56.93, 59.66, 62.87, 56.49, 56.24])

# Load performance data
performance = np.genfromtxt("performance.csv", delimiter=",", skip_header=1)
assert performance.shape[1] == 4

# Reshape performance into epochs
performance = np.reshape(performance, (-1,118,4))

# Calculate sum of correct trials per epoch
epoch_performance = np.sum(performance[:,:,3], axis=1)

# Crate figure
fig, axis = plt.subplots()

# Plot accuracy per epoch
axis.plot(epoch_performance / 60000.0, label="Train accuracy")
axis.plot(test_accuracy / 100.0, label="Test accuracy")
axis.set_ylabel("Accuracy")
axis.set_xlabel("Epoch")

axis.set_ylim((0.0, 1.0))
axis.legend()
plt.show()