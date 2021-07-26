from glob import glob
import numpy as np
import os
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from tonic_classifier_parser import parse_arguments

# Parse command line
output_directory = parse_arguments(description="Plot eProp classifier performance")[1]

fig, axis = plt.subplots()

# Load training data
training_data = np.loadtxt(os.path.join(output_directory, "performance.csv"), delimiter=",", skiprows=1)

# Count epochs
epochs = np.unique(training_data[:,0])

num_trials = np.empty_like(epochs)
num_correct = np.empty_like(epochs)

for i, e in enumerate(epochs):
    epoch_mask = (training_data[:,0] == e)
    
    num_trials[i] = np.sum(training_data[epoch_mask,2])
    num_correct[i] = np.sum(training_data[epoch_mask,3])

print("Max training performance: %f %%" % (100.0 * np.amax(num_correct / num_trials)))
axis.plot(100.0 * num_correct / num_trials, label="Training")

# Find evaluation files, sorting numerically
evaluate_files = list(sorted(glob(os.path.join(output_directory, "performance_evaluate_*.csv")),
                             key=lambda x: int(os.path.basename(x)[21:-4])))

# Loop through evaluate files
test_epoch = []
test_performance = []
for e in evaluate_files:
    # Extract epoch number
    epoch = int(os.path.basename(e)[21:-4])

    # Load file
    test_data = np.loadtxt(e, delimiter=",", skiprows=1)
    
    # Calculate performance
    num_trials = np.sum(test_data[:,1])
    num_correct = np.sum(test_data[:,2])
    
    # Add to list
    test_performance.append(100.0 * num_correct / num_trials)
    test_epoch.append(epoch)

# Plot
axis.plot(test_epoch, test_performance, label="Testing")

print("Max training performance: %f %%" % max(test_performance))
axis.set_xlabel("Epoch")
axis.set_ylabel("Performance [%]")
axis.legend()
plt.show()