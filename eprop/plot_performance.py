from glob import glob
import numpy as np
import os
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from tonic_classifier_parser import parse_arguments

# Parse command line
name_suffix, output_directory, _ = parse_arguments(description="Plot eProp classifier performance")

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

max_train_performance = 100.0 * np.amax(num_correct / num_trials)
print("Max training performance: %f %%" % max_train_performance)
train_actor = axis.plot(100.0 * num_correct / num_trials, label="Training")[0]
axis.axhline(max_train_performance, linestyle="--", color=train_actor.get_color())

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

if len(test_performance) > 0:
    # Plot
    test_actor = axis.plot(test_epoch, test_performance, label="Testing")[0]
    axis.axhline(max(test_performance), linestyle="--", color=test_actor.get_color())
    print("Max testing performance: %f %%" % max(test_performance))

axis.set_title(name_suffix)
axis.set_xlabel("Epoch")
axis.set_ylabel("Performance [%]")
axis.set_ylim((0, 100))
axis.legend()
plt.show()