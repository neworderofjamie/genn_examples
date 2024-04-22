import os
import sys
import numpy as np

assert len(sys.argv) > 2

# Get number of neurons
num_pre = int(sys.argv[2])

# Split up path to weight into path and filename
path, weight_filename = os.path.split(sys.argv[1])

# Split into title and extension and check extension is npy
weight_title, weight_ext = os.path.splitext(weight_filename)
assert weight_ext == ".npy"

# Split up title components and check this is a weight file
weight_title_components = weight_title.split("-")
assert len(weight_title_components) == 3
assert weight_title_components[2] == "g"

# Build titles of files containing indices
pre_title = "-".join(weight_title_components[:2] + ["pre_ind"])
post_title = "-".join(weight_title_components[:2] + ["post_ind"])

# Load indices and weight
weights = np.load(sys.argv[1])

pre_path = os.path.join(path, pre_title + ".npy")
post_path = os.path.join(path, post_title + ".npy")

if os.path.exists(pre_path) and os.path.exists(post_path):
    pre_inds = np.load(pre_path)
    post_inds = np.load(post_path)

    assert weights.shape == pre_inds.shape
    assert weights.shape == post_inds.shape

    assert np.all(pre_inds < num_pre)
    row_lengths = np.bincount(pre_inds, minlength=num_pre)
    row_lengths = row_lengths.astype(np.uint32)
    max_row_length = int(np.amax(row_lengths))

    print(f"Max row length: {max_row_length}")

    ragged_inds = np.empty(num_pre * max_row_length, dtype=np.uint32)
    ragged_weight = np.empty(num_pre * max_row_length, dtype=np.float32)
    synapse_order = np.lexsort((post_inds, pre_inds))

    sorted_post_inds = post_inds[synapse_order]
    sorted_weights = weights[synapse_order]

    row_start_idx = range(0, num_pre * max_row_length, max_row_length)
    syn = 0
    for i, r in zip(row_start_idx, row_lengths):
        # Copy row from non-padded indices into correct location
        ragged_inds[i:i + r] = sorted_post_inds[syn:syn + r]
        ragged_weight[i:i + r] = sorted_weights[syn:syn + r]
        syn += r

    ind_title = "-".join(weight_title_components[:2] + ["ragged_ind"])
    row_length_title = "-".join(weight_title_components[:2] + ["row_length"])

    ragged_inds.tofile(ind_title + ".bin")
    ragged_weight.tofile(weight_title + ".bin")
    row_lengths.tofile(row_length_title + ".bin")
else:
    print("Dense")
    weights.tofile(weight_title + ".bin")
