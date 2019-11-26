import numpy as np 
import matplotlib.pyplot as plt 

import time
import os 
from struct import unpack

from gzip import decompress
from urllib import request
from pygenn import genn_model, genn_wrapper


# ********************************************************************************
#                      Methods
# ********************************************************************************
def get_image_data(url, filename, correct_magic, max):
    if os.path.exists(filename):
        print("Loading existing data")
        return np.load(filename)
    else:
        print("Downloading dataset")
        with request.urlopen(url) as response:
            print("Decompressing dataset")
            image_data = decompress(response.read())

            # Unpack header from first 16 bytes of buffer
            magic, num_items, num_rows, num_cols = unpack('>IIII', image_data[:16])
            assert magic == correct_magic
            assert num_rows == 28
            assert num_cols == 28

            # Convert remainder of buffer to numpy bytes
            image_data_np = np.copy(np.frombuffer(image_data[16:], dtype=np.uint8))

            # Reshape data into individual images
            image_data_np = np.reshape(image_data_np, (num_items, num_rows * num_cols))

            # Get range of image data
            image_min = np.amin(image_data_np)
            image_max = np.amax(image_data_np)
            
            # Normalize and convert to float
            image_data_np -= image_min
            image_data_np = image_data_np.astype(np.float)
            image_data_np /= float(image_max - image_min)
            
            # Clamp as desired
            image_data_np = np.minimum(image_data_np, max)
            
            # Write to disk
            np.save(filename, image_data_np)

            return image_data_np

def get_label_data(url, filename, correct_magic):
    if os.path.exists(filename):
        print("Loading existing data")
        return np.load(filename)
    else:
        print("Downloading dataset")
        with request.urlopen(url) as response:
            print("Decompressing dataset")
            label_data = decompress(response.read())

            # Unpack header from first 8 bytes of buffer
            magic, num_items = unpack('>II', label_data[:8])
            assert magic == correct_magic

            # Convert remainder of buffer to numpy bytes
            label_data_np = np.frombuffer(label_data[8:], dtype=np.uint8)
            assert label_data_np.shape == (num_items,)

            # Write to disk
            np.save(filename, label_data_np)

            return label_data_np

def get_training_data(max):
    images = get_image_data("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "training_images.npy", 2051, max)
    labels = get_label_data("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "training_labels.npy", 2049)
    assert images.shape[0] == labels.shape[0]

    return images, labels

def get_testing_data(max):
    images = get_image_data("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "testing_images.npy", 2051, max)
    labels = get_label_data("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "testing_labels.npy", 2049)
    assert images.shape[0] == labels.shape[0]

    return images, labels

def accuracy(predictions, y_list):
    return np.sum(np.array(predictions) == np.array(y_list)) / float(len(y_list)) * 100

# ********************************************************************************
#                      Model Definitions
# ********************************************************************************
# Neurons for input layer
input_model = genn_model.create_custom_neuron_class(
    "input",
    extra_global_params=[("ImageData", "float*"), ("ImageOffset", "unsigned int*")],
    threshold_condition_code="$(gennrand_uniform) < $(ImageData)[$(ImageOffset)[0] + $(id)]",
    is_auto_refractory_required=False
)

# Neurons for hidden layers
hidden_layer_model = genn_model.create_custom_neuron_class(
    "hidden_layer",
    param_names=["Tau"],
    var_name_types=[("V", "scalar"), ("U", "scalar"), ("PhiF", "scalar"), ("PsiH", "scalar")],
    additional_input_vars=[("IsynBack", "scalar", 0.0)],
    derived_params=[
        ("OneMinusTau", genn_model.create_dpf_class(lambda pars, dt: 1.0 - pars[0])())
    ], 
    sim_code="""
        // Update voltage
        $(V) = ($(OneMinusTau) * $(V)) + ($(Tau) * $(Isyn));
        
        // Calculate sigmoid of voltage and hence positive weighted function
        const scalar sigmoidV = 1.0 / (1.0 + exp(-$(V)));
        $(PsiH) = sigmoidV * (1.0 - sigmoidV);
        
        // Update U
        $(U) = ($(OneMinusTau) * $(U)) + ($(Tau) * $(IsynBack));
        
        // Calculate sigmoid of U
        $(PhiF) = 1.0 / (1.0 + exp(-$(U)));
        """,
    threshold_condition_code="$(gennrand_uniform) < sigmoidV",
    is_auto_refractory_required=False
)

# Neurons for output layer
output_layer_model = genn_model.create_custom_neuron_class(
    "output_layer",
    extra_global_params=[("LabelData", "unsigned int*"), ("LabelOffset", "unsigned int*")],
    param_names=["Tau", "NeuronsPerLabel"],
    var_name_types=[("V", "scalar"), ("U", "scalar"), ("PhiF", "scalar"), ("PsiH", "scalar")],
    derived_params=[
        ("OneMinusTau", genn_model.create_dpf_class(lambda pars, dt: 1.0 - pars[0])())
    ], 
    sim_code="""
        // Determine what label this neuron SHOULD be representing
        const unsigned int neuronLabel = $(id) / (unsigned int)$(NeuronsPerLabel);
        
        // Determine correct label
        const unsigned int correctLabel = $(LabelData)[$(LabelOffset)[0]];
        
        // Thus calculate Phi*
        const scalar phiStar = (neuronLabel == correctLabel) ? 1.0 : 0.0;
        
        // Update voltage
        $(V) = ($(OneMinusTau) * $(V)) + ($(Tau) * $(Isyn));
        
        // Calculate sigmoid of voltage and hence positive weighted function
        const scalar sigmoidV = 1.0 / (1.0 + exp(-$(V)));
        $(PsiH) = sigmoidV * (1.0 - sigmoidV);
        
        // Determine whether neuron spiked
        const bool spike = ($(gennrand_uniform) < (1.0 / (1.0 + exp(-$(V)))));
        
        // Thus calculate U 
        const scalar phiH = spike ? 1.0 : 0.0;
        $(U) = ($(OneMinusTau) * $(U)) + ($(Tau) * (phiH - phiStar));
        
        // Calculate sigmoid of U
        $(PhiF) = 1.0 / (1.0 + exp(-$(U)));
        """,
    threshold_condition_code="spike",
    is_auto_refractory_required=False
)

# Weight update model for continuous, backwards updates
backward_continuous = genn_model.create_custom_weight_update_class(
    "backward_continuous",
    var_name_types=[("g", "scalar")],
    synapse_dynamics_code="$(addToInSyn, $(g) * $(PhiF_pre));",
)

# Postsynaptic update model to deliver input to IsynBack
backwards_delta = genn_model.create_custom_postsynaptic_class(
    "backwards_delta",
    apply_input_code="$(IsynBack) += $(inSyn); $(inSyn) = 0;"
)

# Learning rule running 
forward_learning = genn_model.create_custom_weight_update_class(
    "forward_learning",
    param_names=["nu"],
    var_name_types=[("g", "scalar")],
 
    sim_code=
        """
        $(addToInSyn, $(g));
        
        // Update G
        $(g) += $(nu) * $(PhiF_post) * $(PsiH_post);
        """
    )

# ********************************************************************************
#                      Data
# ********************************************************************************
start = time.time()
training_images, training_labels = get_training_data(0.95)
end = time.time()
print("time needed to load training set:%f" % (end - start))
 
start = time.time()
testing_images, testing_labels = get_testing_data(0.95)
end = time.time()
print("time needed to load test set:%f" % (end - start))

# ********************************************************************************
#                      Parameters and Hyperparameters
# ********************************************************************************
# Global 
dt = 1.0

# Architecture
num_examples = training_images.shape[0]
layer_sizes = [784, 1500, 1500, 1500, 1000]
single_example_time = 5
train_timesteps = num_examples * single_example_time
max_input = 0.95

# Neuron group parameters
hidden_params = {"Tau": 0.95}
output_params = {"Tau": 0.95, "NeuronsPerLabel": 100}

# Neuron group initial values
output_hidden_init = {"V": 0.0, "U": 0.0, "PhiF": 0.0, "PsiH": 0.0}

# Weight update
forward_params = {"nu": 1E-03}
forward_init = {"g": genn_model.init_var("Uniform", {"min": -0.05, "max": 0.05})}
backward_init = {"g": genn_model.init_var("Uniform", {"min": -0.1, "max": 0.1})}

# ********************************************************************************
#                      Model Instances
# ********************************************************************************
model = genn_model.GeNNModel("float","mnist")
model.dT = dt
model.timing_enabled = True

# Add input layer (layer 0)
layers = [model.add_neuron_population("input_layer", layer_sizes[0], input_model, 
                                      {}, {})]

# Add hidden layers
for i, size in enumerate(layer_sizes[1:-1]):
    layers.append(model.add_neuron_population("hidden_layer%u" % i, size, hidden_layer_model, 
                                              hidden_params, output_hidden_init))
# Add output layer
layers.append(model.add_neuron_population("output_layer", layer_sizes[-1], output_layer_model, 
                                          output_params, output_hidden_init))

# Loop through pairs or layers
for pre, post in zip(layers[:-1], layers[1:]):
    # Add forward synapse population
    model.add_synapse_population("forward_%s_%s" % (pre.name, post.name), "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                 pre, post,
                                 forward_learning, forward_params, forward_init, {}, {},
                                 "DeltaCurr", {}, {})
                                 
    # Add backwards synapse population
    if pre.name != "input_layer":
        model.add_synapse_population("backward_%s_%s" % (post.name, pre.name), "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                     post, pre,
                                     backward_continuous, {}, backward_init, {}, {},
                                     backwards_delta, {}, {})

# Upload all image data and labels to EGPs
layers[0].add_extra_global_param("ImageData", training_images.flatten())
layers[-1].add_extra_global_param("LabelData", training_labels)

# Initially set offsets into images and labels to zero
# **YUCK** should be able to use scalar types here but this whole API needs more thought
layers[0].add_extra_global_param("ImageOffset", [0])
layers[-1].add_extra_global_param("LabelOffset", [0])

# ********************************************************************************
#                      Building and Simulation
# ********************************************************************************

print("Building Model")
model.build()
print("Loading Model")
model.load()

# Get views to EGPs used fr
input_image_offset_view = layers[0].extra_global_params["ImageOffset"].view
output_label_offset_view = layers[-1].extra_global_params["LabelOffset"].view

print("Simulating")

# Create arrays to hold spikes for each layer
layer_spikes = [[np.asarray([]), np.asarray([])] for _ in layers]

start = time.time()
while model.timestep < train_timesteps:
    # Calculate the timestep within the presentation
    timestep_in_example = model.timestep % single_example_time

    # If this is the first timestep of the presentation
    if timestep_in_example == 0:
        # Calculate index of example
        example = int(model.timestep // single_example_time)
        print("Example %u" % example)

        # Set offset and upload
        # **YUCK** upload wouldn't be required if we could use scalar
        input_image_offset_view[:] = (example % 60000) * (28 * 28)
        model._slm.push_extra_global_param(layers[0].name, "ImageOffset", 1)
        
        # Set Phi* to 1 for 100 neurons representing selected labels and 0 otherwise
        output_label_offset_view[:] = training_labels[example % 60000]
        model._slm.push_extra_global_param(layers[-1].name, "LabelOffset", 1)
        
    # Advance simulation
    model.step_time()
    
    # Loop through layers and their spike arrays
    for l, s in zip(layers, layer_spikes):
        # Download spikes from GPU
        model.pull_current_spikes_from_device(l.name)
        
        # Build array of spike times
        spike_times = np.ones(l.current_spikes.shape) * model.t
        
        # Add to data structure
        s[0] = np.hstack((s[0], l.current_spikes))
        s[1] = np.hstack((s[1], spike_times))

end = time.time()
print("time to simulate:%f" % (end - start))
print("neuron_update_time:%f, init_time:%f, presynaptic_update_time:%f, postsynaptic_update_time:%f, synapse_dynamics_time:%f, init_sparse_time:%f"
      % (model.neuron_update_time, model.init_time, model.presynaptic_update_time, 
         model.postsynaptic_update_time, model.synapse_dynamics_time, model.init_sparse_time))

fig, axes = plt.subplots(len(layers), sharex=True)

for a, l, s in zip(axes, layers, layer_spikes):
    a.set_title(l.name)
    a.scatter(s[1], s[0], s=2)
plt.show()