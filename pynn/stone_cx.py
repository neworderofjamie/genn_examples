import cv2
import numpy as np

from pygenn import GeNNModel

from pygenn import (create_neuron_model, create_var_init_snippet, create_var_ref,
                    create_weight_update_model,
                    create_sparse_connect_init_snippet,
                    init_postsynaptic, init_sparse_connectivity,
                    init_var, init_weight_update)

from scipy.stats import vonmises
from scipy.interpolate import CubicSpline

# Population sizes
NUM_TN2 = 2
NUM_TL = 16
NUM_CL1 = 16
NUM_TB1 = 8
NUM_CPU4 = 16
NUM_PONTINE = 16
NUM_CPU1 = 16
C = 0.33

# Simulation rendering parameters
PATH_IMAGE_SIZE = 1000
ACTIVITY_IMAGE_WIDTH = 500
ACTIVITY_IMAGE_HEIGHT = 1000

PREFERRED_ANGLE_TN2 = np.asarray([np.pi / 4.0, -np.pi / 4.0])

# Outbound path generation parameters
NUM_OUTWARD_TIMESTEPS = 1500

# Agent dynamics parameters
PATH_LAMBDA = 0.4
PATH_KAPPA = 100.0

AGENT_DRAG = 0.15

AGENT_MIN_ACCELERATION = 0.0
AGENT_MAX_ACCELERATION = 0.15
AGENT_M = 0.5

def draw_neuron_activity(activity, position, get_colour_fn, image):
    # Convert activity to a 8-bit level
    gray = int(255.0 * min(1.0, max(0.0, activity)))
    
    #// Draw rectangle of this colour
    image[position[1]:position[1] + 25, position[0]:position[0] + 25,:] = get_colour_fn(gray)

def draw_population_activity(pop_view, pop_name, position, get_colour_fn, image, num_columns=0):
    # If (invalid) default number of columns is specified, use popsize
    if num_columns == 0:
        num_columns = len(pop_view)

    for i, a in enumerate(pop_view):
        row = i // num_columns
        col = i % num_columns
        
        draw_neuron_activity(a, (position[0] + (col * 27), position[1] + (row * 27)),
                             get_colour_fn, image)

    # Label population
    num_rows = int(np.ceil(len(pop_view) / num_columns))
    cv2.putText(image, pop_name, (position[0], position[1] + 17 + (27 * num_rows)),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0xFF, 0xFF, 0xFF))

sigmoid_neuron_model = create_neuron_model(
    "sigmoid",

    params=["a", "b"],
    vars=[("r", "scalar")],

    sim_code=
    """
    r = 1.0 / (1.0 + exp(-((a * Isyn) - b)));
    """)
    
# Continuous
continuous_weight_update_model = create_weight_update_model(
    "continous",
    vars=[("g", "scalar")],
    pre_neuron_var_refs=[("r", "scalar")],
    synapse_dynamics_code=
    """
    addToPost(g * r);
    """)


# TN2Linear
# **NOTE** this comes from https://github.com/InsectRobotics/path-integration/blob/master/cx_rate.py#L170-L173 rather than the methods section
tn2_linear_neuron_model = create_neuron_model(
    "tn2_linear",
    vars=[("r", "scalar"), ("speed", "scalar")],
    sim_code=
    """
    r = fmin(1.0, fmax(speed, 0.0));
    """)


# TLSigmoid
tl_sigmoid_neuron_model = create_neuron_model(
    "tl_sigmoid",
    
    params=["a", "b", "headingAngle"],
    vars=[("r", "scalar"), ("preferredAngle", "scalar")],
    sim_code=
    """
    const scalar iTL = cos(preferredAngle - headingAngle);
    r = 1.0 / (1.0 + exp(-((a * iTL) - b)));
    """)

# CPU4Sigmoid
cpu4_sigmoid_neuron_model = create_neuron_model(
    "cpu4_sigmoid",
    
    params=["a", "b", "h", "k"],
    vars=[("r", "scalar"), ("i", "scalar")],
    sim_code=
    """
    i += h * fmin(1.0, fmax(Isyn, 0.0));
    i -= h * k;
    i = fmin(1.0, fmax(i, 0.0));
    r = 1.0 / (1.0 + exp(-((a * i) - b)));
    """)

# PreferredAngle
preferred_angle_var_init = create_var_init_snippet(
    "preferred_angle",
    
    var_init_code=
    """
    value = 0.785398163 * (scalar)(id % 8);
    """)


# TBToTB
tb_to_tb_var_init = create_var_init_snippet(
    "tb_to_tb",
    
    params=["c"],
    var_init_code=
    """
    const scalar preferredI = 0.785398163 * (scalar)(id_pre % 8);
    const scalar preferredJ = 0.785398163 * (scalar)(id_post % 8);
    const scalar w = (cos(preferredI - preferredJ) - 1.0) / 2.0;
    value = w * c;
    """)


# TBToCPU
tb_to_cpu_sparse_init = create_sparse_connect_init_snippet(
    "tb_to_cpu",

    row_build_code=
    """
    addSynapse(id_pre);
    addSynapse(id_pre + 8);
    """,
    calc_max_row_len_func=lambda num_pre, num_post, pars: 2,
    calc_max_col_len_func=lambda num_pre, num_post, pars: 1)


# CL1ToTB1
cl1_to_tb1_sparse_init = create_sparse_connect_init_snippet(
    "cl1_to_tb1",
    
    row_build_code=
    """
    addSynapse(id_pre % 8);
    """,

    calc_max_row_len_func=lambda num_pre, num_post, pars: 1,
    calc_max_col_len_func=lambda num_pre, num_post, pars: 1)

# PontineToCPU1
pontine_to_cpu1_sparse_init = create_sparse_connect_init_snippet(
    "pontine_to_cpu1",
    row_build_code=
    """
    if(id_pre < 5) {
       addSynapse(id_pre + 11);
    }
    else if(id_pre < 8) {
       addSynapse(id_pre + 3);
    }
    else if(id_pre < 11) {
       addSynapse(id_pre - 3);
    }
    else {
       addSynapse(id_pre - 11);
    }
    """,
    calc_max_row_len_func=lambda num_pre, num_post, pars: 1,
    calc_max_col_len_func=lambda num_pre, num_post, pars: 1)


# CPU4ToCPU1
cpu4_to_cpu1_sparse_init = create_sparse_connect_init_snippet(
    "cpu4_to_cpu1",
    
    row_build_code=
    """
    if(id_pre == 0) {
       addSynapse(15);
    }
    else if(id_pre < 8) {
       addSynapse(id_pre + 7);
    }
    else if(id_pre < 15) {
       addSynapse(id_pre - 7);
    }
    else {
       addSynapse(0);
    }
    """,
    calc_max_row_len_func=lambda num_pre, num_post, pars: 1,
    calc_max_col_len_func=lambda num_pre, num_post, pars: 1)


# TN2CPU4
tn2_to_cpu4_sparse_init = create_sparse_connect_init_snippet(
    "tn2_to_cpu4",
    
    row_build_code=
    """
    for(unsigned int c = 0; c < 8; c++) {
        addSynapse((id_pre * 8) + c);
    }
    """,
    calc_max_row_len_func=lambda num_pre, num_post, pars: 8,
    calc_max_col_len_func=lambda num_pre, num_post, pars: 1)

# Neuron parameters
sigmoid_init = {"r": 0.0}

# TN
tn2_init = {"r": 0.0, "speed": 0.0}

# TL
tl_params = {"a": 6.8, "b": 3.0, "headingAngle": 0.0}
tl_init = {"r": 0.0, "preferredAngle": init_var(preferred_angle_var_init)}

# CL1
cl1_params = {"a": 3.0, "b": -0.5}

# TB1
tb1_params = {"a": 5.0, "b": 0.0}

# CPU4
cpu4_params = {"a": 5.0, "b": 2.5, "h": 0.0025, "k": 0.125}
cpu4_init = {"r": 0.0, "i": 0.5}

# Pontine
pontine_params = {"a": 5.0, "b": 2.5}

# CPU1 **NOTE** these are the values from https://github.com/InsectRobotics/path-integration/blob/master/cx_rate.py#L231-L232
cpu1_params = {"a": 7.5, "b": -1.0}

# Synapse parameters
exc_init = {"g": 1.0}
inh_init = {"g": -1.0}

cl1_tb1_init = {"g": (1.0 - C)}
cpu4_cpu1_init = {"g": 0.5}
pontine_cpu1_init = {"g": -0.5}
tb1_tb1_init = {"g": init_var(tb_to_tb_var_init, {"c": C})} 

model = GeNNModel("float", "stone_cx", backend="single_threaded_cpu")
model.dt = 1.0

# Neuron populations
tn2 = model.add_neuron_population("TN2", NUM_TN2, tn2_linear_neuron_model, 
                                  {}, tn2_init)
tl = model.add_neuron_population("TL", NUM_TL, tl_sigmoid_neuron_model,
                                 tl_params, tl_init)
cl1 = model.add_neuron_population("CL1", NUM_CL1, sigmoid_neuron_model,
                                  cl1_params, sigmoid_init)
tb1 = model.add_neuron_population("TB1", NUM_TB1, sigmoid_neuron_model, 
                                  tb1_params, sigmoid_init)
cpu4 = model.add_neuron_population("CPU4", NUM_CPU4, cpu4_sigmoid_neuron_model, 
                                   cpu4_params, cpu4_init)
pontine = model.add_neuron_population("Pontine", NUM_PONTINE, sigmoid_neuron_model,
                                      pontine_params, sigmoid_init);
cpu1 = model.add_neuron_population("CPU1", NUM_CPU1, sigmoid_neuron_model,
                                   cpu1_params, sigmoid_init);

tl.set_param_dynamic("headingAngle")

# Synapse populations
model.add_synapse_population(
    "TL_CL1", "SPARSE",
    tl, cl1,
    init_weight_update(continuous_weight_update_model, {}, inh_init,
                       pre_var_refs={"r": create_var_ref(tl, "r")}),
    init_postsynaptic("DeltaCurr"),
    init_sparse_connectivity("OneToOne"))

model.add_synapse_population(
    "CL1_TB1", "SPARSE",
    cl1, tb1,
    init_weight_update(continuous_weight_update_model, {}, cl1_tb1_init,
                       pre_var_refs={"r": create_var_ref(cl1, "r")}),
    init_postsynaptic("DeltaCurr"),
    init_sparse_connectivity(cl1_to_tb1_sparse_init))

model.add_synapse_population(
    "TB1_TB1", "DENSE",
    tb1, tb1,
    init_weight_update(continuous_weight_update_model, {}, tb1_tb1_init,
                       pre_var_refs={"r": create_var_ref(tb1, "r")}),
    init_postsynaptic("DeltaCurr"))

model.add_synapse_population(
    "CPU4_Pontine", "SPARSE",
    cpu4, pontine,
    init_weight_update(continuous_weight_update_model, {}, exc_init,
                       pre_var_refs={"r": create_var_ref(cpu4, "r")}),
    init_postsynaptic("DeltaCurr"),
    init_sparse_connectivity("OneToOne"))

model.add_synapse_population(
    "TB1_CPU4", "SPARSE",
    tb1, cpu4,
    init_weight_update(continuous_weight_update_model, {}, inh_init,
                       pre_var_refs={"r": create_var_ref(tb1, "r")}),
    init_postsynaptic("DeltaCurr"),
    init_sparse_connectivity(tb_to_cpu_sparse_init))

model.add_synapse_population(
    "TB1_CPU1", "SPARSE",
    tb1, cpu1,
    init_weight_update(continuous_weight_update_model, {}, inh_init,
                       pre_var_refs={"r": create_var_ref(tb1, "r")}),
    init_postsynaptic("DeltaCurr"),
    init_sparse_connectivity(tb_to_cpu_sparse_init))

model.add_synapse_population(
    "CPU4_CPU1", "SPARSE",
    cpu4, cpu1,
    init_weight_update(continuous_weight_update_model, {}, cpu4_cpu1_init,
                       pre_var_refs={"r": create_var_ref(cpu4, "r")}),
    init_postsynaptic("DeltaCurr"),
    init_sparse_connectivity(cpu4_to_cpu1_sparse_init))

model.add_synapse_population(
    "TN2_CPU4", "SPARSE",
    tn2, cpu4,
    init_weight_update(continuous_weight_update_model, {}, exc_init,
                       pre_var_refs={"r": create_var_ref(tn2, "r")}),
    init_postsynaptic("DeltaCurr"),
    init_sparse_connectivity(tn2_to_cpu4_sparse_init))

model.add_synapse_population(
    "Pontine_CPU1", "SPARSE",
    pontine, cpu1,
    init_weight_update(continuous_weight_update_model, {}, pontine_cpu1_init,
                       pre_var_refs={"r": create_var_ref(pontine, "r")}),
    init_postsynaptic("DeltaCurr"),
    init_sparse_connectivity(pontine_to_cpu1_sparse_init))

model.build()
model.load()

cv2.namedWindow("Path", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Path", (PATH_IMAGE_SIZE, PATH_IMAGE_SIZE))
path_image = np.zeros((PATH_IMAGE_SIZE, PATH_IMAGE_SIZE, 3), dtype=np.uint8)

cv2.namedWindow("Activity", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Activity", (ACTIVITY_IMAGE_WIDTH, ACTIVITY_IMAGE_HEIGHT))
cv2.moveWindow("Activity", PATH_IMAGE_SIZE, 0)
activity_image = np.zeros((ACTIVITY_IMAGE_HEIGHT, ACTIVITY_IMAGE_WIDTH, 3), dtype=np.uint8)

path_von_mises = vonmises(loc=0.0, kappa=PATH_KAPPA)

# Create vectors to hold the times at which linear acceleration
# should change and it's values at those time
num_acceleration_changes = NUM_OUTWARD_TIMESTEPS // 50

# Generate regular times at which acceleration changes and random accelerations
acceleration_time = np.arange(0.0, 50 * num_acceleration_changes, 50.0)
acceleration_magnitude = np.random.uniform(AGENT_MIN_ACCELERATION, AGENT_MAX_ACCELERATION, size=num_acceleration_changes)

# Cubic interpolate
acceleration_spline = CubicSpline(acceleration_time, acceleration_magnitude)

# Simulate
omega = 0.0
theta = 0.0
x_velocity = 0.0
y_velocity = 0.0
x_position = 0.0
y_position = 0.0
while True:
    tn2.vars["speed"].view[:] = (np.sin(theta + PREFERRED_ANGLE_TN2) * x_velocity) + (np.cos(theta + PREFERRED_ANGLE_TN2) * y_velocity)
    
    # Push inputs to device
    tn2.vars["speed"].push_to_device()

    # Update TL input
    tl.set_dynamic_param_value("headingAngle", theta)

    # Step network
    model.step_time()

    # Pull outputs from device
    tl.vars["r"].pull_from_device()
    tn2.vars["r"].pull_from_device()
    cl1.vars["r"].pull_from_device()
    tb1.vars["r"].pull_from_device()
    cpu4.vars["r"].pull_from_device()
    cpu4.vars["i"].pull_from_device()
    cpu1.vars["r"].pull_from_device()
    pontine.vars["r"].pull_from_device()

    # Visualize network 
    get_red = lambda gray: (gray, 0, 0)
    get_green = lambda gray: (0, gray, 0)
    get_blue = lambda gray: (0, 0, gray)
    
    draw_population_activity(tl.vars["r"].view, "TL", (10, 10),
                             get_red, activity_image, 8);
    draw_population_activity(cl1.vars["r"].view, "CL1", (10, 110),
                             get_red, activity_image, 8);
    draw_population_activity(tb1.vars["r"].view, "TB1", (10, 210),
                             get_red, activity_image);

    draw_population_activity(tn2.vars["r"].view, "TN2", (300, 310),
                             get_blue, activity_image, 1);

    draw_population_activity(cpu4.vars["r"].view, "CPU4", (10, 310),
                             get_green, activity_image, 8);
    draw_population_activity(pontine.vars["r"].view, "Pontine", (10, 410),
                             get_green, activity_image, 8);
    draw_population_activity(cpu1.vars["r"].view, "CPU1", (10, 510),
                             get_green, activity_image, 8);

    # If we are on outbound segment of route
    outbound = (model.timestep < NUM_OUTWARD_TIMESTEPS)
    a = 0.0
    if outbound:
        # Update angular velocity
        omega = (PATH_LAMBDA * omega) + path_von_mises.rvs()

        # Read linear acceleration off spline
        a = acceleration_spline(model.t)

        if model.timestep == (NUM_OUTWARD_TIMESTEPS - 1):
            print(f"Max CPU4 level r={np.amax(cpu4.vars['r'].view)}, i={np.amax(cpu4.vars['i'].view)}")
    # Otherwise we're path integrating home
    else:
        # Sum left and right motor activity
        left_motor = np.sum(cpu1.vars["r"].view[:8])
        right_motor = np.sum(cpu1.vars["r"].view[8:])

        # Use difference between left and right to calculate angular velocity
        omega = -AGENT_M * (right_motor - left_motor)

        # Use fixed acceleration
        a = 0.1

    # Update heading
    theta += omega

    # Update linear velocity
    # **NOTE** this comes from https://github.com/InsectRobotics/path-integration/blob/master/bee_simulator.py#L77-L83 rather than the methods section
    x_velocity += np.sin(theta) * a
    y_velocity += np.cos(theta) * a
    x_velocity -= AGENT_DRAG * x_velocity
    y_velocity -= AGENT_DRAG * y_velocity

    # Update position
    x_position += x_velocity
    y_position += y_velocity

    # Draw agent position (centring so origin is in centre of path image)
    path_image[(PATH_IMAGE_SIZE // 2) + int(y_position), (PATH_IMAGE_SIZE // 2) + int(x_position)] = (0xFF, 0, 0) if outbound else (0, 0xFF, 0)

    # Show output image
    cv2.imshow("Path", path_image)
    cv2.imshow("Activity", activity_image)
    if cv2.waitKey(1) == 27:
        break
