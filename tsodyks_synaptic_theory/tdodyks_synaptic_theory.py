import matplotlib.pyplot as plt

from pygenn import GeNNModel

from pygenn import (create_neuron_model, create_var_init_snippet,
                    init_postsynaptic, init_sparse_connectivity,
                    init_var, init_weight_update)

non_selective_init = create_var_init_snippet(
    "non_selective",
    params=["Jb", "Jp", "p"],
    
    var_init_code=
    """
    value = (gennrand_uniform() < p) ? Jp : Jb;
    """)

delay_init = create_var_init_snippet(
    "delay",
    
    var_init_code=
    """
    value = gennrand() % 50;
    """)

lif = create_neuron_model(
    "lif",
    vars=[("V", "scalar"), ("RefracTime", "scalar")],
    params=["TauM", "Vr", "Vthresh", "TauRefrac"],

    sim_code=
    """
    if (RefracTime <= 0.0) {
      V += (1.0 / TauM) * (-V + Isyn) * dt;
    }
    else {
      RefracTime -= dt;
    }
    """,
    
    threshold_condition_code=
    """
    RefracTime <= 0.0 && V >= Vr
    """,
    
    reset_code=
    """
    V = Vr;
    RefracTime = TauRefrac;
    """)

def add_synapse_pop(model, pre_pop, post_pop, g, num):
    pop = model.add_synapse_population(f"{pre_pop.name}_{post_pop.name}", "SPARSE",
                                       pre_pop, post_pop,
                                       init_weight_update("StaticPulseDendriticDelay", {},
                                                          {"g": g, "d": init_var(delay_init)}),
                                       init_postsynaptic("DeltaCurr"),
                                       init_sparse_connectivity("FixedNumberPreWithReplacement", {"num": num}))
    pop.max_dendritic_delay_timesteps = 50


# Single cell parameters
V_THRESH = 20.0
V_RESET_E = -16.0
V_RESET_I = -13.0
TAU_M_E = 15.0
TAU_M_I = 10.0
TAU_REFRAC = 2.0

# Network parameters
F = 0.1
P = 5
C = 0.20
N_E = 8000
N_I = 2000
MU_EXT_E = 23.1
MU_EXT_I = 21.0 
SIGMA_EXT = 1.0

N_E_SELECT = round(F * N_E)
N_E_NON_SELECT = round((1.0 - (F * P)) * N_E)

# Synaptic parameters
J_IE = -0.135
J_EI = 0.25
J_II = -0.2
J_B = 0.1
J_P = 0.45
GAMMA_0 = 0.10

PARAMS_E = {"TauM": TAU_M_E, "Vr": V_RESET_E,
            "Vthresh": V_THRESH, "TauRefrac": TAU_REFRAC}
PARAMS_I = {"TauM": TAU_M_I, "Vr": V_RESET_I,
            "Vthresh": V_THRESH, "TauRefrac": TAU_REFRAC}
#PARAMS_E = {"C": TAU_M_E, "TauM": TAU_M_E, "Vrest": V_RESET_E, "Vreset": V_RESET_E,
#            "Vthresh": V_THRESH, "Ioffset": 0.0, "TauRefrac": TAU_REFRAC}
#PARAMS_I = {"C": TAU_M_I, "TauM": TAU_M_I, "Vrest": V_RESET_I, "Vreset": V_RESET_I,
#            "Vthresh": V_THRESH, "Ioffset": 0.0, "TauRefrac": TAU_REFRAC}
VARS = {"V": 0.0, "RefracTime": 0.0}

model = GeNNModel("float", "tsodyks_synaptic_theory")
model.dt = 0.1  # Not clear but this is minimum delay

# Add neuron populations and current sources
i_pop = model.add_neuron_population("I", N_I, lif, PARAMS_I, VARS)
i_pop.spike_recording_enabled = True
model.add_current_source("ICurr", "GaussianNoise", i_pop,
                         {"mean": MU_EXT_I, "sd": SIGMA_EXT})

e_non_select_pop = model.add_neuron_population("ENonSelect", N_E_NON_SELECT, lif, PARAMS_E, VARS)
e_non_select_pop.spike_recording_enabled = True

model.add_current_source("ENonSelectCurr", "GaussianNoise", e_non_select_pop,
                         {"mean": MU_EXT_E, "sd": SIGMA_EXT})

e_select_pop = []
for i in range(P):
    e_select_pop.append(model.add_neuron_population(f"ESelect{i}", N_E_SELECT, lif, 
                                                    PARAMS_E, VARS))
    e_select_pop[-1].spike_recording_enabled = True
    model.add_current_source(f"ENonSelectCurr{i}", "GaussianNoise", e_select_pop[-1],
                             {"mean": MU_EXT_E, "sd": SIGMA_EXT})

# Add inhibitory synapse populations
add_synapse_pop(model, i_pop, i_pop, J_II, int(C * N_I))
add_synapse_pop(model, i_pop, e_non_select_pop, J_IE, int(C * N_I))
for e in e_select_pop:
    add_synapse_pop(model, i_pop, e, J_IE, int(C * N_I))

# Add excitatory non-selective synapse populations
"""
Synapses connecting two neurons within the same selective population have 
potentiated efficacy; Synapses connecting a selective neuron to a neuron
from another selective population or to a non-selective neuron, have baseline
efficacy; The remaining synapses (i.e. non-selective to selective and 
non-selective to non-selective) have potentiated efficacy with probability 0.1
"""
num_non_select_num_pre = int(C * (1.0 - (F * P)) * N_E)
add_synapse_pop(model, e_non_select_pop, i_pop, J_EI, num_non_select_num_pre)

non_select_g_init = init_var(non_selective_init, {"Jb": J_B, "Jp": J_P,
                                                  "p": 0.1})
add_synapse_pop(model, e_non_select_pop, e_non_select_pop,
                non_select_g_init, num_non_select_num_pre)
for e in e_select_pop:
    add_synapse_pop(model, e_non_select_pop, e, non_select_g_init,
                    num_non_select_num_pre)

num_select_num_pre = int(C * F * N_E)
for e_pre in e_select_pop:
    add_synapse_pop(model, e_pre, i_pop, J_EI, num_non_select_num_pre)
    add_synapse_pop(model, e_pre, e_non_select_pop, J_B, num_non_select_num_pre)
    
    for e_post in e_select_pop:
        j = J_P if e_pre == e_post else J_B
        add_synapse_pop(model, e_pre, e_post, j, num_non_select_num_pre)

model.build()
model.load(num_recording_timesteps=500)

while model.timestep < 500:
    model.step_time()

model.pull_recording_buffers_from_device()

fig, axis = plt.subplots()

# Plot inhibitory
i_times, i_ids = i_pop.spike_recording_data[0]
axis.scatter(i_times, i_ids, s=1, label="I")

e_non_select_times, e_non_select_ids = e_non_select_pop.spike_recording_data[0]
axis.scatter(e_non_select_times, e_non_select_ids + N_I, s=1, label="ENonSelect")

start = N_I + N_E_NON_SELECT
for i, e in enumerate(e_select_pop):
    e_select_times, e_select_ids = e.spike_recording_data[0]
    axis.scatter(e_select_times, e_select_ids + start, s=1, label=f"ESelect{i}")
    start += N_E_SELECT

axis.legend()
plt.show()
# Add synapse populations
# I->all neurons
# 

# Mapping
# N_E_SELECT neurons per core
# 1 weight (outgoing or incoming? per core) - 

