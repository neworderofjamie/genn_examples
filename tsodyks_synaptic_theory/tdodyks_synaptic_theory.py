import math
import matplotlib.pyplot as plt
import numpy as np

from pygenn import GeNNModel

from pygenn import (create_current_source_model, create_var_init_snippet,
                    create_weight_update_model, init_postsynaptic,
                    init_sparse_connectivity, init_var, init_weight_update)

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
    value = gennrand() % 10;
    """)

stim_model = create_current_source_model(
    "stim",
    
    params=["startTime", "endTime", "mean", "sd"],
    vars=[("current", "scalar")],
    injection_code=
    """
    if(t > startTime && t < endTime) {
        if(fmod(t, 1.0) < 0.001) {
            current = mean + (gennrand_normal() * sd);
        }
        injectCurrent(current);
    }
    """)

stp_model = create_weight_update_model(
    "stp",
    
    params=["TauD", "TauF", "U"],
    pre_vars=[("xTMinus", "scalar"), ("uTPlus", "scalar")],
    vars=[("g", "scalar"), ("d", "uint8_t")],
    
    pre_spike_code=
    """
    const timepoint deltaT = t - st_pre;
    
    // Calculate xTPlus from last spike`
    const scalar xTPlus = xTMinus - (uTPlus * xTMinus);
    
    // Update xTMinus for current spike
    xTMinus = 1.0 + ((xTPlus - 1) * exp(-deltaT / TauD));
    
    // Calculate uTMinus for current spike
    const scalar uTMinus = U + ((uTPlus - U) * exp(-deltaT / TauF));
    
    // Calculate uTPlus frm this
    uTPlus = uTMinus + (U * (1.0 - uTMinus));
    """,
    pre_spike_syn_code=
    """
    addToPostDelay(g  * uTPlus * xTMinus, d);
    """)

def add_stp_synapse_pop(model, pre_pop, post_pop, g, num):
    pop = model.add_synapse_population(f"{pre_pop.name}_{post_pop.name}", "SPARSE",
                                       pre_pop, post_pop,
                                       init_weight_update(stp_model, {"TauD": TAU_D, "TauF": TAU_F, "U": U},
                                                          {"g": g, "d": init_var(delay_init)},
                                                          {"xTMinus": 1.0, "uTPlus": U}),
                                       init_postsynaptic("ExpCurr", {"tau": TAU_SYN}),
                                       init_sparse_connectivity("FixedNumberPreWithReplacement", {"num": num}))
    pop.max_dendritic_delay_timesteps = 10
    
def add_synapse_pop(model, pre_pop, post_pop, g, num):
    pop = model.add_synapse_population(f"{pre_pop.name}_{post_pop.name}", "SPARSE",
                                       pre_pop, post_pop,
                                       init_weight_update("StaticPulseDendriticDelay", {},
                                                          {"g": g, "d": init_var(delay_init)}),
                                       init_postsynaptic("ExpCurr", {"tau": TAU_SYN}),
                                       init_sparse_connectivity("FixedNumberPreWithReplacement", {"num": num}))
    pop.max_dendritic_delay_timesteps = 10

def get_weight(PSP_val, tau_m, C_m = 250.0, tau_syn_ex = 2.0):
    """ Computes weight to elicit a change in the membrane potential.
    Reference:
    [1] Potjans TC. and Diesmann M. 2014. The cell-type specific 
    cortical microcircuit: relating structure and activity in a 
    full-scale spiking network model. Cerebral Cortex. 
    24(3):785-806. DOI: 10.1093/cercor/bhs358.

    Parameters
    ----------
    PSP_val
        Evoked postsynaptic potential.
    net_dict
        Dictionary containing parameters of the microcircuit.

    Output
    -------
    PSC_e
        Weight value(s).
    """

    PSC_e_over_PSP_e = (((C_m) ** (-1) * tau_m * tau_syn_ex / (
        tau_syn_ex - tau_m) * ((tau_m / tau_syn_ex) ** (
            - tau_m / (tau_m - tau_syn_ex)) - (tau_m / tau_syn_ex) ** (
                - tau_syn_ex / (tau_m - tau_syn_ex)))) ** (-1))
    PSC_e = (PSC_e_over_PSP_e * PSP_val)
    return PSC_e

def get_noise_mean(mu_ext, tau_m, C_m=250.0):
    return (C_m / tau_m) * mu_ext

def get_noise_std(sigma_ext, tau_m, dt=0.1, C_m=250.0):
    return math.sqrt(2/(tau_m*dt))*C_m*sigma_ext

# Single cell parameters
# ======================
V_THRESH = 20.0
V_RESET_E = 16.0
V_RESET_I = 13.0
TAU_M_E = 15.0
TAU_M_I = 10.0

C_M = 250.0 # **NOTE** NOT GeNN units
TAU_REFRAC = 2.0
TAU_SYN = 2.0

# Network parameters
# ==================
F = 0.1
P = 5
C = 0.20
N_E = 8000
N_I = 2000
MU_EXT_E = get_noise_mean(23.7, TAU_M_E, C_M) / 1000.0
MU_EXT_I = get_noise_mean(20.5, TAU_M_I, C_M) / 1000.0
MU_LOAD = get_noise_mean(23.7 * (1.15 - 1.0), TAU_M_E, C_M) / 1000.0
SIGMA_EXT_E = get_noise_std(1.0, TAU_M_E, 1.0, C_M) / 1000.0
SIGMA_EXT_I = get_noise_std(1.0, TAU_M_I, 1.0, C_M) / 1000.0

N_E_SELECT = int(F * N_E)
N_E_NON_SELECT = int((1.0 - (F * P)) * N_E)

# Synaptic parameters
# ===================
J_IE = get_weight(-0.25, TAU_M_I, C_M, TAU_SYN) / 1000.0 # **NOTE** J_EI and J_IE swapped for sanity
J_EI = get_weight(0.135, TAU_M_E, C_M, TAU_SYN) / 1000.0
J_II = get_weight(-0.2, TAU_M_I, C_M, TAU_SYN) / 1000.0
J_B = get_weight(0.1, TAU_M_E, C_M, TAU_SYN) / 1000.0
J_P = get_weight(0.45, TAU_M_E, C_M, TAU_SYN) / 1000.0

U = 0.19
TAU_F = 1500.0
TAU_D = 200.0

PARAMS_E = {"C": C_M  / 1000.0, "TauM": TAU_M_E, "Vrest": 0.0, "Vreset": V_RESET_E,
            "Vthresh": V_THRESH, "Ioffset": 0.0, "TauRefrac": TAU_REFRAC}
PARAMS_I = {"C": C_M  / 1000.0, "TauM": TAU_M_I, "Vrest": 0.0, "Vreset": V_RESET_I,
            "Vthresh": V_THRESH, "Ioffset": 0.0, "TauRefrac": TAU_REFRAC}
VARS = {"V": 0.0, "RefracTime": 0.0}

model = GeNNModel("float", "tsodyks_synaptic_theory")
model.dt = 0.1
model.seed = 1234
model.fuse_postsynaptic_models = True
model.fuse_pre_post_weight_update_models = True

# Neuron populations and current sources
# ======================================
i_pop = model.add_neuron_population("I", N_I, "LIF", PARAMS_I, VARS)
i_pop.spike_recording_enabled = True
model.add_current_source("ICurr", stim_model, i_pop,
                         {"mean": MU_EXT_I, "sd": SIGMA_EXT_I, "startTime": 0.0, "endTime": 6000.0},
                         {"current": 0.0})

e_non_select_pop = model.add_neuron_population("ENonSelect", N_E_NON_SELECT, "LIF",
                                               PARAMS_E, VARS)
e_non_select_pop.spike_recording_enabled = True

model.add_current_source("ENonSelectCurr", stim_model, e_non_select_pop,
                         {"mean": MU_EXT_E, "sd": SIGMA_EXT_E, "startTime": 0.0, "endTime": 6000.0},
                         {"current": 0.0})

e_select_pop = []
for i in range(P):
    e_select_pop.append(model.add_neuron_population(f"ESelect{i}", N_E_SELECT, "LIF", 
                                                    PARAMS_E, VARS))
    e_select_pop[-1].spike_recording_enabled = True
    model.add_current_source(f"ENonSelectCurr{i}", stim_model, e_select_pop[-1],
                             {"mean": MU_EXT_E, "sd": SIGMA_EXT_E, "startTime": 0.0, "endTime": 6000.0},
                             {"current": 0.0})

# Load
#model.add_current_source(f"LoadCurr", stim_model, e_select_pop[2],
#                         {"mean": MU_LOAD, "sd": SIGMA_EXT_E,
#                          "startTime": 3000.0, "endTime": 3350.0},
#                         {"current": 0.0})
                             
# Inhibitory synapse populations
# ==============================
add_synapse_pop(model, i_pop, i_pop, J_II, int(C * N_I))
add_synapse_pop(model, i_pop, e_non_select_pop, J_IE, int(C * N_I))
for e in e_select_pop:
    add_synapse_pop(model, i_pop, e, J_IE, int(C * N_I))

# Excitatory non-selective synapse populations
# ============================================
# Synapses connecting two neurons within the same selective population have 
# potentiated efficacy; Synapses connecting a selective neuron to a neuron
# from another selective population or to a non-selective neuron, have baseline
# efficacy; The remaining synapses (i.e. non-selective to selective and 
# non-selective to non-selective) have potentiated efficacy with probability 0.1
num_non_select_num_pre = int(C * (1.0 - (F * P)) * N_E)
add_synapse_pop(model, e_non_select_pop, i_pop, 
                J_EI, num_non_select_num_pre)

non_select_g_init = init_var(non_selective_init, {"Jb": J_B, "Jp": J_P,
                                                  "p": 0.1})
add_stp_synapse_pop(model, e_non_select_pop, e_non_select_pop,
                    non_select_g_init, num_non_select_num_pre)
for e in e_select_pop:
    add_stp_synapse_pop(model, e_non_select_pop, e, non_select_g_init,
                        num_non_select_num_pre)

# Excitatory selective synapse populations
# ========================================
num_select_num_pre = int(C * F * N_E)
for e_pre in e_select_pop:
    add_synapse_pop(model, e_pre, i_pop, J_EI, num_select_num_pre)
    add_stp_synapse_pop(model, e_pre, e_non_select_pop, J_B, num_select_num_pre)
    
    for e_post in e_select_pop:
        j = J_P if e_pre == e_post else J_B
        add_stp_synapse_pop(model, e_pre, e_post, j, num_select_num_pre)

model.build()
model.load(num_recording_timesteps=60000)

while model.t < 6000.0:
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

