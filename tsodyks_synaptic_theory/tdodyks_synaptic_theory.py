import matplotlib.pyplot as plt

from pygenn import GeNNModel

from pygenn import (init_postsynaptic, init_sparse_connectivity,
                    init_weight_update)

# Single cell parameters
V_THRESH = 20.0
V_RESET_E = 16.0
V_RESET_I = 13.0
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

PARAMS_E = {"C": TAU_M_E, "TauM": TAU_M_E, "Vrest": 0.0, "Vreset": V_RESET_E,
            "Vthresh": V_THRESH, "Ioffset": 0.0, "TauRefrac": TAU_REFRAC}
PARAMS_I = {"C": TAU_M_I, "TauM": TAU_M_I, "Vrest": 0.0, "Vreset": V_RESET_I,
            "Vthresh": V_THRESH, "Ioffset": 0.0, "TauRefrac": TAU_REFRAC}
VARS = {"V": 0.0, "RefracTime": 0.0}

model = GeNNModel("float", "tsodyks_synaptic_theory")
model.dt = 1.0

# Add neuron populations
i_pop = model.add_neuron_population("I", N_I, "LIF", PARAMS_I, VARS)
i_pop.spike_recording_enabled = True
e_non_select_pop = model.add_neuron_population("ENonSelect", N_E_NON_SELECT, "LIF", PARAMS_E, VARS)
e_non_select_pop.spike_recording_enabled = True
e_select_pop = [model.add_neuron_population(f"ESelect{i}", N_E_SELECT, "LIF", PARAMS_E, VARS)
                for i in range(P)]

# Add current sources
model.add_current_source("ICurr", "GaussianNoise", i_pop,
                         {"mean": MU_EXT_I, "sd": SIGMA_EXT})
model.add_current_source("ENonSelectCurr", "GaussianNoise", e_non_select_pop,
                         {"mean": MU_EXT_E, "sd": SIGMA_EXT})
for i, e in enumerate(e_select_pop):
    e.spike_recording_enabled = True
    model.add_current_source(f"ENonSelectCurr{i}", "GaussianNoise", e,
                             {"mean": MU_EXT_E, "sd": SIGMA_EXT})

# Add inhibitory synapse populations
model.add_synapse_population("I_I", "SPARSE",
                             i_pop, i_pop,
                             init_weight_update("StaticPulseConstantWeight", {"g": J_II}),
                             init_postsynaptic("DeltaCurr"),
                             init_sparse_connectivity("FixedNumberPreWithReplacement", {"num": int(C * N_I)}))

model.add_synapse_population("I_ENonSelect", "SPARSE",
                             i_pop, e_non_select_pop,
                             init_weight_update("StaticPulseConstantWeight", {"g": J_IE}),
                             init_postsynaptic("DeltaCurr"),
                             init_sparse_connectivity("FixedNumberPreWithReplacement", {"num": int(C * N_I)}))

for i, e in enumerate(e_select_pop):
    model.add_synapse_population(f"I_ESelect{i}", "SPARSE",
                                 i_pop, e,
                                 init_weight_update("StaticPulseConstantWeight", {"g": J_IE}),
                                 init_postsynaptic("DeltaCurr"),
                                 init_sparse_connectivity("FixedNumberPreWithReplacement", {"num": int(C * N_I)}))

# Add excitatory non-selective synapse populations
num_non_select_num_pre = int(C * (1.0 - (F * P)) * N_E)
print(f"num_non_select_num_pre={num_non_select_num_pre}")
model.add_synapse_population("ENonSelect_I", "SPARSE",
                             e_non_select_pop, i_pop,
                             init_weight_update("StaticPulseConstantWeight", {"g": J_EI}),
                             init_postsynaptic("DeltaCurr"),
                             init_sparse_connectivity("FixedNumberPreWithReplacement", {"num": num_non_select_num_pre}))

model.add_synapse_population("ENonSelect_ENonSelect", "SPARSE",
                             e_non_select_pop, e_non_select_pop,
                             init_weight_update("StaticPulseConstantWeight", {"g": J_B}),
                             init_postsynaptic("DeltaCurr"),
                             init_sparse_connectivity("FixedNumberPreWithReplacement", {"num": num_non_select_num_pre}))

for i, e in enumerate(e_select_pop):
    model.add_synapse_population(f"ENonSelect_ESelect{i}", "SPARSE",
                                 e_non_select_pop, e,
                                 init_weight_update("StaticPulseConstantWeight", {"g": J_B}),
                                 init_postsynaptic("DeltaCurr"),
                                 init_sparse_connectivity("FixedNumberPreWithReplacement", {"num": num_non_select_num_pre}))

num_select_num_pre = int(C * F * N_E)
print(f"num_select_num_pre={num_select_num_pre}")
for i_pre, e_pre in enumerate(e_select_pop):
    model.add_synapse_population(f"ESelect{i_pre}_I", "SPARSE",
                                 e_pre, i_pop,
                                 init_weight_update("StaticPulseConstantWeight", {"g": J_EI}),
                                 init_postsynaptic("DeltaCurr"),
                                 init_sparse_connectivity("FixedNumberPreWithReplacement", {"num": num_select_num_pre}))

    model.add_synapse_population(f"ESelect{i_pre}_ENonSelect", "SPARSE",
                                 e_pre, e_non_select_pop,
                                 init_weight_update("StaticPulseConstantWeight", {"g": J_B}),
                                 init_postsynaptic("DeltaCurr"),
                                 init_sparse_connectivity("FixedNumberPreWithReplacement", {"num": num_select_num_pre}))

    for i_post, e_post in enumerate(e_select_pop):
        model.add_synapse_population(f"ESelect{i_pre}_ESelect{i_post}", "SPARSE",
                                     e_pre, e_post,
                                     init_weight_update("StaticPulseConstantWeight", {"g": J_B}),
                                     init_postsynaptic("DeltaCurr"),
                                     init_sparse_connectivity("FixedNumberPreWithReplacement", {"num": num_select_num_pre}))

model.build()
model.load(num_recording_timesteps=2000)

while model.timestep < 2000:
    model.step_time()

model.pull_recording_buffers_from_device()

fig, axis = plt.subplots()

# Plot inhibitory
i_times, i_ids = i_pop.spike_recording_data[0]
axis.scatter(i_times, i_ids, s=1, label="I")

e_non_select_times, e_non_select_ids = e_non_select_pop.spike_recording_data[0]
axis.scatter(e_non_select_times, e_non_select_times + N_I, s=1, label="ENonSelect")

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

