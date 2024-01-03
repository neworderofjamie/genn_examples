import numpy as np
import matplotlib.pyplot as plt
from time import time

from pygenn import (GeNNModel, create_weight_update_model, 
                    init_postsynaptic, init_var, init_weight_update)

# STDP synapse with additive weight dependence
stdp_additive = create_weight_update_model(
    "STDPAdditive",
    params=["tauPlus", "tauMinus", "aPlus", "aMinus", "wMin", "wMax"],
    var_name_types=[("g", "scalar")],
    pre_var_name_types=[("preTrace", "scalar")],
    post_var_name_types=[("postTrace", "scalar")],
    derived_params=[("aPlusScaled", lambda pars, dt: pars["aPlus"] * (pars["wMax"] - pars["wMin"])),
                    ("aMinusScaled", lambda pars, dt: pars["aMinus"] * (pars["wMax"] - pars["wMin"]))],

    pre_spike_syn_code=
        """
        addToPost(g);
        const scalar dt = t - st_post;
        if(dt > 0) {
            const scalar timing = exp(-dt / tauMinus);
            const scalar newWeight = g - (aMinusScaled * postTrace * timing);
            g = fmin(wMax, fmax(wMin, newWeight));
        }
        """,

    post_spike_syn_code=
        """
        const scalar dt = t - st_pre;
        if(dt > 0) {
            const scalar timing = exp(-dt / tauPlus);
            const scalar newWeight = g + (aPlusScaled * preTrace * timing);
            $(g) = fmin(wMax, fmax(wMin, newWeight));
        }
        """,

    pre_spike_code=
        """
        const scalar dt = t - st_pre;
        preTrace = preTrace * exp(-dt / tauPlus) + 1.0;
        """,

    post_spike_code=
        """
        const scalar dt = t - st_post;
        postTrace = postTrace * exp(-dt / tauMinus) + 1.0;
        """)

# STDP synapse with multiplicative weight dependence
stdp_multiplicative = create_weight_update_model(
    "STDPMultiplicative",
    params=["tauPlus", "tauMinus", "aPlus", "aMinus", "wMin", "wMax"],
    var_name_types=[("g", "scalar")],
    pre_var_name_types=[("preTrace", "scalar")],
    post_var_name_types=[("postTrace", "scalar")],

    pre_spike_syn_code=
        """
        addToPost(g);
        const scalar dt = t - st_post;
        if(dt > 0) {
            const scalar timing = exp(-dt / tauMinus);
            g -= (g - wMin) * aMinus * postTrace * timing;
        }
        """,

    post_spike_syn_code=
        """
        const scalar dt = t - st_pre;
        if(dt > 0) {
            const scalar timing = exp(-dt / tauPlus);
            g += (wMax - g) * aPlus * preTrace * timing;
        }
        """,

    pre_spike_code=
        """
        const scalar dt = t - st_pre;
        preTrace = preTrace * exp(-dt / tauPlus) + 1.0;
        """,

    post_spike_code=
        """
        const scalar dt = t - st_post;
        postTrace = postTrace * exp(-dt / tauMinus) + 1.0;
        """)

DT = 1.0
NUM_EX_SYNAPSES = 1000
G_MAX = 0.01
DURATION_MS = 300000.0

A_PLUS = 0.01
A_MINUS = 1.05 * A_PLUS


model = GeNNModel("float", "song")
model.dt = DT

lif_params = {"C": 0.17, "TauM": 10.0, "Vrest": -74.0, "Vreset": -60.0,
              "Vthresh": -54.0, "Ioffset": 0.0, "TauRefrac": 1.0}

lif_init = {"V": -60.0, "RefracTime": 0.0}

poisson_params = {"rate" : 15.0}

poisson_init = {"timeStepToSpike" : 0.0}

post_syn_params = {"tau": 5.0}

stdp_init = {"g": init_var("Uniform", {"min": 0.0, "max": G_MAX})}
stdp_params = {"tauPlus": 20.0, "tauMinus": 20.0, "aPlus": A_PLUS, "aMinus": A_MINUS, "wMin": 0.0, "wMax": G_MAX}
stdp_pre_init = {"preTrace": 0.0}
stdp_post_init = {"postTrace": 0.0}

# Create neuron populations
additive_pop = model.add_neuron_population("additive", 1, "LIF", lif_params, lif_init)
multiplicative_pop = model.add_neuron_population("multiplicative", 1, "LIF", lif_params, lif_init)

poisson_pop = model.add_neuron_population("input", NUM_EX_SYNAPSES, "PoissonNew", poisson_params, poisson_init)

input_additive = model.add_synapse_population("input_additive", "DENSE",
    poisson_pop, additive_pop,
    init_weight_update(stdp_additive, stdp_params, stdp_init, stdp_pre_init, stdp_post_init),
    init_postsynaptic("ExpCurr", post_syn_params))

input_multiplicative = model.add_synapse_population("input_multiplicative", "DENSE",
    poisson_pop, multiplicative_pop,
    init_weight_update(stdp_multiplicative, stdp_params, stdp_init, stdp_pre_init, stdp_post_init),
    init_postsynaptic("ExpCurr", post_syn_params))

print("Building Model")
model.build()
print("Loading Model")
model.load()

print("Simulating")

# Simulate
while model.t < DURATION_MS:
    model.step_time()

# Pull weight variables from device
input_additive.vars["g"].pull_from_device()
input_multiplicative.vars["g"].pull_from_device()

# Get weights
weights = [input_additive.vars["g"].values,
           input_multiplicative.vars["g"].values]

figure, axes = plt.subplots(1, 2, sharey=True)

axes[0].set_xlabel("Normalised weight")

for i, (w, l) in enumerate(zip(weights, ["Additive", "Multiplicative"])):
    axes[i].hist(w / G_MAX, np.linspace(0.0, 1.0, 20))
    axes[i].set_title("%s weight dependence" % l)
    axes[i].set_ylabel("Number of synapses")
    #axes[i].set_xlim((0.0, 1.0))

plt.show()
