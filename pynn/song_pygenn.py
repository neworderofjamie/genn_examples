import numpy as np
import matplotlib.pyplot as plt
from time import time

from pygenn import genn_wrapper
from pygenn import genn_model

import pygenn_models
#from pygenn_models import (lif_model, exp_curr_model,
#                           stdp_additive)

# LIF neuron model
lif_model = genn_model.create_custom_neuron_class(
    "LIF",
    param_names=["C", "TauM", "Vrest", "Vreset", "Vthresh", "Ioffset", "TauRefrac"],
    var_name_types=[(vn, "scalar") for vn in ["V", "RefracTime"]],
    sim_code="""
    if ($(RefracTime) <= 0.0)
    {
        scalar alpha = (($(Isyn) + $(Ioffset)) * $(Rmembrane)) + $(Vrest);
        $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
    }
    else
    {
        $(RefracTime) -= DT;
    }
    """,
    reset_code="""
    $(V) = $(Vreset);
    $(RefracTime) = $(TauRefrac);""",
    threshold_condition_code="$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)",
    derived_params=[
        ("ExpTC", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[1]))()),
        ("Rmembrane", genn_model.create_dpf_class(lambda pars, dt: pars[1] / pars[0])())])()

# ExpCurr synapse
exp_curr_model = genn_model.create_custom_postsynaptic_class(
    "ExpCurr",
    param_names=["tau"],
    decay_code="$(inSyn) *= $(expDecay);",
    apply_input_code="$(Isyn) += $(init) * $(inSyn);",
    derived_params=[("expDecay", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))()),
                   ("init", genn_model.create_dpf_class(lambda pars, dt: (pars[0] * (1.0 - np.exp(-dt / pars[0]))) * (1.0 / dt))())])()

# STDP synapse with additive weight dependence
stdp_additive = genn_model.create_custom_weight_update_class(
    "STDPAdditive",
    param_names=["tauPlus", "tauMinus", "aPlus", "aMinus", "wMin", "wMax"],
    var_name_types=[("g", "scalar")],
    pre_var_name_types=[("preTrace", "scalar")],
    post_var_name_types=[("postTrace", "scalar")],
    derived_params=[("aPlusScaled", genn_model.create_dpf_class(lambda pars, dt: pars[2] * (pars[5] - pars[4]))()),
                    ("aMinusScaled", genn_model.create_dpf_class(lambda pars, dt: pars[3] * (pars[5] - pars[4]))())],

    sim_code=
        """
        $(addToInSyn, $(g));
        const scalar dt = $(t) - $(sT_post);
        if(dt > 0) {
            const scalar timing = exp(-dt / $(tauMinus));
            const scalar newWeight = $(g) - ($(aMinusScaled) * $(postTrace) * timing);
            $(g) = min($(wMax), max($(wMin), newWeight));
        }
        """,

    learn_post_code=
        """
        const scalar dt = $(t) - $(sT_pre);
        if(dt > 0) {
            const scalar timing = exp(-dt / $(tauPlus));
            const scalar newWeight = $(g) + ($(aPlusScaled) * $(preTrace) * timing);
            $(g) = min($(wMax), max($(wMin), newWeight));
        }
        """,

    pre_spike_code=
        """
        const scalar dt = $(t) - $(sT_pre);
        $(preTrace) = $(preTrace) * exp(-dt / $(tauPlus)) + 1.0;
        """,

    post_spike_code=
        """
        const scalar dt = $(t) - $(sT_post);
        $(postTrace) = $(postTrace) * exp(-dt / $(tauMinus)) + 1.0;
        """,

    is_pre_spike_time_required=True,
    is_post_spike_time_required=True)()

# STDP synapse with multiplicative weight dependence
stdp_multiplicative = genn_model.create_custom_weight_update_class(
    "STDPMultiplicative",
    param_names=["tauPlus", "tauMinus", "aPlus", "aMinus", "wMin", "wMax"],
    var_name_types=[("g", "scalar")],
    pre_var_name_types=[("preTrace", "scalar")],
    post_var_name_types=[("postTrace", "scalar")],

    sim_code=
        """
        $(addToInSyn, $(g));
        const scalar dt = $(t) - $(sT_post);
        if(dt > 0) {
            const scalar timing = exp(-dt / $(tauMinus));
            $(g) -= ($(g) - $(wMin)) * $(aMinus) * $(postTrace) * timing;
        }
        """,

    learn_post_code=
        """
        const scalar dt = $(t) - $(sT_pre);
        if(dt > 0) {
            const scalar timing = exp(-dt / $(tauPlus));
            $(g) += ($(wMax) - $(g)) * $(aPlus) * $(preTrace) * timing;
        }
        """,

    pre_spike_code=
        """
        const scalar dt = $(t) - $(sT_pre);
        $(preTrace) = $(preTrace) * exp(-dt / $(tauPlus)) + 1.0;
        """,

    post_spike_code=
        """
        const scalar dt = $(t) - $(sT_post);
        $(postTrace) = $(postTrace) * exp(-dt / $(tauMinus)) + 1.0;
        """,

    is_pre_spike_time_required=True,
    is_post_spike_time_required=True)()

DT = 1.0
NUM_EX_SYNAPSES = 1000
G_MAX = 0.01
DURATION_MS = 300000.0

A_PLUS = 0.01
A_MINUS = 1.05 * A_PLUS


model = genn_model.GeNNModel("float", "song", enable_debug=False, cpu_only=False)
model.dT = DT

model.default_var_mode = genn_wrapper.VarMode_LOC_HOST_DEVICE_INIT_DEVICE

lif_params = {"C": 0.17, "TauM": 10.0, "Vrest": -74.0, "Vreset": -60.0,
              "Vthresh": -54.0, "Ioffset": 0.0, "TauRefrac": 1.0}

lif_init = {"V": -60.0, "RefracTime": 0.0}

poisson_params = {"rate" : 15.0}

poisson_init = {"timeStepToSpike" : 0.0}

post_syn_params = {"tau": 5.0}

stdp_init = {"g": genn_model.init_var("Uniform", {"min": 0.0, "max": G_MAX})}
stdp_params = {"tauPlus": 20.0, "tauMinus": 20.0, "aPlus": A_PLUS, "aMinus": A_MINUS, "wMin": 0.0, "wMax": G_MAX}
stdp_pre_init = {"preTrace": 0.0}
stdp_post_init = {"postTrace": 0.0}

# Create neuron populations
additive_pop = model.add_neuron_population("additive", 1, lif_model, lif_params, lif_init)
multiplicative_pop = model.add_neuron_population("multiplicative", 1, lif_model, lif_params, lif_init)

poisson_pop = model.add_neuron_population("input", NUM_EX_SYNAPSES, "PoissonNew", poisson_params, poisson_init)

input_additive = model.add_synapse_population("input_additive", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
    poisson_pop, additive_pop,
    stdp_additive, stdp_params, stdp_init, stdp_pre_init, stdp_post_init,
    exp_curr_model, post_syn_params, {})

input_multiplicative = model.add_synapse_population("input_multiplicative", "DENSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
    poisson_pop, multiplicative_pop,
    stdp_multiplicative, stdp_params, stdp_init, stdp_pre_init, stdp_post_init,
    exp_curr_model, post_syn_params, {})

print("Building Model")
model.build()
print("Loading Model")
model.load()

print("Simulating")

# Simulate
while model.t < DURATION_MS:
    model.step_time()

# Pull synaptic state from device
model.pull_state_from_device("input_additive")
model.pull_state_from_device("input_multiplicative")

# Get weights
weights = [input_additive.get_var_values("g"),
           input_multiplicative.get_var_values("g")]

figure, axes = plt.subplots(1, 2, sharey=True)

axes[0].set_xlabel("Normalised weight")

for i, (w, l) in enumerate(zip(weights, ["Additive", "Multiplicative"])):
    axes[i].hist(w / G_MAX, np.linspace(0.0, 1.0, 20))
    axes[i].set_title("%s weight dependence" % l)
    axes[i].set_ylabel("Number of synapses")
    #axes[i].set_xlim((0.0, 1.0))

plt.show()