import numpy as np
import matplotlib.pyplot as plt

from pygenn.genn_model import GeNNModel
from pygenn.genn_wrapper import NO_DELAY
from pygenn.genn_model import create_custom_weight_update_class
from pygenn.genn_model import create_custom_neuron_class


if_model = create_custom_neuron_class(
    "if_model",
    param_names=["Vtheta", "lambda", "Vrest", "Vreset"],
    var_name_types=[("V", "scalar")],
    sim_code="""
    if ($(V) >= $(Vtheta)) {
        $(V) = $(Vreset);
    }
    $(V) += (-$(lambda) + $(Isyn)) * DT;
    $(V) = fmax($(V), $(Vrest));
    """,
    reset_code="""
    """,
    threshold_condition_code="$(V) >= $(Vtheta)")

fusi_model = create_custom_weight_update_class(
    "fusi_model",
    param_names=["tauC", "a", "b", "thetaV", "thetaLUp", "thetaLDown", "thetaHUp", "thetaHDown",
                 "thetaX", "alpha", "beta", "Xmax", "Xmin", "JC", "Jplus", "Jminus"],
    var_name_types=[("X", "scalar")],
    post_var_name_types=[("C", "scalar")],
    sim_code="""
    $(addToInSyn, (($(X) > $(thetaX)) ? $(Jplus) : $(Jminus)));
    const scalar dt = $(t) - $(sT_post);
    const scalar decayC = $(C) * exp(-dt / $(tauC));
    if ($(V_post) > $(thetaV) && $(thetaLUp) < decayC && decayC < $(thetaHUp)) {
        $(X) += $(a);
    }
    else if ($(V_post) <= $(thetaV) && $(thetaLDown) < decayC && decayC < $(thetaHDown)) {
        $(X) -= $(b);
    }
    else {
        const scalar X_dt = $(t) - $(prev_sT_pre);
        if ($(X) > $(thetaX)) {
            $(X) += $(alpha) * X_dt;
        }
        else {
            $(X) -= $(beta) * X_dt;
        }
    }
    $(X) = fmin($(Xmax), fmax($(Xmin), $(X)));
    """,
    post_spike_code="""
    const scalar dt = $(t) - $(sT_post);
    $(C) = ($(C) * exp(-dt / $(tauC))) + $(JC);
    """,
    is_prev_pre_spike_time_required=True,
    is_post_spike_time_required=True)

TIMESTEP = 1.0
PRESENT_TIMESTEPS = 1500
POSTSYN_WT = 0.09

def get_spike_times(spike_list):
    return np.concatenate([np.ones_like(s) * i * TIMESTEP 
                           for i, s in enumerate(spike_list)])


if_params = {"Vtheta": 1.0,
             "lambda": 0.01,
             "Vrest": 0.0,
             "Vreset": 0.0}

if_init = {"V": 0.0}

fusi_params = {"tauC": 60.0, "a": 0.1, "b": 0.1, "thetaV": 0.8, "thetaLUp": 3.0,
               "thetaLDown": 3.0, "thetaHUp": 13.0, "thetaHDown": 4.0, "thetaX": 0.5,
               "alpha": 0.0035, "beta": 0.0035, "Xmax": 1.0, "Xmin": 0.0, "JC": 1.0,
               "Jplus": 1.0, "Jminus": 0.0}

fusi_init = {"X": 0.0}
fusi_post_init = {"C": 2.0}

presyn_params = {"rate" : 50.0}
extra_poisson_params = {"rate" : 100.0}
poisson_init = {"timeStepToSpike" : 0.0}

model = GeNNModel("float", "fusi")
model.dT = TIMESTEP

presyn = model.add_neuron_population("presyn", 1, "PoissonNew", presyn_params, poisson_init)
postsyn = model.add_neuron_population("postsyn", 1, if_model, if_params, if_init)
extra_poisson = model.add_neuron_population("extra_poisson", 10, "PoissonNew",
                                            extra_poisson_params, poisson_init)



pre2post = model.add_synapse_population(
            "pre2post", "DENSE_INDIVIDUALG", NO_DELAY,
            presyn, postsyn,
            fusi_model, fusi_params, fusi_init, {}, fusi_post_init,
            "DeltaCurr", {}, {})

extra_poisson2post = model.add_synapse_population(
            "extra_poisson2post", "DENSE_GLOBALG", NO_DELAY,
            extra_poisson, postsyn,
            "StaticPulse", {}, {"g": POSTSYN_WT}, {}, {},
            "DeltaCurr", {}, {})

#model.build()
model.load()

pre_spikes = []
post_spikes = []
post_v = []
c = []
x = []


while model.timestep < PRESENT_TIMESTEPS:
    model.step_time()

    # Record presynaptic spikes
    presyn.pull_current_spikes_from_device()
    pre_spikes.append(np.copy(presyn.current_spikes))
    
    # Record postsynaptic spikes
    postsyn.pull_current_spikes_from_device()
    post_spikes.append(np.copy(postsyn.current_spikes))

    # Record value of postsyn_V
    model.pull_var_from_device("postsyn", "V")
    post_v.append(np.copy(postsyn.vars["V"].view))

    # Record value of X
    pre2post.pull_var_from_device("X")
    x.append(np.copy(pre2post.get_var_values("X")))
    
    # Record value of C
    pre2post.pull_var_from_device("C")
    c.append(np.copy(pre2post.post_vars["C"].view))

pre_spike_times = get_spike_times(pre_spikes)
post_spike_times = get_spike_times(post_spikes)

post_v = np.concatenate(post_v)
c = np.concatenate(c)
x = np.concatenate(x)

postsyn_spike_rate = float(len(post_spike_times)) / (PRESENT_TIMESTEPS / 1000)

# Create plot
fig, axes = plt.subplots(4, sharex=True, figsize=(12, 5))
fig.tight_layout(pad=2.0)

# plot presyn spikes
axes[0].set_xlim((0,PRESENT_TIMESTEPS))
axes[0].vlines(pre_spike_times, 0, 1)
axes[0].title.set_text("Presynaptic spikes")

# plot X
axes[1].title.set_text("Synaptic internal variable X(t)")
axes[1].plot(x)
axes[1].set_ylim((0,1))
axes[1].axhline(0.5, linestyle="--", color="black", linewidth=0.5)
axes[1].set_yticklabels(["0", "$\\theta_X$", "1"])

# plot postsyn V
axes[2].title.set_text('Postsynaptic voltage V(t) (Spike rate: ' + str(postsyn_spike_rate) + " Hz)")
axes[2].plot(post_v)
axes[2].set_ylim((0,1.2))
axes[2].axhline(1, linestyle="--", color="black", linewidth=0.5)
axes[2].axhline(0.8, linestyle="--", color="black", linewidth=0.5)
axes[2].vlines(post_spike_times, 0, 1, color="red", linewidth=0.5)

# plot C
axes[3].plot(c)
axes[3].title.set_text("Calcium variable C(t)")
for i in [3, 4, 13]:
    axes[3].axhline(i, linestyle="--", color="black", linewidth=0.5)

plt.show()