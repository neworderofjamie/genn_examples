import matplotlib.pyplot as plt
import numpy as np

from pygenn import GeNNModel

from pygenn import (create_neuron_model, init_postsynaptic,
                    init_sparse_connectivity, init_weight_update)

# Define custom SEMD model based on standard LIF
# * 'Trigger' input is received on standard 'ISyn' channel
# * 'Fac' input is received on additional 'IsynFac' channel
semd_model = create_neuron_model(
    "semd",
    sim_code=
    """
    if(ISynFac > 0.00001) {
       ISynTrigger += trigInit * Isyn;
    }
    if (RefracTime <= 0.0) {
       const scalar inputCurrent = ISynTrigger * ISynFac;
       const scalar alpha = ((inputCurrent + Ioffset) * Rmembrane) + Vrest;
       V = alpha - (ExpTC * (alpha - V));
    }
    else {
      RefracTime -= dt;
    }
    ISynTrigger *= trigExpDecay;
    """,
    threshold_condition_code="RefracTime <= 0.0 && V >= Vthresh",
    reset_code=
    """
    V = Vreset;
    RefracTime = TauRefrac;
    """,
    
    params=["C", "TauM", "Vrest", "Vreset", "Vthresh", "Ioffset", "TauRefrac", "TauSynTrig"],
    derived_params=[("ExpTC", lambda pars, dt: np.exp(-dt / pars["TauM"])),
                    ("Rmembrane", lambda pars, dt: pars["TauM"] / pars["C"]),
                    ("trigExpDecay", lambda pars, dt: np.exp(-dt / pars["TauSynTrig"])),
                    ("trigInit", lambda pars, dt: (pars["TauSynTrig"] * (1.0 - np.exp(-dt / pars["TauSynTrig"]))) * (1.0 / dt))],
    vars=[("V", "scalar"), ("RefracTime", "scalar"), ("ISynTrigger", "scalar")],
    additional_input_vars=[("ISynFac", "scalar", 0.0)])

# Parameters
semd_params = {"C": 0.25, "TauM": 10.0, "Vrest": 0.0, "Vreset": 0.0,
               "Vthresh": 5.0, "Ioffset": 0.0, "TauRefrac": 0.0, "TauSynTrig": 3.0}
semd_vars = {"V": 0.0, "RefracTime": 0.0, "ISynTrigger": 0.0}


# Create model
model = GeNNModel("float", "semd")
model.dt = 0.1

# Add facilitation population with 1 spike source which fires a single spike in first timestep
fac_pop = model.add_neuron_population("Fac", 1, "SpikeSourceArray", 
                                      {}, {"startSpike": 0, "endSpike": 1})
fac_pop.extra_global_params["spikeTimes"].set_init_values([0.0])

# Add trigger population with 10 spike sources which fire in sequence every 3 timesteps
trig_pop = model.add_neuron_population("Trig", 10, "SpikeSourceArray", 
                                       {}, {"startSpike": np.arange(10), "endSpike": np.arange(1, 11)})
trig_pop.spike_recording_enabled = True
trig_pop.extra_global_params["spikeTimes"].set_init_values(np.arange(0.0, 30.0, 3.0))

# Add output population of 10 SEMD neurons
out_pop = model.add_neuron_population("Output", 10, semd_model,
                                      semd_params, semd_vars)
out_pop.spike_recording_enabled = True

# Add synapse population connecting facilitation to all output neurons
# **NOTE** this has exponential synapse so facilitation decays with 5ms timeconstant
fac_out = model.add_synapse_population(
    "FacOut", "DENSE",
    fac_pop, out_pop,
    init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
    init_postsynaptic("ExpCurr", {"tau": 5.0}))

# Connect this to output populations additiona 'facilitation' input
fac_out.post_target_var = "ISynFac"

# Add synapse population connection each trigger to corresponding output neuron
# **NOTE** this has delta synapse because neuron model already decays this input
trigger_out = model.add_synapse_population(
    "TrigOut", "SPARSE",
    trig_pop, out_pop,
    init_weight_update("StaticPulseConstantWeight", {"g": 20.0}),
    init_postsynaptic("DeltaCurr"),
    init_sparse_connectivity("OneToOne"))

# Build and load model
model.build()
model.load(num_recording_timesteps=400)

# Simulate, recording V and Trigger every timestep
out_v = []
out_fac = []
out_trigger = []
while model.t < 40.0:
    model.step_time()
    out_pop.vars["V"].pull_from_device()
    out_pop.vars["ISynTrigger"].pull_from_device()
    fac_out.out_post.pull_from_device()
    out_v.append(out_pop.vars["V"].values)
    out_trigger.append(out_pop.vars["ISynTrigger"].values)
    out_fac.append(fac_out.out_post.view[0,0])

# Stack recordings together
out_v = np.vstack(out_v)
out_trigger = np.vstack(out_trigger)

# Download spikes
model.pull_recording_buffers_from_device()

# Plot
fig, axes = plt.subplots(11, sharex=True)
timesteps = np.arange(0.0, 40.0, 0.1)

# Plot trigger spikes
trig_spike_times, trig_spike_ids = trig_pop.spike_recording_data[0]
axes[0].scatter(trig_spike_times, trig_spike_ids, s=1)
axes[0].set_title("Trigger spikes")

# Count output spikes
out_spike_times, out_spike_ids = out_pop.spike_recording_data[0]
out_spike_counts = np.bincount(out_spike_ids, minlength=10)

# Loop through output neurons
for i in range(10):
    ax = axes[1 + i]
    
    # Plot membrane voltage; and facilitation and trigger currents
    v_actor = ax.plot(timesteps, out_v[:,i])[0]
    trigger_actor = ax.plot(timesteps, out_trigger[:,i])[0]
    fac_actor = ax.plot(timesteps, out_fac)[0]
    
    ax.set_title(f"Output {i}: {out_spike_counts[i]} spikes")
    ax.set_ylim((0.0, 20.0))

axes[-1].set_xlabel("Time [ms]")

fig.legend([v_actor, trigger_actor, fac_actor], ["V", "Trigger", "Fac"], loc="lower center", ncol=3)
plt.show()

