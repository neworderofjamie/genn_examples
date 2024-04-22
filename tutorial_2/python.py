import matplotlib.pyplot as plt
import numpy as np
from pygenn import (GeNNModel, init_postsynaptic, init_sparse_connectivity,
                    init_weight_update, create_sparse_connect_init_snippet,
                    create_var_ref)

ring_model = create_sparse_connect_init_snippet(
    "ring",
    row_build_code=
        """
        addSynapse((id_pre + 1) % num_post);
        """,

    calc_max_row_len_func=lambda num_pre, num_post, pars: 1)

model = GeNNModel("float", "tennHHRing")
model.dt = 0.1

p = {"gNa": 7.15,   # Na conductance in [muS]
     "ENa": 50.0,   # Na equi potential [mV]
     "gK": 1.43,    # K conductance in [muS]
     "EK": -95.0,   # K equi potential [mV] 
     "gl": 0.02672, # leak conductance [muS]
     "El": -63.563, # El: leak equi potential in mV, 
     "C": 0.143}    # membr. capacity density in nF

ini = {"V": -60.0,      # membrane potential
       "m": 0.0529324,  # prob. for Na channel activation
       "h": 0.3176767,  # prob. for not Na channel blocking
       "n": 0.5961207}  # prob. for K channel activation


stim_ini = {"startSpike": [0], "endSpike": [1]}

pop1 = model.add_neuron_population("Pop1", 10, "TraubMiles", p, ini)
stim = model.add_neuron_population("Stim", 1, "SpikeSourceArray", {}, stim_ini)

weight_init = init_weight_update("StaticPulseConstantWeight", {"g": -0.2})
postsynaptic_init = init_postsynaptic("ExpCond", {"tau": 1.0, "E": -80.0}, 
                                      var_refs={"V": create_var_ref(pop1, "V")})

model.add_synapse_population("Pop1self", "SPARSE", 10,
    pop1, pop1,
    weight_init, postsynaptic_init,
    init_sparse_connectivity(ring_model, {}))

model.add_synapse_population("StimPop1", "SPARSE", 0,
    stim, pop1,
    weight_init, postsynaptic_init,
    init_sparse_connectivity("OneToOne", {}))
stim.extra_global_params["spikeTimes"].set_values([0.0])

model.build()
model.load()

v_rec = np.empty((2000, 10))
v = pop1.vars["V"]
while model.t < 200.0:
    model.step_time()

    v.pull_from_device()
    v_rec[model.timestep - 1,:]= v.view[:]

fig, axis = plt.subplots()
axis.plot(v_rec)
plt.show()
