from pygenn.genn_model import GeNNModel
import matplotlib.pyplot as plt
import numpy as np

model = GeNNModel("float", "tennHH")
model.dT = 0.1

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

pop1 = model.add_neuron_population("Pop1", 10, "TraubMiles", p, ini)

model.build()
model.load()

v = np.empty((10000, 10))
v_view = pop1.vars["V"].view
while model.t < 1000.0:
    model.step_time()

    pop1.pull_var_from_device("V")
    
    v[model.timestep - 1,:]=v_view[:]

fig, axis = plt.subplots()
axis.plot(v)
plt.show()
