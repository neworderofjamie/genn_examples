import math
import pylab
import pynn_genn as sim

pylab.rc("text", usetex = True)

labels = ["Regular spiking", "Fast spiking", "Chattering", "Intrinsically bursting"]
a = [0.02,  0.1,    0.02,   0.02]
b = [0.2,   0.2,    0.2,    0.2]
c = [-65.0, -65.0,  -50.0,  -55.0]
d = [8.0,   2.0,    2.0,    4.0]

sim.setup(timestep=0.1, min_delay=0.1, max_delay=4.0)
ifcell = sim.Population(len(labels),
                        sim.Izhikevich(i_offset=0.01, a=a, b=b, c=c, d=d))
ifcell.record("v")

sim.run(200.0)

data = ifcell.get_data()

sim.end()

figure, axes = pylab.subplots(2, 2, sharex="col", sharey="row")

axes[0, 0].set_ylabel("Membrane potential [mV]")
axes[1, 0].set_ylabel("Membrane potential [mV]")
axes[1, 0].set_xlabel("Time [ms]")
axes[1, 1].set_xlabel("Time [ms]")

signal = data.segments[0].analogsignals[0]
for i, l in enumerate(labels):
    axis = axes[i // 2, i % 2]

    axis.plot(signal.times, signal[:, i])
    axis.set_title(l)
    
#figure.savefig("izk.png")
pylab.show()
