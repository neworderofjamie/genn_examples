import pynn_genn as sim
import numpy as np
sim.setup(timestep=1.0, min_delay=1.0)
neurons = sim.Population(3, sim.Izhikevich(a=0.02, b=0.2, c=-65, d=6, i_offset=[0.014, 0.0, 0.0]))
spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=np.arange(10.0, 51, 1)))


connection = sim.Projection(spike_source, neurons[1:2], sim.OneToOneConnector(),
                            sim.StaticSynapse(weight=3.0, delay=1.0),
                            receptor_type='excitatory')
electrode = sim.DCSource(start=2.0, stop=92.0, amplitude=0.014)
electrode.inject_into(neurons[2:3])

neurons.record(['v'])  # , 'u'])
neurons.initialize(v=-70.0, u=-14.0)
sim.run(100.0)
