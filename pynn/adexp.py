import matplotlib.pyplot as plt
import numpy as np

from pygenn import GeNNModel

from pygenn import create_neuron_model

def dv(v: str, w: str):
    return f"(1.0 / c) * ((-gL * (({v}) - eL)) + (gL * deltaT * exp((({v}) - vThresh) / deltaT)) + i - ({w}))"

def dw(v: str, w: str):
    return f"(1.0 / tauW) * ((a * ({v} - eL)) - {w})"

adexp_model = create_neuron_model(
    "adexp",
    params=["c",        # Membrane capacitance [pF]
            "gL",       # Leak conductance [nS]
            "eL",       # Leak reversal potential [mV]
            "deltaT",   # Slope factor [mV]
            "vThresh",  # Threshold voltage [mV]
            "vSpike",   # Artificial spike height [mV]
            "vReset",   # Reset voltage [mV]
            "tauW",     # Adaption time constant
            "a",        # Subthreshold adaption [nS]
            "b",        # Spike-triggered adaptation [nA]
            "iOffset"], # Offset current
    vars=[("V", "scalar"), ("W", "scalar")],
    
    sim_code=
        f"""
        const scalar i = Isyn + iOffset;
        // If voltage is above artificial spike height
        if(V >= vSpike) {{
           V = vReset;
        }}
        // Calculate RK4 terms
        const scalar v1 = {dv('V', 'W')};
        const scalar w1 = {dw('V', 'W')};
        const scalar v2 = {dv('V + (dt * 0.5 * v1)', 'W + (dt * 0.5 * w1)')};
        const scalar w2 = {dw('V + (dt * 0.5 * v1)', 'W + (dt * 0.5 * w1)')};
        const scalar v3 = {dv('V + (dt * 0.5 * v2)', 'W + (dt * 0.5 * w2)')};
        const scalar w3 = {dw('V + (dt * 0.5 * v2)', 'W + (dt * 0.5 * w2)')};
        const scalar v4 = {dv('V + (dt * v3)', 'W + (dt * w3)')};
        const scalar w4 = {dw('V + (dt * v3)', 'W + (dt * w3)')};
        // Update V
        V += (dt / 6.0) * (v1 + (2.0 * (v2 + v3)) + v4);
        // If we're not above peak, update w
        // **NOTE** it's not safe to do this at peak as wn may well be huge
        if(V <= -40.0) {{
           W += (dt / 6.0) * (w1 + (2.0 * (w2 + w3)) + w4);
        }}
        """,

    threshold_condition_code="V > -40",

    reset_code=
        """
        // **NOTE** we reset v to arbitrary plotting peak rather than to actual reset voltage
        V = vSpike;
        W += (b * 1000.0);
        """)

# Parameters
adexp_params = {
    "c":        281.0,
    "gL":       30.0,
    "eL":       -70.6,
    "deltaT":   2.0,
    "vThresh":  -50.4,
    "vSpike":   10.0,
    "vReset":   -70.6,
    "tauW":     144.0,
    "a":        4.0,
    "b":        0.0805,
    "iOffset":  700.0}
adexp_vars = {"V": -70.6, "W": 0.0}


# Create model
model = GeNNModel("float", "adexp", backend="single_threaded_cpu")
model.dt = 0.1

# Add facilitation population with 1 spike source which fires a single spike in first timestep
adexp_pop = model.add_neuron_population("AdExp", 1, adexp_model, 
                                        adexp_params, adexp_vars)


# Build and load model
model.build()
model.load()

# Simulate, recording V and Trigger every timestep
adexp_v = []
adexp_w = []
while model.t < 1000.0:
    model.step_time()
    adexp_pop.vars["V"].pull_from_device()
    adexp_pop.vars["W"].pull_from_device()
    adexp_v.append(adexp_pop.vars["V"].values)
    adexp_w.append(adexp_pop.vars["W"].values)

# Stack recordings together
adexp_v = np.vstack(adexp_v)
adexp_w = np.vstack(adexp_w)

# Create plot
figure, axes = plt.subplots(2)

# Plot voltages
axes[0].set_title("Voltage")
axes[0].plot(adexp_v)

axes[1].set_title("Adaption current")
axes[1].plot(adexp_w)

# Show plot
plt.show()


