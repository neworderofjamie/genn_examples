import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------
# Sjostrom et al. (2001) experimental data
data_w = [
    [-0.29, -0.41, -0.34, 0.56, 0.75],
    [-0.04, 0.14, 0.29, 0.53, 0.56]
]
data_e = [
    [0.08, 0.11, 0.1, 0.32, 0.19],
    [0.05, 0.1, 0.14, 0.11, 0.26]
]

data = np.loadtxt("weights.csv", delimiter=",", skiprows=1,
                  dtype={"names": ("frequency", "delta_t", "raw_weight"),
                         "formats": (np.float, np.float, np.float)})

negative_mask = (data["delta_t"] == -10.0)
positive_mask = np.logical_not(negative_mask)
frequencies = data["frequency"][negative_mask]

weights = [data["raw_weight"][negative_mask],
           data["raw_weight"][positive_mask]]

# Create plot
figure, axis = plt.subplots()

# Plot Frequency response
axis.set_xlabel("Frequency/Hz")
axis.set_ylabel(r"$(\frac{\Delta w_{ij}}{w_{ij}})$", rotation="horizontal",
                size="xx-large")

line_styles = ["--", "-"]
delta_t = [-10, 10]
for m_w, d_w, d_e, t, l in zip(weights, data_w, data_e, data["delta_t"], line_styles):
    # Calculate deltas from end weights
    delta_w = (m_w - 0.5) / 0.5

    # Plot experimental data and error bars
    axis.errorbar(frequencies, d_w, yerr=d_e, color="black", linestyle=l,
                label=r"Experimental data, delta $(\Delta{t}=%dms)$" % t)

    # Plot model data
    axis.plot(frequencies, delta_w, color="blue", linestyle=l,
            label=r"Triplet rule, delta $(\Delta{t}=%dms)$" % t)

axis.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))

# Show plot
plt.show()
