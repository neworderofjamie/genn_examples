import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("tenHHModel_output.V.dat", dtype=float)

times = data[:,0]
voltages = data[:,1:]

print(voltages.shape)
# Create plot
figure, axis = plt.subplots()
axis.set_xlabel("time [ms]")
axis.set_ylabel("membrane voltage [mV]")
# Plot voltages
for i in range(voltages.shape[1]):
    axis.plot(times, voltages[:,i])

# Show plot
plt.show()
