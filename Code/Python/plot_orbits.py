
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
from CR3BP import *
from helper_functions import *
from plotting import *

def generate_propagation(x0, period):
    print(period)
    new_propagation = solve_ivp(CR3BP_DEs, [0, period], x0, args=(mu,), rtol=1e-12, atol=1e-12).y
    return new_propagation

L2_halo_S = np.loadtxt("L2_halo/L2_halo_southern_ICs.csv", delimiter=",")
# L1_halo_N = np.loadtxt("L1_halo/L1_halo_northern_ICs.csv", delimiter=",")
L1_lyapunov = np.loadtxt("L1_lyapunov/L1_lyapunov_ICs.csv", delimiter=",")
DRO = np.loadtxt("DRO/DRO_ICs.csv", delimiter=",")
L2_halo_S = np.array([[1.149946551054244370e+00, 0, -1.465767685210560556e-01, 0, -2.181937221565175000e-01, 0, 3.193209656999975277e+00]])
L1_lyapunov = np.array([[6.139604230137676311e-01, 0, 0, 0, 8.397208166184089162e-01, 0, 6.82]])
DRO = np.array([[9.255293901714612970e-01, 0, 0, 0, 5.121704760209937479e-01, 0, 0.8]])
L1_halo_N = np.array([[0.82338697, 0, -0.02282543, 0, 0.13455489, 0, 2.74650039]])

data_list = [L2_halo_S, L1_lyapunov, DRO, L1_halo_N]

propagations = []

for data in data_list:
    ICs = data[:, 0:6]
    periods = data[:, 6]
    num_orbits = len(periods)

    propagations.append(Parallel(n_jobs=8)(delayed(generate_propagation)(ICs[orbit_index], periods[orbit_index]) for orbit_index in range(0, num_orbits, 4)))

# propagations = []
# for orbit_index in range(num_orbits):
#     print(orbit_index)
#     x0 = ICs[orbit_index]
#     period = periods[orbit_index]

#     new_propagation = solve_ivp(CR3BP_DEs, [0, period], x0, args=(mu,), rtol=1e-12, atol=1e-12).y
#     propagations.append(new_propagation)



colors = ["blue", "red", "green", "orange"]
ax = plt.figure().add_subplot(projection="3d")
for type_index in range(4):
    type_propagations = propagations[type_index]
    num_orbits = len(type_propagations)
    color = colors[type_index]
    for orbit_index in range(num_orbits):
        tbp = type_propagations[orbit_index]
        ax.plot(tbp[0], tbp[1], tbp[2], alpha=0.5, c=color)
        ax.scatter(tbp[0, 0], tbp[1, 0], tbp[2, 0], c=color)
plot_moon(ax, mu)
ax.set_aspect("equal")

plt.show()