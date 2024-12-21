
import numpy as np
import scipy
import matplotlib.pyplot as plt
from configuration_forback_shooting import *
from CR3BP_pontryagin import *
from plotting import *

initial_costate = np.loadtxt("LT_transfers/solution_02_1_25.csv", delimiter=",")[-1, 0:6]

initial_conditions = np.concatenate((initial_state, initial_costate))

tspan = np.array([0, tf])
propagation = scipy.integrate.solve_ivp(minimum_fuel_ODE, tspan, initial_conditions, args=(mu, umax, truth_rho), atol=1e-12, rtol=1e-12).y

final_conditions = propagation[:, -1]
control = get_min_fuel_control(propagation[6:12, :], umax, truth_rho)

initial_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, np.array([0, 4]), initial_state, args=(mu,), atol=1e-12, rtol=1e-12).y
final_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, np.array([0, 4]), final_conditions[0:6], args=(mu,), atol=1e-12, rtol=1e-12).y

ax = plt.figure().add_subplot(projection="3d")
ax.plot(initial_propagation[0], initial_propagation[1], initial_propagation[2])
ax.plot(final_propagation[0], final_propagation[1], final_propagation[2])
ax.plot(propagation[0], propagation[1], propagation[2])
ax.set_aspect("equal")
plot_moon(ax, mu)

fig = plt.figure()
for ax_index in np.arange(3):
    thing = int("31" + str(ax_index + 1))
    ax = fig.add_subplot(thing)
    ax.plot(control[ax_index])

plt.show()