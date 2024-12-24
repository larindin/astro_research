
import numpy as np
import scipy
import matplotlib.pyplot as plt
from configuration_forback_shooting import *
from CR3BP_pontryagin import *
from plotting import *

initial_costate = np.loadtxt("LT_transfers/solution_30_06_1_27.csv", delimiter=",")
initial_costate = np.loadtxt("LT_transfers/solution_" + str(orbit1) + str(orbit2) + "_e4_25.csv", delimiter=",")[1:]

initial_conditions = np.concatenate((initial_state, initial_costate[0:6]))
final_conditions = np.concatenate((final_state, initial_costate[6:12]))

forward_tspan = np.array([0, tf*patching_time_factor])
backward_tspan = np.array([tf, tf*patching_time_factor])
forward_result = scipy.integrate.solve_ivp(minimum_fuel_ODE, forward_tspan, initial_conditions, args=(mu, umax, truth_rho), atol=1e-12, rtol=1e-12)
forward_propagation = forward_result.y
forward_time = forward_result.t
backward_result = scipy.integrate.solve_ivp(minimum_fuel_ODE, backward_tspan, final_conditions, args=(mu, umax, truth_rho), atol=1e-12, rtol=1e-12)
backward_propagation = backward_result.y
backward_time = backward_result.t

total_time = np.concatenate((forward_time, np.flip(backward_time)))
total_propagation = np.concatenate((forward_propagation, np.flip(backward_propagation, 1)), axis=1)

control = get_min_fuel_control(total_propagation[6:12, :], umax, truth_rho)

moon_vectors = np.vstack((total_propagation[0] - 1 + mu, total_propagation[1], total_propagation[2]))
moon_distances = np.linalg.norm(moon_vectors, axis=0)

initial_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, np.array([0, 4]), initial_state, args=(mu,), atol=1e-12, rtol=1e-12).y
final_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, np.array([0, 4]), final_state, args=(mu,), atol=1e-12, rtol=1e-12).y

ax = plt.figure().add_subplot(projection="3d")
ax.plot(initial_propagation[0], initial_propagation[1], initial_propagation[2])
ax.plot(final_propagation[0], final_propagation[1], final_propagation[2])
ax.plot(forward_propagation[0], forward_propagation[1], forward_propagation[2])
ax.plot(backward_propagation[0], backward_propagation[1], backward_propagation[2])
plot_moon(ax, mu)
ax.set_aspect("equal")

fig = plt.figure()
for ax_index in np.arange(3):
    thing = int("31" + str(ax_index + 1))
    ax = fig.add_subplot(thing)
    ax.plot(total_time, control[ax_index])

ax = plt.figure().add_subplot()
ax.plot(total_time, moon_distances*NONDIM_LENGTH)
ax.hlines((1740, 1840), 0, total_time[-1])

plt.show()