
import numpy as np
import scipy
import matplotlib.pyplot as plt
from CR3BP import *
from configuration_IMM import *
from helper_functions import *
from plotting import *

backprop_time_vals = -np.arange(0, backprop_time, dt)
forprop_time_vals = np.arange(0, final_time, dt)
backprop_tspan = np.array([backprop_time_vals[0], backprop_time_vals[-1]])
forprop_tspan = np.array([forprop_time_vals[0], forprop_time_vals[-1]])
back_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, backprop_tspan, initial_truth[0:6], args=(mu,), t_eval=backprop_time_vals, atol=1e-12, rtol=1e-12).y
back_propagation = np.vstack((back_propagation, np.full(np.shape(back_propagation), 1e-12)))
back_propagation = np.flip(back_propagation, axis=1)
forward_propagation = scipy.integrate.solve_ivp(dynamics_equation, forprop_tspan, initial_truth, args=truth_dynamics_args, t_eval=forprop_time_vals, atol=1e-12, rtol=1e-12).y
truth_vals = np.concatenate((back_propagation[:, :-1], forward_propagation), axis=1)
time_vals = np.concatenate((np.flip(backprop_time_vals[1:]), forprop_time_vals)) + abs(backprop_time_vals[-1])

print(umax*NONDIM_LENGTH*1e6/NONDIM_TIME**2)

final_state = boundary_states[final_orbit_index][0:6]

initial_orbit_prop = scipy.integrate.solve_ivp(CR3BP_DEs, [0, 3], initial_state, args=(mu,), atol=1e-12, rtol=1e-12).y
final_orbit_prop = scipy.integrate.solve_ivp(CR3BP_DEs, [0, 10], final_state, args=(mu,), atol=1e-12, rtol=1e-12).y

ax = plt.figure().add_subplot(projection="3d")
plot_moon(ax, mu, "km")
ax.plot(initial_orbit_prop[0]*NONDIM_LENGTH, initial_orbit_prop[1]*NONDIM_LENGTH, initial_orbit_prop[2]*NONDIM_LENGTH)
ax.plot(final_orbit_prop[0]*NONDIM_LENGTH, final_orbit_prop[1]*NONDIM_LENGTH, final_orbit_prop[2]*NONDIM_LENGTH)
ax.plot(truth_vals[0]*NONDIM_LENGTH, truth_vals[1]*NONDIM_LENGTH, truth_vals[2]*NONDIM_LENGTH)
ax.scatter(truth_vals[0, 0]*NONDIM_LENGTH, truth_vals[1, 0]*NONDIM_LENGTH, truth_vals[2, 0]*NONDIM_LENGTH, marker="^")
ax.scatter(final_state[0]*NONDIM_LENGTH, final_state[1]*NONDIM_LENGTH, final_state[2]*NONDIM_LENGTH, marker="v")
ax.set_aspect("equal")

plt.show()