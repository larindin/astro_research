
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

check_results = np.ones(len(time_vals))
check_results[350:] = 0
check_results[450:] = 1
check_results = check_results == 0

observation_arc_indices = get_thrusting_arc_indices(check_results[None, :])
print(observation_arc_indices)

truth_control = get_min_fuel_control(truth_vals[6:12, :], umax, truth_rho)*NONDIM_LENGTH*1e6/NONDIM_TIME**2
thrusting_arc_indices = get_thrusting_arc_indices(truth_control)
print(thrusting_arc_indices)
umax_dim = umax*NONDIM_LENGTH*1e6/NONDIM_TIME**2

final_state = boundary_states[final_orbit_index][0:6]

initial_orbit_prop = scipy.integrate.solve_ivp(CR3BP_DEs, [0, 3], initial_state, args=(mu,), atol=1e-12, rtol=1e-12).y
final_orbit_prop = scipy.integrate.solve_ivp(CR3BP_DEs, [0, 10], final_state, args=(mu,), atol=1e-12, rtol=1e-12).y

# accel_IMM_avg_error_vals = np.load("data/accel_IMM_avg_error.npy")
# OCIMM_avg_error_vals = np.load("data/OCIMM_avg_error.npy")
# accel_IMM_avg_norm_error_vals = np.load("data/accel_IMM_avg_norm_error.npy")
# OCIMM_avg_norm_error_vals = np.load("data/OCIMM_avg_norm_error.npy")

accel_IMM_avg_error_vals = np.load("data/accel_IMM_avg_error1.npy")
OCIMM_avg_error_vals = np.load("data/OCIMM_avg_error1.npy")
accel_IMM_avg_norm_error_vals = np.load("data/accel_IMM_avg_norm_error1.npy")
OCIMM_avg_norm_error_vals = np.load("data/OCIMM_avg_norm_error1.npy")

plot_time = time_vals * NONDIM_TIME_HR/24

ax = plt.figure(layout="constrained").add_subplot()
ax.plot(plot_time, OCIMM_avg_norm_error_vals[0], alpha=1, c="red")
ax.plot(plot_time, accel_IMM_avg_norm_error_vals[0], alpha=1, ls="--", c="black")
for arc in thrusting_arc_indices:
    ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="black", alpha=0.15, ls="--")
ax.axvspan(plot_time[observation_arc_indices[0][0]], plot_time[observation_arc_indices[0][1]], color="red", alpha=0.15, ls="--")
ax.set_ylabel(r"$||\boldsymbol{r} - \hat{\boldsymbol{r}}||_2$  [km]", fontname="Times New Roman")
ax.set_xlabel("Time [days]", fontname="Times New Roman")
ax.grid(True)
ax.legend(["IMM", "OCIMM", "Maneuver", "Observation Gap"], prop={"family":"Times New Roman", "size":"small"})

ax = plt.figure(layout="constrained").add_subplot()
ax.plot(plot_time, OCIMM_avg_norm_error_vals[1], alpha=1, c="red")
ax.plot(plot_time, accel_IMM_avg_norm_error_vals[1], alpha=1, ls="--", c="black")
for arc in thrusting_arc_indices:
    ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="black", alpha=0.15, ls="--")
ax.axvspan(plot_time[observation_arc_indices[0][0]], plot_time[observation_arc_indices[0][1]], color="red", alpha=0.15, ls="--")
ax.set_ylabel(r"$||\boldsymbol{v} - \hat{\boldsymbol{v}}||_2$  [m/s]", fontname="Times New Roman")
ax.set_xlabel("Time [days]", fontname="Times New Roman")
ax.grid(True)
ax.legend(["IMM", "OCIMM", "Maneuver", "Observation Gap"], prop={"family":"Times New Roman", "size":"small"})

ax = plt.figure(layout="constrained").add_subplot()
ax.plot(plot_time, OCIMM_avg_norm_error_vals[2], alpha=1, c="red")
ax.plot(plot_time, accel_IMM_avg_norm_error_vals[2], alpha=1, ls="--", c="black")
for arc in thrusting_arc_indices:
    ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="black", alpha=0.15, ls="--")
ax.axvspan(plot_time[observation_arc_indices[0][0]], plot_time[observation_arc_indices[0][1]], color="red", alpha=0.15, ls="--")
ax.set_ylabel(r"$||\boldsymbol{a} - \hat{\boldsymbol{a}}||_2$  [mm/s$^2$]", fontname="Times New Roman")
ax.set_xlabel("Time [days]", fontname="Times New Roman")
ax.grid(True)
ax.legend(["IMM", "OCIMM", "Maneuver", "Observation Gap"], prop={"family":"Times New Roman", "size":"small"})

rmse_r_fig = plt.figure()
rmse_ax_labels = ["$x$", "$y$", "$z$"]
ylims = []
for ax_index in range(3):
    thing = int("31" + str(ax_index + 1))
    ax = rmse_r_fig.add_subplot(thing)
    ax.plot(plot_time, accel_IMM_avg_error_vals[ax_index], alpha=1, ls="--", c="black")
    ax.plot(plot_time, OCIMM_avg_error_vals[ax_index], alpha=1, c="red")
    ax.set_ylabel(rmse_ax_labels[ax_index])
    for arc in thrusting_arc_indices:
        ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="black", alpha=0.15, ls="--")
    ylims.append(ax.get_ylim())
upper_bound = max([ylims[ax_index][1] for ax_index in range(3)])
lower_bound = min([ylims[ax_index][0] for ax_index in range(3)])
for ax_index in range(3):
    ax.set_ylim(lower_bound, upper_bound)
ax.set_xlabel("Time [days]")
ax.legend(["IMM", "OCIMM"])

rmse_v_fig = plt.figure()
rmse_ax_labels = ["$v_x$", "$v_y$", "$v_z$"]
ylims=[]
for ax_index in range(3):
    thing = int("31" + str(ax_index + 1))
    ax = rmse_v_fig.add_subplot(thing)
    ax.plot(plot_time, accel_IMM_avg_error_vals[ax_index+3], alpha=1, ls="--", c="black")
    ax.plot(plot_time, OCIMM_avg_error_vals[ax_index+3], alpha=1, c="red")
    ax.set_ylabel(rmse_ax_labels[ax_index])
    for arc in thrusting_arc_indices:
        ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="red", alpha=0.15, ls="--")
    ylims.append(ax.get_ylim())
upper_bound = max([ylims[ax_index][1] for ax_index in range(3)])
lower_bound = min([ylims[ax_index][0] for ax_index in range(3)])
for ax_index in range(3):
    ax.set_ylim(lower_bound, upper_bound)
ax.set_xlabel("Time [days]")

rmse_a_fig = plt.figure()
rmse_ax_labels = ["$a_x$", "$a_y$", "$a_z$"]
for ax_index in range(3):
    thing = int("31" + str(ax_index + 1))
    ax = rmse_a_fig.add_subplot(thing)
    ax.plot(plot_time, accel_IMM_avg_error_vals[ax_index+6], alpha=1, ls="--", c="black")
    ax.plot(plot_time, OCIMM_avg_error_vals[ax_index+6], alpha=1, c="red")
    ax.set_ylabel(rmse_ax_labels[ax_index])
    ax.set_ylim(0, umax_dim)
    for arc in thrusting_arc_indices:
        ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="red", alpha=0.15, ls="--")
ax.set_xlabel("Time [days]")

fig, axes = plt.subplots(3, 1, layout="constrained") 
control_ax_labels = ["$u_x$ [mm/s$^2$]", "$u_y$ [mm/s$^2$]", "$u_z$ [mm/s$^2$]"]
for ax_index in range(3):
    ax = axes[ax_index]
    ax.plot(plot_time, truth_control[ax_index], c="black")
    ax.set_ylabel(control_ax_labels[ax_index], fontname="Times New Roman")
    ax.set_ylim(-umax_dim, umax_dim)
    ax.grid(True)
ax.set_xlabel("Time [days]", fontname="Times New Roman")

ax = plt.figure(layout="constrained").add_subplot(projection="3d")
plot_moon(ax, mu, "km")
ax.plot(initial_orbit_prop[0]*NONDIM_LENGTH, initial_orbit_prop[1]*NONDIM_LENGTH, initial_orbit_prop[2]*NONDIM_LENGTH)
ax.plot(final_orbit_prop[0]*NONDIM_LENGTH, final_orbit_prop[1]*NONDIM_LENGTH, final_orbit_prop[2]*NONDIM_LENGTH)
ax.plot(truth_vals[0]*NONDIM_LENGTH, truth_vals[1]*NONDIM_LENGTH, truth_vals[2]*NONDIM_LENGTH)
ax.scatter(truth_vals[0, 0]*NONDIM_LENGTH, truth_vals[1, 0]*NONDIM_LENGTH, truth_vals[2, 0]*NONDIM_LENGTH, marker="^")
ax.scatter(final_state[0]*NONDIM_LENGTH, final_state[1]*NONDIM_LENGTH, final_state[2]*NONDIM_LENGTH, marker="v")
ax.set_aspect("equal")

plt.show()