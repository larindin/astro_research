
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from CR3BP import *
from configuration_IMM import *
from helper_functions import *
from plotting import *

backprop_time_vals = -np.arange(0, backprop_time, dt)
forprop_time_vals = np.arange(0, final_time, dt)
additional_time_vals = np.arange(forprop_time_vals[-1], forprop_time_vals[-1]+additional_time, dt)
backprop_tspan = np.array([backprop_time_vals[0], backprop_time_vals[-1]])
forprop_tspan = np.array([forprop_time_vals[0], forprop_time_vals[-1]])
additional_tspan = np.array([additional_time_vals[0], additional_time_vals[-1]])
back_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, backprop_tspan, initial_truth[0:6], args=(mu,), t_eval=backprop_time_vals, atol=1e-12, rtol=1e-12).y
back_propagation = np.vstack((back_propagation, np.full(np.shape(back_propagation), 1e-12)))
back_propagation = np.flip(back_propagation, axis=1)
forward_propagation = scipy.integrate.solve_ivp(dynamics_equation, forprop_tspan, initial_truth, args=truth_dynamics_args, t_eval=forprop_time_vals, atol=1e-12, rtol=1e-12).y
additional_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, additional_tspan, forward_propagation[0:6, -1], args=(mu,), t_eval=additional_time_vals, atol=1e-12, rtol=1e-12).y
additional_propagation = np.vstack((additional_propagation, np.full(np.shape(additional_propagation), 1e-12)))
truth_vals = np.concatenate((back_propagation[:, :-1], forward_propagation, additional_propagation[:, 1:]), axis=1)
time_vals = np.concatenate((np.flip(backprop_time_vals[1:]), forprop_time_vals, additional_time_vals[1:])) + abs(backprop_time_vals[-1])


if gap == False:
    # accel_IMM_avg_error_vals = np.load("data/accel_IMM_avg_error.npy")
    # accel_IMM_avg_norm_error_vals = np.load("data/accel_IMM_avg_norm_error.npy")
    # accel_IMM_est_control_vals = np.load("data/accel_IMM_est_control.npy")
    # accel_IMM_est_error_vals = np.load("data/accel_IMM_est_errors.npy")
    # accel_IMM_est_3sigma_vals = np.load("data/accel_IMM_est_3sigmas.npy")
    # accel_IMM_ctrl_errors = np.load("data/accel_IMM_ctrl_errors.npy")
    # accel_IMM_ctrl_3sigmas = np.load("data/accel_IMM_ctrl_3sigmas.npy")
    
    accel_IMM_avg_error_vals = np.load("data/accel_avg_error.npy")
    accel_IMM_avg_norm_error_vals = np.load("data/accel_avg_norm_error.npy")
    accel_IMM_est_control_vals = np.load("data/accel_est_control.npy")
    accel_IMM_est_error_vals = np.load("data/accel_est_errors.npy")
    accel_IMM_est_3sigma_vals = np.load("data/accel_est_3sigmas.npy")
    accel_IMM_ctrl_errors = np.load("data/accel_ctrl_errors.npy")
    accel_IMM_ctrl_3sigmas = np.load("data/accel_ctrl_3sigmas.npy")
    
    OCIMM_avg_error_vals = np.load("data/OCIMM_avg_error.npy")
    OCIMM_avg_norm_error_vals = np.load("data/OCIMM_avg_norm_error.npy")
    OCIMM_est_control_vals = np.load("data/OCIMM_est_control.npy")
    OCIMM_est_error_vals = np.load("data/OCIMM_est_errors.npy")
    OCIMM_est_3sigma_vals = np.load("data/OCIMM_est_errors.npy")
    OCIMM_ctrl_errors = np.load("data/OCIMM_ctrl_errors.npy")
    OCIMM_ctrl_3sigmas = np.load("data/OCIMM_ctrl_3sigmas.npy")
    OCIMM_mode_probabilities = np.load("data/OCIMM_mode_probabilities.npy")
elif gap == True:
    # accel_IMM_avg_error_vals = np.load("data/accel_IMM_avg_error1.npy")
    # accel_IMM_avg_norm_error_vals = np.load("data/accel_IMM_avg_norm_error1.npy")
    # accel_IMM_est_control_vals = np.load("data/accel_IMM_est_control1.npy")
    # accel_IMM_est_error_vals = np.load("data/accel_IMM_est_errors1.npy")
    # accel_IMM_est_3sigma_vals = np.load("data/accel_IMM_est_3sigmas1.npy")
    # accel_IMM_ctrl_errors = np.load("data/accel_IMM_ctrl_errors1.npy")
    # accel_IMM_ctrl_3sigmas = np.load("data/accel_IMM_ctrl_3sigmas1.npy")

    accel_IMM_avg_error_vals = np.load("data/accel_avg_error1.npy")
    accel_IMM_avg_norm_error_vals = np.load("data/accel_avg_norm_error1.npy")
    accel_IMM_est_control_vals = np.load("data/accel_est_control1.npy")
    accel_IMM_est_error_vals = np.load("data/accel_est_errors1.npy")
    accel_IMM_est_3sigma_vals = np.load("data/accel_est_3sigmas1.npy")
    accel_IMM_ctrl_errors = np.load("data/accel_ctrl_errors1.npy")
    accel_IMM_ctrl_3sigmas = np.load("data/accel_ctrl_3sigmas1.npy")

    OCIMM_avg_error_vals = np.load("data/OCIMM_avg_error1.npy")
    OCIMM_avg_norm_error_vals = np.load("data/OCIMM_avg_norm_error1.npy")
    OCIMM_est_control_vals = np.load("data/OCIMM_est_control1.npy")
    OCIMM_est_error_vals = np.load("data/OCIMM_est_errors1.npy")
    OCIMM_est_3sigma_vals = np.load("data/OCIMM_est_3sigmas1.npy")
    OCIMM_ctrl_errors = np.load("data/OCIMM_ctrl_errors1.npy")
    OCIMM_ctrl_3sigmas = np.load("data/OCIMM_ctrl_3sigmas1.npy")
    OCIMM_mode_probabilities = np.load("data/OCIMM_mode_probabilities1.npy")

if vary_scenarios == False:
    sensor_phase = generator.uniform(0, 1)
    sun_phase = generator.uniform(0, 2*np.pi)

    sensor_position_vals = generate_sensor_positions(sensor_dynamics_equation, sensor_initial_conditions, (mu,), time_vals, sensor_phase, sensor_period)

    num_sensors = int(np.size(sensor_position_vals, 0)/3)
    earth_vectors = np.empty((3*num_sensors, len(time_vals)))
    moon_vectors = np.empty((3*num_sensors, len(time_vals)))
    sun_vectors = np.empty((3*num_sensors, len(time_vals)))
    for sensor_index in range(num_sensors):
        sensor_positions = sensor_position_vals[sensor_index*3:(sensor_index + 1)*3, :]
        earth_vectors[sensor_index*3:(sensor_index + 1)*3, :] = generate_earth_vectors(time_vals, sensor_positions)
        moon_vectors[sensor_index*3:(sensor_index + 1)*3, :] = generate_moon_vectors(time_vals, sensor_positions)
        sun_vectors[sensor_index*3:(sensor_index + 1)*3, :] = generate_sun_vectors(time_vals, sun_phase)

    earth_results = np.empty((num_sensors, len(time_vals)))
    moon_results = np.empty((num_sensors, len(time_vals)))
    sun_results = np.empty((num_sensors, len(time_vals)))
    check_results = np.empty((num_sensors, len(time_vals)))
    shadow_results = check_validity(time_vals, truth_vals[0:3, :], sensor_positions, sun_vectors[0:3, :], check_shadow, ())
    for sensor_index in range(num_sensors):
        sensor_positions = sensor_position_vals[sensor_index*3:(sensor_index + 1)*3, :]
        earth_results[sensor_index, :] = check_validity(time_vals, truth_vals[0:3, :], sensor_positions, earth_vectors[sensor_index*3:(sensor_index+1)*3, :], check_exclusion, (earth_exclusion_angle,))
        moon_results[sensor_index, :] = check_validity(time_vals, truth_vals[0:3, :], sensor_positions, moon_vectors[sensor_index*3:(sensor_index+1)*3, :], check_exclusion, (moon_exclusion_angle,))
        sun_results[sensor_index, :] = check_validity(time_vals, truth_vals[0:3, :], sensor_positions, sun_vectors[sensor_index*3:(sensor_index+1)*3, :], check_exclusion, (sun_exclusion_angle,))
        check_results[sensor_index, :] = earth_results[sensor_index, :] * moon_results[sensor_index, :] * sun_results[sensor_index, :] * shadow_results

num_observers = np.sum(check_results, axis=0)
check_results = (np.sum(check_results, axis=0) == 0) #+ (np.sum(check_results, axis=0) == 1)
observation_arc_indices = get_thrusting_arc_indices(check_results[None, :])
num_sensors = np.sum(check_results, axis=0)
print(observation_arc_indices)

truth_control = get_min_fuel_control(truth_vals[6:12, :], umax, truth_rho)*NONDIM_LENGTH*1e6/NONDIM_TIME**2
thrusting_arc_indices = get_thrusting_arc_indices(truth_control)
thrusting_bool = np.linalg.norm(truth_control, axis=0) > 0.01
print(thrusting_arc_indices)
umax_dim = umax*NONDIM_LENGTH*1e6/NONDIM_TIME**2

coasting_truth = truth_vals.copy()
thrusting_truth = truth_vals.copy()
unobserved_truth = truth_vals.copy()
observed_thrusting_truth = truth_vals.copy()
observed_coasting_truth = truth_vals.copy()
unobserved_thrusting_truth = truth_vals.copy()
unobserved_coasting_truth = truth_vals.copy()
coasting_truth[:, thrusting_bool] = np.nan
thrusting_truth[:, thrusting_bool==0] = np.nan
unobserved_truth[:, check_results==0] = np.nan
observed_thrusting_truth[:, ((num_observers!=0) & (thrusting_bool)) == 0] = np.nan
observed_coasting_truth[:, ((num_observers!=0) & (thrusting_bool==0)) == 0] = np.nan
unobserved_thrusting_truth[:, ((num_observers==0) & (thrusting_bool)) == 0] = np.nan
unobserved_coasting_truth[:, ((num_observers==0) & (thrusting_bool==0)) == 0] = np.nan


final_state = boundary_states[final_orbit_index][0:6]

initial_orbit_prop = scipy.integrate.solve_ivp(CR3BP_DEs, [0, 3], initial_state, args=(mu,), atol=1e-12, rtol=1e-12).y
final_orbit_prop = scipy.integrate.solve_ivp(CR3BP_DEs, [0, 10], final_state, args=(mu,), atol=1e-12, rtol=1e-12).y

sensor_position_vals = generate_sensor_positions(sensor_dynamics_equation, sensor_initial_conditions, (mu,), time_vals)

plot_time = time_vals * NONDIM_TIME_HR/24

for run_index in range(num_runs):
    coasting_bool = OCIMM_mode_probabilities[run_index][0] > 0.5
    for index in range(3):
        OCIMM_ctrl_3sigmas[run_index][index, coasting_bool] = 2*umax_dim
        pass
        
OCIMM_est_error_vals[:, 0:3] *= NONDIM_LENGTH
OCIMM_est_3sigma_vals[:, 0:3] *= NONDIM_LENGTH
OCIMM_est_error_vals[:, 3:6] *= NONDIM_LENGTH*1000/NONDIM_TIME
OCIMM_est_3sigma_vals[:, 3:6] *= NONDIM_LENGTH*1000/NONDIM_TIME

ax = plt.figure().add_subplot()
ax.hlines(1, 0, 1, colors="black", label="blank")
ax.hlines(1, 0, 1, colors="red", label="blank2")
myhandle, labels = ax.get_legend_handles_labels()

ax = plt.figure(layout="constrained", figsize=((7.75, 7.75/2))).add_subplot()
for run_index in range(num_runs):
    ax.scatter(plot_time, OCIMM_mode_probabilities[run_index][0], color="black", label="Coasting", alpha=0.15, s=4, edgecolors="none")
    ax.scatter(plot_time, OCIMM_mode_probabilities[run_index][1], color="red", label="Maneuvering", alpha=0.15, s=4, edgecolors="none")
    # ax.plot(plot_time, OCIMM_mode_probabilities[run_index][0], color="black", label="Coasting", alpha=0.15)
    # ax.plot(plot_time, OCIMM_mode_probabilities[run_index][1], color="magenta", label="Maneuvering", alpha=0.15)
for arc in thrusting_arc_indices:
    ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="black", alpha=0.15, ls="--", label="Maneuver")
if gap == True:
    for arc in observation_arc_indices:
        ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="red", alpha=0.15, ls="--", label="Observation Gap")
ax.set_ylabel("Mode Probability", fontname="Times New Roman", fontsize=10)
ax.set_xlabel("Time [days]", fontname="Times New Roman", fontsize=10)
for tick in ax.get_xticklabels():
    tick.set_fontname("Times New Roman")
for tick in ax.get_yticklabels():
    tick.set_fontname("Times New Roman")
ax.tick_params(axis="both", which="major", labelsize=10)
ax.set_xlim(0, 35)
ax.set_ylim(0, 1)
# ax.set_ylim(1e-3, 1)
# ax.set_yscale("log")
if gap==True:
    plt.savefig("figures/mode_probabilities.png", dpi=600, bbox_inches="tight")

fig, axes = plt.subplots(3, 1, layout="constrained", figsize=((7.75, 7.75/2+0.5)))
ax_labels = [r"$x$ [km]", r"$y$ [km]", r"$z$ [km]"]
for ax_index in range(3):
    ax = axes[ax_index]
    for run_index in range(num_runs):
        ax.plot(plot_time, abs(OCIMM_est_error_vals[run_index][ax_index]), label="Estimation Error", alpha=0.1, color="black")
        ax.plot(plot_time, OCIMM_est_3sigma_vals[run_index][ax_index], label=r"$3\sigma$-bounds", alpha=0.1, color="red")
    for arc in thrusting_arc_indices:
        ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="black", alpha=0.15, ls="--", label="Maneuver")
    if gap == True:
        for arc in observation_arc_indices:
            ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="red", alpha=0.15, ls="--", label="Observation Gap")
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
    ax.set_ylabel(ax_labels[ax_index], fontname="Times New Roman")
    ax.grid(axis="y", which="both", linestyle="--")
    ax.set_yscale("log")
    ax.set_xlim(0, 35)
    ax.set_ylim(1e-1, 1e5)
ax.set_xlabel("Time [days]", fontname="Times New Roman")
handles, labels = ax.get_legend_handles_labels()
handles[0] = myhandle[0]
handles[1] = myhandle[1]
handles[2] = handles[-2]
handles[3] = handles[-1]
# fig.legend(handles[0:4], ["Estimation Error", r"$3\sigma$-bounds", "Maneuver", "Observation Gap"], prop={"family":"Times New Roman", "size":"small"}, fancybox=False, loc="upper right", bbox_to_anchor=(1.2, 1), bbox_transform=axes[0].transAxes)
if gap == True:
    plt.savefig("figures/position_3sigmas.png", dpi=600, bbox_inches="tight")

fig, axes = plt.subplots(3, 1, layout="constrained", figsize=((7.75, 7.75/2+0.5)))
ax_labels = [r"$v_x$ [m/s]", r"$v_y$ [m/s]", r"$v_z$ [m/s]"]
for ax_index in range(3):
    ax = axes[ax_index]
    for run_index in range(num_runs):
        ax.plot(plot_time, abs(OCIMM_est_error_vals[run_index][ax_index+3]), label="Estimation Error", alpha=0.1, color="black")
        ax.plot(plot_time, OCIMM_est_3sigma_vals[run_index][ax_index+3], label=r"$3\sigma$-bounds", alpha=0.1, color="red")
    for arc in thrusting_arc_indices:
        ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="black", alpha=0.15, ls="--", label="Maneuver")
    if gap == True:
        for arc in observation_arc_indices:
            ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="red", alpha=0.15, ls="--", label="Observation Gap")
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
    ax.set_ylabel(ax_labels[ax_index], fontname="Times New Roman")
    ax.grid(axis="y", which="both", linestyle="--")
    ax.set_yscale("log")
    ax.set_xlim(0, 35)
    ax.set_ylim(1e-3, 1e3)
ax.set_xlabel("Time [days]", fontname="Times New Roman")
handles, labels = ax.get_legend_handles_labels()
handles[0] = myhandle[0]
handles[1] = myhandle[1]
handles[2] = handles[-2]
handles[3] = handles[-1]
# fig.legend(handles[0:4], ["Estimation Error", r"$3\sigma$-bounds", "Maneuver", "Observation Gap"], prop={"family":"Times New Roman", "size":"small"}, fancybox=False, loc="upper right", bbox_to_anchor=(1.2, 1), bbox_transform=axes[0].transAxes)
if gap == True:
    plt.savefig("figures/velocity_3sigmas.png", dpi=600, bbox_inches="tight")

fig, axes = plt.subplots(3, 1, layout="constrained", figsize=((7.75, 7.75/2+0.5)))
ax_labels = [r"$u_x$ [mm/s$^2$]", r"$u_y$ [mm/s$^2$]", r"$u_z$ [mm/s$^2$]"]
for ax_index in range(3):
    ax = axes[ax_index]
    for run_index in range(num_runs):
        ax.plot(plot_time, abs(OCIMM_ctrl_errors[run_index][ax_index]), label="Estimation Error", alpha=0.1, color="black")
        ax.plot(plot_time, OCIMM_ctrl_3sigmas[run_index][ax_index]*OCIMM_mode_probabilities[run_index][1], label=r"$3\sigma$-bounds", alpha=0.1, color="red")
    for arc in thrusting_arc_indices:
        ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="black", alpha=0.15, ls="--", label="Maneuver")
    if gap == True:
        for arc in observation_arc_indices:
            ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="red", alpha=0.15, ls="--", label="Observation Gap")
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
    ax.set_ylabel(ax_labels[ax_index], fontname="Times New Roman")
    ax.grid(axis="y", which="both", linestyle="--")
    ax.set_yscale("log")
    ax.set_xlim(0, 35)
    ax.set_ylim(1e-5, 1e1)
ax.set_xlabel("Time [days]", fontname="Times New Roman")
handles, labels = ax.get_legend_handles_labels()
handles[0] = myhandle[0]
handles[1] = myhandle[1]
handles[2] = handles[-2]
handles[3] = handles[-1]
# fig.legend(handles[0:4], ["Estimation Error", r"$3\sigma$-bounds", "Maneuver", "Observation Gap"], prop={"family":"Times New Roman", "size":"small"}, fancybox=False, loc="upper right", bbox_to_anchor=(1.2, 1), bbox_transform=axes[0].transAxes)
if gap == True:
    plt.savefig("figures/control_3sigmas.png", dpi=600, bbox_inches="tight")

MAE_fig, axes = plt.subplots(3, 1, layout="constrained", figsize=((7.75, 7.75/2+0.5)))
MAE_fig_labels = [r"$||\boldsymbol{r} - \hat{\boldsymbol{r}}||_2$  [km]", r"$||\boldsymbol{v} - \hat{\boldsymbol{v}}||_2$  [m/s]", r"$||\boldsymbol{a} - \hat{\boldsymbol{a}}||_2$  [mm/s$^2$]"]
for ax_index in range(3):
    ax = axes[ax_index]
    ax.plot(plot_time, OCIMM_avg_norm_error_vals[ax_index], alpha=1, c="red")
    ax.plot(plot_time, accel_IMM_avg_norm_error_vals[ax_index], alpha=1, c="black", ls="--")
    ax.set_ylabel(MAE_fig_labels[ax_index], fontname="Times New Roman")
    if gap == True:
        for arc in observation_arc_indices:
            ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="red", alpha=0.15, ls="--")
    for arc in thrusting_arc_indices:
        ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="black", alpha=0.15, ls="--")
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
    ax.grid(True)
    ax.set_xlim(0, 35)
ax.set_xlabel("Time [days]", fontname="Times New Roman")
MAE_fig.align_ylabels()
# MAE_fig.legend(["OCIMM", "IMM", "Maneuver"], prop={"family":"Times New Roman", "size":"small"}, fancybox=False, loc="upper right", bbox_to_anchor=(1.2, 1), bbox_transform=axes[0].transAxes)
# MAE_fig.legend(["OCIMM", "IMM", "Observation Gap", "Maneuver"], prop={"family":"Times New Roman", "size":"small"}, fancybox=False, loc="upper right", bbox_to_anchor=(1.2, 1), bbox_transform=axes[0].transAxes)
if gap == True:
    plt.savefig("figures/MAE_gap.png", dpi=600, bbox_inches="tight")
elif gap == False:
    plt.savefig("figures/MAE_normal.png", dpi=600, bbox_inches="tight")

ax = plt.figure().add_subplot()
ax.hlines(1, 0, 1, colors="red", label="blank")
myhandle, labels = ax.get_legend_handles_labels()

start_index = 13*24
end_index = 22*24
start_index = 0
end_index = -1

fig, axes = plt.subplots(3, 2, layout="constrained", figsize=((7.75, 7.75/2+0.5)))
control_ax_labels = ["$u_x$ [mm/s$^2$]", "$u_y$ [mm/s$^2$]", "$u_z$ [mm/s$^2$]"]
for ax_index in range(3):
    ax = axes[ax_index, 0]
    ax.plot(plot_time[start_index:end_index], truth_control[ax_index, start_index:end_index], alpha=1, c="black", zorder=3, label="Truth")
    for run_index in range(num_runs):
        ax.step(plot_time[start_index:end_index], OCIMM_est_control_vals[run_index, ax_index, start_index:end_index], alpha=0.1, c="red", label="Estimated")
    for arc in thrusting_arc_indices:
        ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="black", alpha=0.15, ls="--", label="Maneuver")
    for arc in observation_arc_indices:
        ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="red", alpha=0.15, ls="--", label="Observation Gap")
    ax.set_ylabel(control_ax_labels[ax_index], fontname="Times New Roman", fontsize=10)
    ax.grid(axis="y", which="both", linestyle="--")
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
    ax.set_xlim(0, 35)
    ax.set_ylim(-0.4-(0.8*0.05), 0.4+(0.8*0.05))
    ax.set_yticks((-0.4, 0, 0.4))
ax.set_xlabel("Time [days]", fontname="Times New Roman", fontsize=10)
for ax_index in range(3):
    ax = axes[ax_index, 1]
    ax.plot(plot_time[start_index:end_index], truth_control[ax_index, start_index:end_index], alpha=1, c="black", zorder=3, label="Truth")
    for run_index in range(num_runs):
        ax.step(plot_time[start_index:end_index], accel_IMM_est_control_vals[run_index, ax_index, start_index:end_index], alpha=0.1, c="tab:blue", label="Estimated")
    for arc in thrusting_arc_indices:
        ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="black", alpha=0.15, ls="--", label="Maneuver")
    for arc in observation_arc_indices:
        ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="red", alpha=0.15, ls="--", label="Observation Gap")
    # ax.set_ylabel(control_ax_labels[ax_index], fontname="Times New Roman", fontsize=10)
    ax.grid(axis="y", which="both", linestyle="--")
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
    ax.set_xlim(0, 35)
    ax.set_ylim(-0.4-(0.8*0.05), 0.4+(0.8*0.05))
    ax.set_yticks((-0.4, 0, 0.4))
ax.set_xlabel("Time [days]", fontname="Times New Roman", fontsize=10)
handles, labels = ax.get_legend_handles_labels()
handles[1] = myhandle[0]
handles[2] = handles[-2]
handles[3] = handles[-1]
# fig.legend(handles[0:4], ["Truth Control", "Estimated Control", "Maneuver", "Observation Gap"], prop={"family":"Times New Roman", "size":"small"}, fancybox=False, loc="upper right", bbox_to_anchor=(1.2, 1), bbox_transform=axes[0].transAxes)
plt.savefig("figures/control.png", dpi=600, bbox_inches="tight")

ax = plt.figure().add_subplot(projection="3d")
ax.scatter(0, 0, 0, color="grey", s=6, label="Moon")
myhandle, labels = ax.get_legend_handles_labels()

mosaic = [["trajectory", ".", "x"],
          ["trajectory", ".", "y"],
          ["trajectory", ".", "z"],
          ["trajectory", ".", "norm"],
          ["trajectory", ".", "."]]
fig = plt.figure(layout="constrained", figsize=((7.75, 7.75/2+0.5)))
ax_dict = fig.subplot_mosaic(mosaic, per_subplot_kw={"trajectory":{"projection":"3d"}}, width_ratios=(0.7, 0, 0.3), height_ratios=(0.2, 0.2, 0.2, 0.2, 0.2), empty_sentinel=".")
ax = ax_dict["trajectory"]
# ax.scatter((1-mu)*NONDIM_LENGTH, 0, 0, c="grey", s=6, label="Moon", zorder=-1)
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)*1740 + (1 - mu)*NONDIM_LENGTH
y = np.sin(u)*np.sin(v)*1740
z = np.cos(v)*1740
ax.plot_wireframe(x, y, z, color="grey", label="Moon")
ax.plot(initial_orbit_prop[0]*NONDIM_LENGTH, initial_orbit_prop[1]*NONDIM_LENGTH, initial_orbit_prop[2]*NONDIM_LENGTH, label="Initial Orbit", color="c")
ax.scatter(truth_vals[0, 0]*NONDIM_LENGTH, truth_vals[1, 0]*NONDIM_LENGTH, truth_vals[2, 0]*NONDIM_LENGTH, marker="^", label="Initial Point", color="c")
ax.plot(final_orbit_prop[0]*NONDIM_LENGTH, final_orbit_prop[1]*NONDIM_LENGTH, final_orbit_prop[2]*NONDIM_LENGTH, label="Terminal Orbit", color="magenta")
ax.scatter(truth_vals[0, -1]*NONDIM_LENGTH, truth_vals[1, -1]*NONDIM_LENGTH, truth_vals[2, -1]*NONDIM_LENGTH, marker="v", label="Terminal Point", color="magenta")
ax.plot(coasting_truth[0]*NONDIM_LENGTH, coasting_truth[1]*NONDIM_LENGTH, coasting_truth[2]*NONDIM_LENGTH, c="black", label="Coasting Arc")
ax.plot(thrusting_truth[0]*NONDIM_LENGTH, thrusting_truth[1]*NONDIM_LENGTH, thrusting_truth[2]*NONDIM_LENGTH, c="red", label="Thrusting Arc")
ax.plot(sensor_position_vals[0]*NONDIM_LENGTH, sensor_position_vals[1]*NONDIM_LENGTH, sensor_position_vals[2]*NONDIM_LENGTH, c="blue", label="Observer Orbit")
ax.scatter(sensor_position_vals[(0, 3, 6), 0]*NONDIM_LENGTH, sensor_position_vals[(1, 4, 7), 0]*NONDIM_LENGTH, sensor_position_vals[(2, 5, 8), 0]*NONDIM_LENGTH, c="blue", label="Obs. Init. Pos.", alpha=1)
ax.grid(False)
ax.tick_params(axis="both", which="major", labelsize=8)
ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
ax.set_xlabel("$x$ [km]", fontname="Times New Roman", fontsize=8)
ax.set_ylabel("$y$ [km]", fontname="Times New Roman", fontsize=8)
ax.set_zlabel("$z$ [km]", fontname="Times New Roman", fontsize=8)
ax.set_zlim(-5e4, 5e4)
ax.set_zticks((-5e4, 0, 5e4))
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.set_aspect("equal")
ax.get_xaxis().get_offset_text().set_fontname("Times New Roman")
ax.get_yaxis().get_offset_text().set_fontname("Times New Roman")
ax.get_zaxis().get_offset_text().set_fontname("Times New Roman")
ax.get_xaxis().get_offset_text().set_fontsize(8)
ax.get_yaxis().get_offset_text().set_fontsize(8)
ax.get_zaxis().get_offset_text().set_fontsize(8)
ax.get_zaxis().get_offset_text().set_position((0, 0, 1))
for tick in ax.get_xticklabels():
    tick.set_fontname("Times New Roman")
for tick in ax.get_yticklabels():
    tick.set_fontname("Times New Roman")
for tick in ax.get_zticklabels():
    tick.set_fontname("Times New Roman")
handles, labels = ax.get_legend_handles_labels()
handles[0] = myhandle[0]
ax.legend(handles, labels, prop={"family":"Times New Roman", "size":8}, fancybox=False, bbox_to_anchor=(1.1, 1))
ax.view_init(elev=25, azim=-110, roll=0)

ax_index = 0
ax_labels = ["$u_x$ [mm/s$^2$]", "$u_y$ [mm/s$^2$]", "$u_z$ [mm/s$^2$]"]
for ax_name in ["x", "y", "z"]:
    ax = ax_dict[ax_name]
    ax.plot(plot_time, truth_control[ax_index], color="black")
    ax.set_ylabel(ax_labels[ax_index], fontsize=8, fontname="Times New Roman")
    ax.set_xlim(0, 35)
    ax.set_ylim(-0.4-(0.8*0.05), 0.4+(0.8*0.05))
    ax.set_yticks((-0.4, 0, 0.4))
    ax.hlines((-0.4, 0.4), 0, 35, color="black", linestyle="--", linewidth=0.75)
    # ax.hlines(0, 0, 35, color="black", linestyle="--", linewidth=0.75, alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
    ax_index += 1

ax = ax_dict["norm"]
ax.plot(plot_time, np.linalg.norm(truth_control, axis=0), color="black")
ax.set_xlabel("Time [days]", fontname="Times New Roman", fontsize=8)
ax.set_ylabel(r"$||\boldsymbol{u}||_2$ [mm/s$^2$]", fontname="Times New Roman", fontsize=8)
ax.set_xlim(0, 35)
ax.set_yticks((0, 0.4))
ax.tick_params(axis="both", which="major", labelsize=8)
for tick in ax.get_xticklabels():
    tick.set_fontname("Times New Roman")
for tick in ax.get_yticklabels():
    tick.set_fontname("Times New Roman")
plt.grid(which="both", axis="y", ls="--", color="black")
fig.align_ylabels(axs=[ax_dict["x"], ax_dict["y"], ax_dict["z"], ax_dict["norm"]])

plt.savefig("figures/truth_trajectory.png", dpi=600, bbox_inches="tight")

close_bool = np.linalg.norm([(1 - mu) - final_orbit_prop[0], -final_orbit_prop[1], -final_orbit_prop[2]], axis=0) < 40000/NONDIM_LENGTH

fig = plt.figure(figsize=(7.75, 7.75/2))
ax = fig.add_subplot(121, projection="3d")
ax.plot_wireframe(x, y, z, color="grey", label="Moon")
ax.scatter(L1*NONDIM_LENGTH, 0, 0, color="black", marker="+", label="L1")
ax.plot(initial_orbit_prop[0]*NONDIM_LENGTH, initial_orbit_prop[1]*NONDIM_LENGTH, initial_orbit_prop[2]*NONDIM_LENGTH, label="L1 Halo", color="c")
ax.plot(final_orbit_prop[0]*NONDIM_LENGTH, final_orbit_prop[1]*NONDIM_LENGTH, final_orbit_prop[2]*NONDIM_LENGTH, label="L1 Lyapunov", color="magenta")
ax.set_aspect("equal")
ax.grid(True, linewidth=0.5, alpha=0.5)
ax.tick_params(axis="both", which="major", labelsize=8)
ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
ax.set_xlabel("$x$ [km]", fontname="Times New Roman", fontsize=8)
ax.set_ylabel("$y$ [km]", fontname="Times New Roman", fontsize=8)
ax.set_zlabel("$z$ [km]", fontname="Times New Roman", fontsize=8)
ax.set_zlim(-5e4, 5e4)
ax.set_zticks((-5e4, 0, 5e4))
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.set_aspect("equal")
ax.get_xaxis().get_offset_text().set_fontname("Times New Roman")
ax.get_yaxis().get_offset_text().set_fontname("Times New Roman")
ax.get_zaxis().get_offset_text().set_fontname("Times New Roman")
ax.get_xaxis().get_offset_text().set_fontsize(8)
ax.get_yaxis().get_offset_text().set_fontsize(8)
ax.get_zaxis().get_offset_text().set_fontsize(8)
ax.get_zaxis().get_offset_text().set_position((0, 0, 1))
for tick in ax.get_xticklabels():
    tick.set_fontname("Times New Roman")
for tick in ax.get_yticklabels():
    tick.set_fontname("Times New Roman")
for tick in ax.get_zticklabels():
    tick.set_fontname("Times New Roman")
ax.view_init(elev=25, azim=-110, roll=0)

ax = fig.add_subplot(122, projection="3d")
ax.plot_wireframe(x, y, z, color="grey", label="Moon")
ax.scatter(L1*NONDIM_LENGTH, 0, 0, color="black", marker="+", label="L1")
ax.plot(initial_orbit_prop[0]*NONDIM_LENGTH, initial_orbit_prop[1]*NONDIM_LENGTH, initial_orbit_prop[2]*NONDIM_LENGTH, label="L1 Halo", color="c")
ax.plot(final_orbit_prop[0, close_bool]*NONDIM_LENGTH, final_orbit_prop[1, close_bool]*NONDIM_LENGTH, final_orbit_prop[2, close_bool]*NONDIM_LENGTH, label="L1 Lyapunov", color="magenta")
ax.set_xlim(300000, 400000)
ax.set_ylim(-20000, 20000)
ax.set_aspect("equal")
ax.grid(True, linewidth=0.5, alpha=0.5)
ax.tick_params(axis="both", which="major", labelsize=8)
ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
ax.set_xlabel("$x$ [km]", fontname="Times New Roman", fontsize=8)
ax.set_ylabel("$y$ [km]", fontname="Times New Roman", fontsize=8)
ax.set_zlabel("$z$ [km]", fontname="Times New Roman", fontsize=8)
ax.set_zlim(-2e4, 2e4)
ax.set_zticks((-2e4, 0, 2e4))
ax.set_yticks((-2e4, -1e4, 0, 1e4, 2e4))
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.set_aspect("equal")
ax.get_xaxis().get_offset_text().set_fontname("Times New Roman")
ax.get_yaxis().get_offset_text().set_fontname("Times New Roman")
ax.get_zaxis().get_offset_text().set_fontname("Times New Roman")
ax.get_xaxis().get_offset_text().set_fontsize(8)
ax.get_yaxis().get_offset_text().set_fontsize(8)
ax.get_zaxis().get_offset_text().set_fontsize(8)
ax.get_zaxis().get_offset_text().set_position((0, 0, 1))
for tick in ax.get_xticklabels():
    tick.set_fontname("Times New Roman")
for tick in ax.get_yticklabels():
    tick.set_fontname("Times New Roman")
for tick in ax.get_zticklabels():
    tick.set_fontname("Times New Roman")
handles, labels = ax.get_legend_handles_labels()
handles[0] = myhandle[0]
ax.legend(handles, labels, prop={"family":"Times New Roman", "size":8}, fancybox=False, bbox_to_anchor=(0.1, 1))
ax.view_init(elev=23, azim=-140, roll=0)

plt.savefig("figures/orbits.png", dpi=600, bbox_inches="tight")

coasting_truth[:, check_results==1] = np.nan
thrusting_truth[:, check_results==1] = np.nan

ax = plt.figure(layout="constrained", figsize=(7.75, 7.75/2)).add_subplot(projection="3d")
ax.plot_wireframe(x, y, z, color="grey", label="Moon")
# ax.plot(coasting_truth[0]*NONDIM_LENGTH, coasting_truth[1]*NONDIM_LENGTH, coasting_truth[2]*NONDIM_LENGTH, c="black", label="Coasting Arc")
# ax.plot(thrusting_truth[0]*NONDIM_LENGTH, thrusting_truth[1]*NONDIM_LENGTH, thrusting_truth[2]*NONDIM_LENGTH, c="red", label="Thrusting Arc")
# ax.plot(unobserved_truth[0]*NONDIM_LENGTH, unobserved_truth[1]*NONDIM_LENGTH, unobserved_truth[2]*NONDIM_LENGTH, c="red", label="Observation Gap", ls=(0, (1, 1.2)))
ax.plot(observed_coasting_truth[0]*NONDIM_LENGTH, observed_coasting_truth[1]*NONDIM_LENGTH, observed_coasting_truth[2]*NONDIM_LENGTH, c="black", label="Coasting Arc")
ax.plot(observed_thrusting_truth[0]*NONDIM_LENGTH, observed_thrusting_truth[1]*NONDIM_LENGTH, observed_thrusting_truth[2]*NONDIM_LENGTH, c="red", label="Thrusting Arc")
ax.plot(unobserved_coasting_truth[0]*NONDIM_LENGTH, unobserved_coasting_truth[1]*NONDIM_LENGTH, unobserved_coasting_truth[2]*NONDIM_LENGTH, c="black", label="Observation Gap", ls=(0, (1, 1.2)))
ax.plot(unobserved_thrusting_truth[0]*NONDIM_LENGTH, unobserved_thrusting_truth[1]*NONDIM_LENGTH, unobserved_thrusting_truth[2]*NONDIM_LENGTH, c="red", ls=(0, (1, 1.2)))
ax.scatter(truth_vals[0, 0]*NONDIM_LENGTH, truth_vals[1, 0]*NONDIM_LENGTH, truth_vals[2, 0]*NONDIM_LENGTH, marker="^", label="Initial Point", color="c")
ax.scatter(truth_vals[0, -1]*NONDIM_LENGTH, truth_vals[1, -1]*NONDIM_LENGTH, truth_vals[2, -1]*NONDIM_LENGTH, marker="v", label="Terminal Point", color="magenta")
ax.plot(sensor_position_vals[0]*NONDIM_LENGTH, sensor_position_vals[1]*NONDIM_LENGTH, sensor_position_vals[2]*NONDIM_LENGTH, c="blue", label="Observer Orbit")
ax.scatter(sensor_position_vals[(0, 3, 6), 221]*NONDIM_LENGTH, sensor_position_vals[(1, 4, 7), 221]*NONDIM_LENGTH, sensor_position_vals[(2, 5, 8), 221]*NONDIM_LENGTH, c="blue", label="Obs. Positions", alpha=1)
ax.grid(False)
ax.tick_params(axis="both", which="major", labelsize=8)
ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
ax.set_xlabel("$x$ [km]", fontname="Times New Roman", fontsize=8)
ax.set_ylabel("$y$ [km]", fontname="Times New Roman", fontsize=8)
ax.set_zlabel("$z$ [km]", fontname="Times New Roman", fontsize=8)
ax.set_zlim(-5e4, 5e4)
ax.set_zticks((-5e4, 0, 5e4))
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.set_aspect("equal")
ax.get_xaxis().get_offset_text().set_fontname("Times New Roman")
ax.get_yaxis().get_offset_text().set_fontname("Times New Roman")
ax.get_zaxis().get_offset_text().set_fontname("Times New Roman")
ax.get_xaxis().get_offset_text().set_fontsize(8)
ax.get_yaxis().get_offset_text().set_fontsize(8)
ax.get_zaxis().get_offset_text().set_fontsize(8)
ax.get_zaxis().get_offset_text().set_position((0, 0, 1))
for tick in ax.get_xticklabels():
    tick.set_fontname("Times New Roman")
for tick in ax.get_yticklabels():
    tick.set_fontname("Times New Roman")
for tick in ax.get_zticklabels():
    tick.set_fontname("Times New Roman")
handles, labels = ax.get_legend_handles_labels()
handles[0] = myhandle[0]
ax.legend(handles, labels, prop={"family":"Times New Roman", "size":8}, fancybox=False, bbox_to_anchor=(1.2, 1))
ax.view_init(elev=25, azim=-110, roll=0)

plt.savefig("figures/unobserved_trajectory.png", dpi=600, bbox_inches="tight")

thrusting_truth_control = truth_control.copy()
thrusting_truth_control[:, thrusting_bool==0] = np.nan

ax = plt.figure(layout="constrained", figsize=(7.75, 7.75/2)).add_subplot()
circle1 = plt.Circle(((1-mu)*NONDIM_LENGTH, 0), 1740, color="grey")
ax.add_patch(circle1)
ax.plot(coasting_truth[0]*NONDIM_LENGTH, coasting_truth[1]*NONDIM_LENGTH, c="black", label="Coasting Arc")
ax.plot(thrusting_truth[0]*NONDIM_LENGTH, thrusting_truth[1]*NONDIM_LENGTH, c="red", label="Thrusting Arc")
ax.quiver(thrusting_truth[0, 0:-1:2]*NONDIM_LENGTH, thrusting_truth[1, 0:-1:2]*NONDIM_LENGTH, thrusting_truth_control[0, 0:-1:2], thrusting_truth_control[1, 0:-1:2], color="black", scale=3, zorder=5, width=0.004, headwidth=4)
ax.set_xlim((1 - mu - 0.07)*NONDIM_LENGTH, (1 - mu + 0.07)*NONDIM_LENGTH)
ax.set_ylim(-0.07*NONDIM_LENGTH, 0.07*NONDIM_LENGTH)
ax.set_aspect("equal")
ax.tick_params(axis="both", which="major", labelsize=10)
ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
ax.set_xlabel("$x$ [km]", fontname="Times New Roman", fontsize=10)
ax.set_ylabel("$y$ [km]", fontname="Times New Roman", fontsize=10)
ax.get_xaxis().get_offset_text().set_fontname("Times New Roman")
ax.get_yaxis().get_offset_text().set_fontname("Times New Roman")
ax.get_xaxis().get_offset_text().set_fontsize(10)
ax.get_yaxis().get_offset_text().set_fontsize(10)
for tick in ax.get_xticklabels():
    tick.set_fontname("Times New Roman")
for tick in ax.get_yticklabels():
    tick.set_fontname("Times New Roman")

plt.savefig("figures/closeup.png", dpi=600, bbox_inches="tight")

# ax = plt.figure(layout="constrained").add_subplot()
# ax.plot(plot_time, OCIMM_avg_norm_error_vals[0], alpha=1, c="red")
# ax.plot(plot_time, accel_IMM_avg_norm_error_vals[0], alpha=1, ls="--", c="black")
# ax.axvspan(plot_time[observation_arc_indices[0][0]], plot_time[observation_arc_indices[0][1]], color="red", alpha=0.15, ls="--")
# for arc in thrusting_arc_indices:
#     ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="black", alpha=0.15, ls="--")
# ax.set_ylabel(r"$||\boldsymbol{r} - \hat{\boldsymbol{r}}||_2$  [km]", fontname="Times New Roman")
# ax.set_xlabel("Time [days]", fontname="Times New Roman")
# ax.grid(True)
# ax.legend(["OCIMM", "IMM", "Observation Gap", "Maneuver"], prop={"family":"Times New Roman", "size":"small"}, fancybox=False)
# for tick in ax.get_xticklabels():
#     tick.set_fontname("Times New Roman")
# for tick in ax.get_yticklabels():
#     tick.set_fontname("Times New Roman")

# ax = plt.figure(layout="constrained").add_subplot()
# ax.plot(plot_time, OCIMM_avg_norm_error_vals[1], alpha=1, c="red")
# ax.plot(plot_time, accel_IMM_avg_norm_error_vals[1], alpha=1, ls="--", c="black")
# ax.axvspan(plot_time[observation_arc_indices[0][0]], plot_time[observation_arc_indices[0][1]], color="red", alpha=0.15, ls="--")
# for arc in thrusting_arc_indices:
#     ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="black", alpha=0.15, ls="--")
# ax.set_ylabel(r"$||\boldsymbol{v} - \hat{\boldsymbol{v}}||_2$  [m/s]", fontname="Times New Roman")
# ax.set_xlabel("Time [days]", fontname="Times New Roman")
# ax.grid(True)
# ax.legend(["OCIMM", "IMM", "Observation Gap", "Maneuver"], prop={"family":"Times New Roman", "size":"small"}, fancybox=False)
# for tick in ax.get_xticklabels():
#     tick.set_fontname("Times New Roman")
# for tick in ax.get_yticklabels():
#     tick.set_fontname("Times New Roman")

# ax = plt.figure(layout="constrained").add_subplot()
# ax.plot(plot_time, OCIMM_avg_norm_error_vals[2], alpha=1, c="red")
# ax.plot(plot_time, accel_IMM_avg_norm_error_vals[2], alpha=1, ls="--", c="black")
# ax.axvspan(plot_time[observation_arc_indices[0][0]], plot_time[observation_arc_indices[0][1]], color="red", alpha=0.15, ls="--")
# for arc in thrusting_arc_indices:
#     ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="black", alpha=0.15, ls="--")
# ax.set_ylabel(r"$||\boldsymbol{a} - \hat{\boldsymbol{a}}||_2$  [mm/s$^2$]", fontname="Times New Roman")
# ax.set_xlabel("Time [days]", fontname="Times New Roman")
# ax.grid(True)
# ax.legend(["OCIMM", "IMM", "Observation Gap", "Maneuver"], prop={"family":"Times New Roman", "size":"small"}, fancybox=False)
# for tick in ax.get_xticklabels():
#     tick.set_fontname("Times New Roman")
# for tick in ax.get_yticklabels():
#     tick.set_fontname("Times New Roman")

# rmse_r_fig = plt.figure()
# rmse_ax_labels = ["$x$", "$y$", "$z$"]
# ylims = []
# for ax_index in range(3):
#     thing = int("31" + str(ax_index + 1))
#     ax = rmse_r_fig.add_subplot(thing)
#     ax.plot(plot_time, accel_IMM_avg_error_vals[ax_index], alpha=1, ls="--", c="black")
#     ax.plot(plot_time, OCIMM_avg_error_vals[ax_index], alpha=1, c="red")
#     ax.set_ylabel(rmse_ax_labels[ax_index])
#     for arc in thrusting_arc_indices:
#         ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="black", alpha=0.15, ls="--")
#     ylims.append(ax.get_ylim())
# upper_bound = max([ylims[ax_index][1] for ax_index in range(3)])
# lower_bound = min([ylims[ax_index][0] for ax_index in range(3)])
# for ax_index in range(3):
#     ax.set_ylim(lower_bound, upper_bound)
# ax.set_xlabel("Time [days]")
# ax.legend(["IMM", "OCIMM"])

# rmse_v_fig = plt.figure()
# rmse_ax_labels = ["$v_x$", "$v_y$", "$v_z$"]
# ylims=[]
# for ax_index in range(3):
#     thing = int("31" + str(ax_index + 1))
#     ax = rmse_v_fig.add_subplot(thing)
#     ax.plot(plot_time, accel_IMM_avg_error_vals[ax_index+3], alpha=1, ls="--", c="black")
#     ax.plot(plot_time, OCIMM_avg_error_vals[ax_index+3], alpha=1, c="red")
#     ax.set_ylabel(rmse_ax_labels[ax_index])
#     for arc in thrusting_arc_indices:
#         ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="red", alpha=0.15, ls="--")
#     ylims.append(ax.get_ylim())
# upper_bound = max([ylims[ax_index][1] for ax_index in range(3)])
# lower_bound = min([ylims[ax_index][0] for ax_index in range(3)])
# for ax_index in range(3):
#     ax.set_ylim(lower_bound, upper_bound)
# ax.set_xlabel("Time [days]")

# rmse_a_fig = plt.figure()
# rmse_ax_labels = ["$a_x$", "$a_y$", "$a_z$"]
# for ax_index in range(3):
#     thing = int("31" + str(ax_index + 1))
#     ax = rmse_a_fig.add_subplot(thing)
#     ax.plot(plot_time, accel_IMM_avg_error_vals[ax_index+6], alpha=1, ls="--", c="black")
#     ax.plot(plot_time, OCIMM_avg_error_vals[ax_index+6], alpha=1, c="red")
#     ax.set_ylabel(rmse_ax_labels[ax_index])
#     ax.set_ylim(0, umax_dim)
#     for arc in thrusting_arc_indices:
#         ax.axvspan(plot_time[arc[0]], plot_time[arc[1]], color="red", alpha=0.15, ls="--")
# ax.set_xlabel("Time [days]")

# fig, axes = plt.subplots(3, 1, layout="constrained") 
# control_ax_labels = ["$u_x$ [mm/s$^2$]", "$u_y$ [mm/s$^2$]", "$u_z$ [mm/s$^2$]"]
# for ax_index in range(3):
#     ax = axes[ax_index]
#     ax.plot(plot_time, truth_control[ax_index], c="black")
#     ax.set_ylabel(control_ax_labels[ax_index], fontname="Times New Roman")
#     ax.set_ylim(-umax_dim, umax_dim)
#     ax.grid(True)
# ax.set_xlabel("Time [days]", fontname="Times New Roman")