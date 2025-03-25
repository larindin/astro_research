
import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from helper_functions import *

mpl.rcParams["mathtext.fontset"] = "cm"

def plot_3sigma(time_vals, estimation_errors, three_sigmas, labels="position", alpha=0.5, scale="log", ylim=(None, None)):

    lr_labels = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"]
    lv_labels = [r"$\lambda_4$", r"$\lambda_5$", r"$\lambda_6$"]
    r_labels = [r"$x$ [km]", r"$y$ [km]", r"$z$ [km]"]
    v_labels = [r"$v_x$ [m/s]", r"$v_y$ [m/s]", r"$v_z$ [m/s]"]
    a_labels = [r"$a_x$ [mm/s$^2$]", r"$a_y$ [mm/s$^2$]", r"$a_z$ [mm/s$^2$]"]
    c_labels = a_labels
    label_dict = {"position":r_labels, "velocity":v_labels, "acceleration":a_labels, "lambdar":lr_labels, "lambdav":lv_labels, "control":c_labels}
    scaling_dict = {"position":NONDIM_LENGTH, "velocity":NONDIM_LENGTH*1e3/NONDIM_TIME, "acceleration":NONDIM_LENGTH*1e6/NONDIM_TIME**2, "lambdar":1, "lambdav":1, "control":NONDIM_LENGTH*1e6/NONDIM_TIME**2}
    offset_dict = {"position":0, "velocity":3, "acceleration":6, "lambdar":6, "lambdav":9, "control":0}
    
    scaling_factor = scaling_dict[labels]
    ylabels = label_dict[labels]
    offset = offset_dict[labels]

    num_runs = len(estimation_errors)
    
    plot_time = time_vals * NONDIM_TIME_HR/24

    fig, axes = plt.subplots(3, 1, layout="constrained")
    # fig.set_figheight(6.4)
    
    for ax_index in range(3):
        state_index = ax_index + offset
        
        ax = axes[ax_index]
        if ax_index == 2:
            ax.set_xlabel("Time [days]", fontname="Times New Roman")
        ax.set_ylabel(ylabels[ax_index], fontname="Times New Roman")
        ax.set_yscale(scale)
        ax.tick_params(axis="both", which="major", labelsize=6.5)
        if scale == "log":
            for run_num in range(num_runs):
                ax.step(plot_time, abs(estimation_errors[run_num][state_index])*scaling_factor, c="black", alpha=alpha)
                ax.step(plot_time, three_sigmas[run_num][state_index]*scaling_factor, c="red", ls="-", alpha=alpha)
                ax.grid(True)
        elif scale == "linear":
            for run_num in range(num_runs):
                ax.step(plot_time, estimation_errors[run_num][state_index]*scaling_factor, c="black", alpha=alpha)
                ax.step(plot_time, three_sigmas[run_num][state_index]*scaling_factor, c="red", ls="-", alpha=alpha)
                ax.step(plot_time, -three_sigmas[run_num][state_index]*scaling_factor, c="red", ls="-", alpha=alpha)
                ax.grid(True)
        ax.set_ylim(*ylim)

def compute_3sigmas(posterior_covariances, state_indices: tuple):

    three_sigmas = []

    for posterior_covariance_vals in posterior_covariances:
        three_sigmas.append(3*np.sqrt(np.diagonal(posterior_covariance_vals, axis1=0, axis2=1)[:, state_indices[0]:state_indices[1]].T))

    return three_sigmas

def compute_estimation_errors(truth, posterior_estimates, state_indices: tuple):

    estimation_errors = []

    for posterior_estimate_vals in posterior_estimates:
        estimation_errors.append(posterior_estimate_vals[state_indices[0]:state_indices[1], :] - truth[state_indices[0]:state_indices[1], :])
    
    return estimation_errors

def compute_norm_errors(estimation_errors, state_indices: tuple):

    norm_errors = []

    for estimation_error_vals in estimation_errors:
        norm_errors.append(np.linalg.norm(estimation_error_vals[state_indices[0]:state_indices[1], :], axis=0)[None, :])
    
    return norm_errors

def compute_anees(estimation_errors, posterior_covariances, state_indices: tuple):

    num_runs = len(estimation_errors)
    num_steps = np.size(estimation_errors[0], axis=1)

    anees_vals = np.zeros(num_steps)
    for run_index in range(num_runs):
        for step_index in range(num_steps):
            error = estimation_errors[run_index][state_indices[0]:state_indices[1], step_index]
            covariance = posterior_covariances[run_index][state_indices[0]:state_indices[1], state_indices[0]:state_indices[1], step_index]
            anees = error @ (np.linalg.inv(covariance) @ error)
            anees_vals[step_index] += anees
    
    anees_vals /= num_runs
    
    return anees_vals

def compute_rmse(estimation_errors, state_indices: tuple):

    num_runs = len(estimation_errors)
    num_steps = np.size(estimation_errors[0], axis=1)

    rmse_vals = np.zeros((state_indices[1]- state_indices[0], num_steps))
    for run_index in range(num_runs):
        rmse_vals += estimation_errors[run_index][state_indices[0]:state_indices[1]]**2
    rmse_vals /= num_runs
    rmse_vals = np.sqrt(rmse_vals)

    return rmse_vals

def compute_avg_error(estimation_errors, state_indices: tuple):

    num_runs = len(estimation_errors)
    num_steps = np.size(estimation_errors[0], axis=1)

    avg_error_vals = np.zeros((state_indices[1] - state_indices[0], num_steps))
    for run_index in range(num_runs):
        avg_error_vals += abs(estimation_errors[run_index][state_indices[0]:state_indices[1]])
    avg_error_vals /= num_runs

    return avg_error_vals

def plot_mode_probabilities(time_vals, mode_probability_vals, truth_control, alpha=0.5):
    
    plot_time = time_vals * NONDIM_TIME_HR/24

    ax = plt.figure().add_subplot()
    ax.step(plot_time, mode_probability_vals[0])
    ax.step(plot_time, mode_probability_vals[1])
    ax.step(plot_time, np.linalg.norm(truth_control, axis=0), alpha=alpha)
    ax.hlines((0, 1), plot_time[0], plot_time[-1], ls="--")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Mode Probability")
    ax.legend(["Coasting", "Thrusting"])

def check_divergence(estimation_errors, three_sigmas):

    num_runs = len(estimation_errors)
    state_size = np.size(estimation_errors[0], 0)
    
    divergence_results = np.ones(num_runs)
    for run_index in range(num_runs):
        errors = estimation_errors[run_index]
        sigmas = three_sigmas[run_index]
        for state_index in range(state_size):
            if abs(errors[state_index][-1]) > sigmas[state_index][-1]:
                divergence_results[run_index] = 0
    
    return divergence_results

def plot_moon(ax, mu, units="nd"):
    if units == "nd":
        ax.scatter(1-mu, 0, 0, c="grey")
    elif units == "km":
        ax.scatter((1-mu)*NONDIM_LENGTH, 0, 0, c="grey")

def plot_L2(ax):
    ax.scatter(L2, 0, 0, c="orange", marker="+")

def compute_total_GM_vals(posterior_estimate_vals, posterior_covariance_vals, weights):

    state_size = np.size(posterior_estimate_vals, 0)
    num_timesteps = np.size(posterior_estimate_vals, 1)
    num_kernels = np.size(posterior_estimate_vals, 2)

    total_estimates = np.zeros((state_size, num_timesteps))
    total_covariances = np.zeros((state_size, state_size, num_timesteps))

    for timestep_index in range(num_timesteps):
        for kernel_index in range(num_kernels):
            total_estimates[:, timestep_index] += weights[kernel_index, timestep_index] * posterior_estimate_vals[:, timestep_index, kernel_index]
            reshaped_posterior_estimate = posterior_estimate_vals[:, timestep_index, kernel_index, np.newaxis]
            total_covariances[:, :, timestep_index] += weights[kernel_index, timestep_index]*(posterior_covariance_vals[:, :, timestep_index, kernel_index] + reshaped_posterior_estimate @ reshaped_posterior_estimate.T)

        reshaped_total_estimate = total_estimates[:, timestep_index, np.newaxis]
        total_covariances[:, :, timestep_index] -= reshaped_total_estimate @ reshaped_total_estimate.T
    
    return total_estimates, total_covariances

def plot_GM_heatmap(truth_vals, posterior_estimate_vals, posterior_covariance_vals, weight_vals, timestamp, xbounds=[0.5, 1.5], ybounds=[-0.5, 0.5], state_indices=[0, 1], resolution=101):

    labels = ["x", "y", "z", "vx", "vy", "vz", "l1", "l2", "l3", "l4", "l5", "l6"]

    raw_truth = truth_vals[:, timestamp]
    raw_estimates = posterior_estimate_vals[:, timestamp, :]
    raw_covariances = posterior_covariance_vals[:, :, timestamp, :]
    weights = weight_vals[:, timestamp]
    x_state_index = state_indices[0]
    y_state_index = state_indices[1]

    num_kernels = np.size(raw_estimates, 1)

    truth = np.array([raw_truth[x_state_index], raw_truth[y_state_index]])
    estimates = np.empty((2, num_kernels))
    covariances = np.empty((2, 2, num_kernels))
    for kernel_index in range(num_kernels):
        estimates[:, kernel_index] = np.array([raw_estimates[x_state_index, kernel_index], raw_estimates[y_state_index, kernel_index]])
        covariances[:, :, kernel_index] = np.array([[raw_covariances[x_state_index, x_state_index, kernel_index], raw_covariances[x_state_index, y_state_index, kernel_index]],
                                                    [raw_covariances[y_state_index, x_state_index, kernel_index], raw_covariances[y_state_index, y_state_index, kernel_index]]])

    x_label = labels[x_state_index]
    y_label = labels[y_state_index]

    x_vals = np.linspace(xbounds[0], xbounds[1], resolution)
    y_vals = np.linspace(ybounds[0], ybounds[1], resolution)
    denominators = np.empty((resolution, resolution, num_kernels))
    exponents = np.empty((resolution, resolution, num_kernels))
    z_array = np.zeros((resolution, resolution))

    for x_pixel_index in range(resolution):
        x_val = x_vals[x_pixel_index]
        for y_pixel_index in range(resolution):
            y_val = y_vals[y_pixel_index]
            for kernel_index in range(num_kernels):
                assess_val = np.array([x_val, y_val])
                denominators[x_pixel_index, y_pixel_index, kernel_index], exponents[x_pixel_index, y_pixel_index, kernel_index] = assess_measurement_likelihood(assess_val - estimates[:, kernel_index], covariances[:, :, kernel_index])

    normalized_denominators = denominators/denominators.min()
    normalized_exponents = exponents - exponents.max()

    for x_pixel_index in range(resolution):
        for y_pixel_index in range(resolution):
            z_array[-y_pixel_index-1, x_pixel_index] = np.sum(weights/normalized_denominators[x_pixel_index, y_pixel_index, :] * np.exp(normalized_exponents[x_pixel_index, y_pixel_index, :]))
    
    z_array /= z_array.max()

    ax = plt.figure().add_subplot()
    ax.imshow(z_array, extent=xbounds+ybounds)
    ax.scatter(truth[0], truth[1], s=1, c="r")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def plot_weights(time_vals, weights, alpha=0.25, semilog=True):

    num_particles = np.size(weights, 0)

    if semilog:
        yscale = "log"
    else:
        yscale = "linear"

    ax = plt.figure().add_subplot()
    
    ax.set_title("Particle Weights vs. Time")
    ax.set_xlabel("Time [TU]")
    ax.set_ylabel("Particle Weight")
    ax.set_yscale(yscale)

    for particle_index in range(num_particles):
        ax.scatter(time_vals, weights[particle_index], alpha=alpha, s=4)

def plot_particles_1d(time_vals, y_vals, truth_y_vals=None, alpha=0.25, y_label="Y", equal=False):

    num_particles = np.size(y_vals, 1)

    ax = plt.figure().add_subplot()
    ax.set_xlabel("Time [TU]")
    ax.set_ylabel(y_label)

    ax.plot(time_vals, truth_y_vals, alpha=0.75)

    for particle_index in range(num_particles):
        ax.scatter(time_vals, y_vals[:, particle_index], alpha=alpha, s=4)
    
    if equal:
        ax.set_aspect("equal")

def plot_particles_2d(xy_vals, truth_xy_vals=np.full((2, 1), np.nan), alpha=0.25, xy_labels=["X", "Y"], plt_moon=False, equal=True):

    num_particles = np.size(xy_vals, 2)

    ax = plt.figure().add_subplot()
    ax.set_xlabel(xy_labels[0])
    ax.set_ylabel(xy_labels[1])

    ax.plot(truth_xy_vals[0], truth_xy_vals[1], alpha=0.75)

    for particle_index in range(num_particles):
        ax.scatter(xy_vals[0, :, particle_index], xy_vals[1, :, particle_index], alpha=alpha, s=4)
    
    if equal:
        ax.set_aspect("equal")

def plot_particles_3d(xyz_vals, truth_xyz_vals=np.full((3, 1), np.nan), alpha=0.25, xyz_labels=["X", "Y", "Z"], plt_moon=False, equal=True):

    num_particles = np.size(xyz_vals, 2)

    ax = plt.figure().add_subplot(projection="3d")
    ax.set_xlabel(xyz_labels[0])
    ax.set_ylabel(xyz_labels[1])
    ax.set_zlabel(xyz_labels[2])

    ax.plot(truth_xyz_vals[0], truth_xyz_vals[1], truth_xyz_vals[2], alpha=0.75)

    for particle_index in range(num_particles):
        ax.scatter(xyz_vals[0, :, particle_index], xyz_vals[1, :, particle_index], xyz_vals[2, :, particle_index], alpha=alpha, s=4)
    
    if plt_moon:
        plot_moon(ax, 1.215059e-2)
    
    if equal:
        ax.set_aspect("equal")