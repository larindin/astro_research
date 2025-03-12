
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from helper_functions import *


def plot_3sigma(time_vals, estimation_errors, three_sigmas, labels="position", alpha=0.5):

    lr_labels = [r"$\lambda_1$ Error", r"$\lambda_2$ Error", r"$\lambda_3$ Error"]
    lv_labels = [r"$\lambda_4$ Error", r"$\lambda_5$ Error", r"$\lambda_6$ Error"]
    r_labels = [r"$x$ Error [km]", r"$y$ Error [km]", r"$z$ Error [km]"]
    v_labels = [r"$v_x$ Error [m/s]", r"$v_y$ Error [m/s]", r"$v_z$ Error [m/s]"]
    a_labels = [r"$a_x$ Error [mm/s$^2$]", r"$a_y$ Error [mm/s$^2$]", r"$a_z$ Error [mm/s$^2$]"]
    label_dict = {"position":r_labels, "velocity":v_labels, "acceleration":a_labels, "lambdar":lr_labels, "lambdav":lv_labels}
    scaling_dict = {"position":NONDIM_LENGTH, "velocity":NONDIM_LENGTH*1e3/NONDIM_TIME, "acceleration":NONDIM_LENGTH*1e6/NONDIM_TIME**2, "lambdar":1, "lambdav":1}
    
    scaling_factor = scaling_dict[labels]
    ylabels = label_dict[labels]

    num_runs = len(estimation_errors)
    
    plot_time = time_vals * NONDIM_TIME_HR/24
    for state_index in range(3):
        for run_index in range(num_runs):
            estimation_errors[run_index][state_index] *= scaling_factor
            three_sigmas[run_index][state_index] *= scaling_factor

    fig, axes = plt.subplots(3, 1, layout="constrained")
    # fig.set_figheight(6.4)
    
    for state_index in range(3):
        
        ax = axes[state_index]
        if state_index == 2:
            ax.set_xlabel("Time [days]", fontname="Times New Roman")
        ax.set_ylabel(ylabels[state_index], fontname="Times New Roman")
        ax.set_yscale("log")
        ax.tick_params(axis="both", which="major", labelsize=6.5)
        for run_num in range(num_runs):
            ax.step(plot_time, abs(estimation_errors[run_num][state_index]), c="black", alpha=alpha)
            ax.step(plot_time, three_sigmas[run_num][state_index], c="red", ls="--", alpha=alpha)
            ax.grid(True)

def compute_3sigmas(posterior_covariances, state_size):

    three_sigmas = []

    for posterior_covariance_vals in posterior_covariances:
        three_sigma_vals = []
        for state_index in range(state_size):
            three_sigma_vals.append(3*np.sqrt(posterior_covariance_vals[state_index, state_index, :]))
        three_sigmas.append(three_sigma_vals)

    return three_sigmas

def compute_estimation_errors(truth, posterior_estimates, state_size):

    estimation_errors = []

    for posterior_estimate_vals in posterior_estimates:
        estimation_error_vals = []
        for state_index in range(state_size): 
            estimation_error_vals.append(posterior_estimate_vals[state_index, :] - truth[state_index, :] )
        estimation_errors.append(estimation_error_vals)
    
    return estimation_errors

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

def plot_moon(ax, mu):
    ax.scatter(1-mu, 0, 0, c="grey")

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