
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from helper_functions import *


def plot_3sigma(time_vals, estimation_errors, three_sigmas, state_size, bounds=(None, None), alpha=0.5):

    ylabels = [r"X Position Estimation Error", r"Y Position Estimation Error", r"Z Position Estimation Error", r"X Velocity Estimation Error", r"Y Velocity Estimation Error", r"Z Velocity Estimation Error"]
    fig, axes = plt.subplots(2, 3, layout="constrained")
    fig.set_figheight(6.4)
    
    axes_rowref = [0, 0, 0, 1, 1, 1]
    axes_colref = [0, 1, 2, 0, 1, 2]
    for state_index in range(state_size):
        
        ax = axes[axes_rowref[state_index], axes_colref[state_index]]
        ax.set_ylim(bounds[0], bounds[1], auto=bounds[0]==None)
        if axes_rowref[state_index] == 1:
            ax.set_xlabel("Time, s", fontname="Times New Roman")
        ax.set_ylabel(ylabels[state_index], fontname="Times New Roman")
        ax.tick_params(axis="both", which="major", labelsize=6.5)
        for run_num in range(len(estimation_errors)):
            ax.step(time_vals, estimation_errors[run_num][state_index], color="blue", alpha=alpha)
            ax.step(time_vals, three_sigmas[run_num][state_index], color="red", alpha=alpha)
            ax.step(time_vals, -three_sigmas[run_num][state_index], color="red", alpha=alpha)
            ax.grid(True)

def plot_3sigma_costate(time_vals, estimation_errors, three_sigmas, state_size, bounds=(None, None), alpha=0.5):

    ylabels = [r"$\lambda_1$ Estimation Error", r"$\lambda_2$ Estimation Error", r"$\lambda_3$ Estimation Error", r"$\lambda_4$ Estimation Error", r"$\lambda_5$ Estimation Error", r"$\lambda_6$ Estimation Error"]
    fig, axes = plt.subplots(2, 3, layout="constrained")
    fig.set_figheight(6.4)
    
    axes_rowref = [0, 0, 0, 1, 1, 1]
    axes_colref = [0, 1, 2, 0, 1, 2]
    for state_index in range(state_size):
        
        ax = axes[axes_rowref[state_index], axes_colref[state_index]]
        ax.set_ylim(bounds[0], bounds[1], auto=bounds[0]==None)
        if axes_rowref[state_index] == 1:
            ax.set_xlabel("Time, s", fontname="Times New Roman")
        ax.set_ylabel(ylabels[state_index], fontname="Times New Roman")
        ax.tick_params(axis="both", which="major", labelsize=6.5)
        for run_num in range(len(estimation_errors)):
            ax.step(time_vals, estimation_errors[run_num][state_index+6], color="blue", alpha=alpha)
            ax.step(time_vals, three_sigmas[run_num][state_index+6], color="red", alpha=alpha)
            ax.step(time_vals, -three_sigmas[run_num][state_index+6], color="red", alpha=alpha)
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
                denominators[x_pixel_index, y_pixel_index, kernel_index], exponents[x_pixel_index, y_pixel_index, kernel_index] = assess_measurement_probability(assess_val - estimates[:, kernel_index], covariances[:, :, kernel_index])

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