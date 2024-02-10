
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


def plot_3sigma(time_vals, estimation_errors, three_sigmas, alpha, num_runs, state_size):

    ylabels = [r"X Position Estimation Error", r"X Velocity Estimation Error", r"Y Position Estimation Error", r"Y Velocity Estimation Error", r"Z Position Estimation Error", r"Z Velocity Estimation Error"]
    fig, axes = plt.subplots(2, 3, layout="constrained")
    
    axes_rowref = [0, 1, 0, 1, 0, 1]
    axes_colref = [0, 0, 1, 1, 2, 2]
    for state_index in range(state_size):
        
        ax = axes[axes_rowref[state_index], axes_colref[state_index]]
        if axes_rowref[state_index] == 1:
            ax.set_xlabel("Time, s", fontname="Times New Roman")
        ax.set_ylabel(ylabels[state_index], fontname="Times New Roman")
        ax.tick_params(axis="both", which="major", labelsize=6.5)
        for run_num in range(num_runs):
            ax.plot(time_vals, estimation_errors[run_num][state_index], color="blue", alpha=alpha)
            ax.plot(time_vals, three_sigmas[run_num][state_index], color="red", alpha=alpha)
            ax.plot(time_vals, -three_sigmas[run_num][state_index], color="red", alpha=alpha)
            ax.grid(True)

def compute_3sigmas(posterior_covariances, state_size):

    three_sigmas = []

    for posterior_covariance_vals in posterior_covariances:
        three_sigma_vals = []
        for state_index in np.arange(state_size):
            three_sigma_vals.append(3*np.sqrt(posterior_covariance_vals[state_index, state_index, :]))
        three_sigmas.append(three_sigma_vals)

    return three_sigmas

def compute_estimation_errors(truth, posterior_estimates, state_size):

    estimation_errors = []

    for posterior_estimate_vals in posterior_estimates:
        estimation_error_vals = []
        for state_index in np.arange(state_size): 
            estimation_error_vals.append(-truth[state_index, :] + posterior_estimate_vals[state_index, :])
        estimation_errors.append(estimation_error_vals)
    
    return estimation_errors