
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


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
        for state_index in np.arange(state_size):
            three_sigma_vals.append(3*np.sqrt(posterior_covariance_vals[state_index, state_index, :]))
        three_sigmas.append(three_sigma_vals)

    return three_sigmas

def compute_estimation_errors(truth, posterior_estimates, state_size):

    estimation_errors = []

    for posterior_estimate_vals in posterior_estimates:
        estimation_error_vals = []
        for state_index in np.arange(state_size): 
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