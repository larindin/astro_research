

import numpy as np
import scipy.integrate
from joblib import Parallel, delayed, parallel_config
from EKF import *
from helper_functions import *

def run_EKF_smoothing(dynamics_equation, dynamics_args, time_vals, anterior_estimate_vals, posterior_estimate_vals, anterior_covariance_vals, posterior_covariance_vals, horizon):

    num_vals = horizon + 1
    state_size = np.size(posterior_estimate_vals, 0)

    smoothed_estimate_vals = np.empty((state_size, num_vals))
    smoothed_covariance_vals = np.empty((state_size, state_size, num_vals))

    smoothed_estimate_vals[:, -1] = posterior_estimate_vals[:, -1]
    smoothed_covariance_vals[:, :, -1] = posterior_covariance_vals[:, :, -1]

    for val_index in range(-2, -num_vals-1, -1):
        
        tspan = [time_vals[val_index+1], time_vals[val_index]]
        initial_state = smoothed_estimate_vals[:, val_index+1]
        ICs = np.concatenate((initial_state, np.eye(state_size).flatten()))
        propagation = scipy.integrate.solve_ivp(dynamics_equation, tspan, ICs, args=dynamics_args, atol=1e-12, rtol=1e-12).y[:, -1]
        
        STM = np.linalg.inv(np.reshape(propagation[state_size:], (state_size, state_size)))

        S = posterior_covariance_vals[:, :, val_index] @ STM.T @ np.linalg.inv(anterior_covariance_vals[:, :, val_index+1])
        smoothed_estimate_vals[:, val_index] = posterior_estimate_vals[:, val_index] + S @ (smoothed_estimate_vals[:, val_index+1] - anterior_estimate_vals[:, val_index+1])
        smoothed_covariance_vals[:, :, val_index] = posterior_covariance_vals[:, :, val_index] + S @ (smoothed_covariance_vals[:, :, val_index+1] - anterior_covariance_vals[:, :, val_index+1]) @ S.T
        
    return smoothed_estimate_vals, smoothed_covariance_vals

def run_EKF_smoothing_MC(dynamics_equation, dynamics_args, time_vals, anterior_estimates, posterior_estimates, anterior_covariances, posterior_covariances, horizon):

    num_runs = len(anterior_estimates)
    smoothed_estimates = []
    smoothed_covariances = []

    with parallel_config(verbose=100, n_jobs=-1):
        results = Parallel()(delayed(run_EKF_smoothing)(dynamics_equation, dynamics_args, time_vals, anterior_estimates[run_index], posterior_estimates[run_index], anterior_covariances[run_index], posterior_covariances[run_index], horizon) for run_index in range(num_runs))
    
    for run_index in range(num_runs):
        smoothed_estimates.append(results[run_index][0])
        smoothed_covariances.append(results[run_index][1])

    return smoothed_estimates, smoothed_covariances

def run_smoothing_consistency_test(posterior_estimate_vals, smoothed_estimate_vals, posterior_covariance_vals, smoothed_covariance_vals, detection_threshold):

    num_timesteps = np.size(posterior_estimate_vals, 1)

    estimate_differences = posterior_estimate_vals - smoothed_estimate_vals
    covariance_differences = posterior_covariance_vals - smoothed_covariance_vals

    violated_bool_vector = np.full(num_timesteps, False)
    for timestep_index in range(num_timesteps):
        cov_difference_diag = np.diag(covariance_differences[:, :, timestep_index])
        violation_vector = np.diag(1/np.sqrt(cov_difference_diag)) @ estimate_differences[:, timestep_index]
        violated_bool_vector[timestep_index] = max(violation_vector.flatten()) > detection_threshold
    
    return violated_bool_vector

def run_maneuver_detection_alg(dynamics_equation, dynamics_args, time_vals, anterior_estimate_vals, posterior_estimate_vals, anterior_covariance_vals, posterior_covariance_vals, horizon, detection_threshold):

    num_timesteps = len(time_vals)

    previous_timestep_index = 0
    next_timestep_index = horizon
    maneuver_start_index = np.empty(0)

    while next_timestep_index < num_timesteps:

        current_anterior_vals = anterior_estimate_vals[:, previous_timestep_index:next_timestep_index+1]
        current_anterior_covs = anterior_covariance_vals[:, :, previous_timestep_index:next_timestep_index+1]
        current_posterior_vals = posterior_estimate_vals[:, previous_timestep_index:next_timestep_index+1]
        current_posterior_covs = posterior_covariance_vals[:, :, previous_timestep_index:next_timestep_index+1]

        current_smoothed_estimates, current_smoothed_covariances = run_EKF_smoothing(dynamics_equation,
                                                                                     dynamics_args,
                                                                                     time_vals,
                                                                                     current_anterior_vals,
                                                                                     current_posterior_vals,
                                                                                     current_anterior_covs,
                                                                                     current_posterior_covs,
                                                                                     horizon)
        
        maneuver_detection_bool = run_smoothing_consistency_test(current_posterior_vals,
                                                                 current_smoothed_estimates,
                                                                 current_posterior_covs,
                                                                 current_smoothed_covariances,
                                                                 detection_threshold)
        
        maneuvering_indices = np.where(maneuver_detection_bool)
        if np.size(maneuvering_indices) != 0:
            maneuver_start_index = previous_timestep_index + maneuvering_indices[0]
            break
        
        previous_timestep_index = next_timestep_index
        next_timestep_index += horizon

    return int(maneuver_start_index[0])

def run_maneuver_detection_alg_MC(dynamics_equation, dynamics_args, time_vals, anterior_estimates, posterior_estimates, anterior_covariances, posterior_covariances, horizon, detection_threshold):

    num_runs = len(anterior_estimates)
    maneuver_start_indices = np.empty(num_runs)

    with parallel_config(verbose=100, n_jobs=-1):
        results = Parallel()(delayed(run_maneuver_detection_alg)(dynamics_equation, dynamics_args, time_vals, anterior_estimates[run_index], posterior_estimates[run_index], anterior_covariances[run_index], posterior_covariances[run_index], horizon, detection_threshold) for run_index in range(num_runs))
    
    for run_index in range(num_runs):
        maneuver_start_indices[run_index] = results[run_index]

    return maneuver_start_indices


def run_OCBE_smoothing(results, control_noise_covariance):

    t = results.t
    posterior_estimate_vals = results.posterior_estimate_vals
    anterior_covariance_vals = results.anterior_covariance_vals
    posterior_covariance_vals = results.posterior_covariance_vals
    STM_vals = results.STM_vals
    costate_vals = results.costate_vals
    control_vals = results.control_vals

    num_vals = len(t)
    state_size = np.size(posterior_estimate_vals, 0)
    B = np.vstack((np.zeros((3, 3)), np.eye(3)))

    smoothed_estimate_vals = np.empty((state_size, num_vals))
    smoothed_covariance_vals = np.empty((state_size, state_size, num_vals))
    smoothed_costate_vals = np.empty((state_size, num_vals))
    smoothed_control_vals = np.empty((3, num_vals))

    smoothed_estimate_vals[:, -1] = posterior_estimate_vals[:, -1]
    smoothed_covariance_vals[:, :, -1] = posterior_covariance_vals[:, :, -1]
    smoothed_costate_vals[:, -1] = costate_vals[:, -1]
    smoothed_control_vals[:, -1] = control_vals[:, -1]

    for val_index in range(num_vals-2, 0, -1):

        STM_xx = STM_vals[:, :, val_index+1][0:state_size, 0:state_size]
        STM_ll = STM_vals[:, :, val_index][state_size:2*state_size, state_size:2*state_size]

        S = posterior_covariance_vals[:, :, val_index] @ STM_xx.T @ np.linalg.inv(anterior_covariance_vals[:, :, val_index+1])
        smoothed_estimate_vals[:, val_index] = posterior_estimate_vals[:, val_index] + S @ (smoothed_estimate_vals[:, val_index+1] - STM_xx @ posterior_estimate_vals[:, val_index])
        smoothed_covariance_vals[:, :, val_index] = S @ (smoothed_covariance_vals[:, :, val_index+1] - anterior_covariance_vals[:, :, val_index+1]) @ S.T
        smoothed_costate_vals[:, val_index] = -np.linalg.inv(posterior_covariance_vals[:, :, val_index]) @ S @ (smoothed_estimate_vals[:, val_index+1] - STM_xx @ posterior_estimate_vals[:, val_index])
        smoothed_control_vals[:, val_index] = -control_noise_covariance @ B.T @ STM_ll @ np.linalg.inv(posterior_covariance_vals[:, :, val_index]) @ (posterior_estimate_vals[:, val_index] - smoothed_estimate_vals[:, val_index])

    return smoothed_estimate_vals, smoothed_covariance_vals, smoothed_costate_vals, smoothed_control_vals
    