
import numpy as np
from helper_functions import *
from EKF import *
from GM_EKF import *

def run_IMM(initial_estimate, initial_covariance, initial_mode_probabilities,
               dynamics_equations, measurement_equation, measurements,
               process_noise_covariance, measurement_noise_covariance,
               dynamics_args, measurement_args, mode_transition_matrix):

    time_vals = measurements.t
    measurement_vals = measurements.measurements
    individual_measurement_size = measurements.individual_measurement_size

    state_size = np.size(initial_estimate, 0)
    num_modes = np.size(initial_mode_probabilities, 0)
    measurement_size = np.size(measurement_vals, 0)
    num_measurements = np.size(measurement_vals, 1)

    mode_probability_vals = np.empty((num_modes, num_measurements+1))
    anterior_estimate_vals = np.empty((state_size, num_measurements, num_modes))
    posterior_estimate_vals = np.empty((state_size, num_measurements+1, num_modes))
    anterior_covariance_vals = np.empty((state_size, state_size, num_measurements, num_modes))
    posterior_covariance_vals = np.empty((state_size, state_size, num_measurements+1, num_modes))
    innovations_vals = np.empty((measurement_size, num_measurements, num_modes))

    mode_probability_vals[:, 0] = initial_mode_probabilities
    for mode_index in np.arange(num_modes):
        posterior_estimate_vals[:, 0, mode_index] = initial_estimate
        posterior_covariance_vals[:, :, 0, mode_index] = initial_covariance


    previous_time = 0
    previous_posterior_estimates = posterior_estimate_vals[:, 0, :]
    previous_posterior_covariances = posterior_covariance_vals[:, :, 0, :]
    previous_mode_probabilities = initial_mode_probabilities
    denominators = np.empty(num_modes)
    exponents = np.empty(num_modes)
    individual_total_conditional_probabilities = np.empty(num_modes)
    raw_mode_probabilities = np.empty(num_modes)
    
    for time_index in np.arange(1, num_measurements+1):

        measurement = measurement_vals[:, time_index-1]
        current_time = time_vals[time_index-1]
        timespan = current_time - previous_time
        
        anterior_estimates = np.empty((state_size, num_modes))
        posterior_estimates = np.empty((state_size, num_modes))
        anterior_covariances = np.empty((state_size, state_size, num_modes))
        posterior_covariances = np.empty((state_size, state_size, num_modes))

        if np.array_equal(measurement, np.empty(measurement_size)*np.nan, equal_nan=True):
            for mode_index in np.arange(num_modes):

                individual_total_conditional_probabilities[mode_index] = 1/num_modes

                previous_posterior_estimate = previous_posterior_estimates[:, mode_index]
                previous_posterior_covariance = previous_posterior_covariances[:, :, mode_index]
                dynamics_equation = dynamics_equations[mode_index]

                propagation_inputs = (previous_posterior_estimate, previous_posterior_covariance, dynamics_equation,
                                    process_noise_covariance, timespan, dynamics_args, measurement_size)        
                anterior_estimate, anterior_covariance, posterior_estimate, posterior_covariance, innovations = propagate_EKF(*propagation_inputs)

                anterior_estimates[:, mode_index] = anterior_estimate
                posterior_estimates[:, mode_index] = posterior_estimate
                anterior_covariances[:, :, mode_index] = anterior_covariance
                posterior_covariances[:, :, mode_index] = posterior_covariance
                # innovations_vals[:, time_index, mode_index] = innovations
                denominators[mode_index] = 1
                exponents[mode_index] = 1
        else:
            for mode_index in np.arange(num_modes):
                
                raw_conditional_probabilities = mode_transition_matrix[:, mode_index] * previous_mode_probabilities
                conditional_probabilities = raw_conditional_probabilities / np.sum(raw_conditional_probabilities)
                individual_total_conditional_probabilities[mode_index] = np.sum(raw_conditional_probabilities)    

                previous_posterior_estimate = previous_posterior_estimates[:, mode_index]
                previous_posterior_covariance = previous_posterior_covariances[:, :, mode_index]
                dynamics_equation = dynamics_equations[mode_index]

                mixed_initial_estimate = np.zeros(state_size)
                mixed_initial_covariance = np.zeros((state_size, state_size))

                for model_index in np.arange(num_modes):
                    mixed_initial_estimate += conditional_probabilities[model_index] * previous_posterior_estimates[:, model_index]
                for model_index in np.arange(num_modes):
                    difference = np.reshape(previous_posterior_estimates[:, model_index] - mixed_initial_estimate, (state_size, 1))
                    mixed_initial_covariance += conditional_probabilities[model_index] * (previous_posterior_covariances[:, :, model_index] + difference @ difference.T)

                EKF_inputs = (time_index, mixed_initial_estimate, mixed_initial_covariance, 
                            dynamics_equation, measurement_equation, individual_measurement_size, 
                            process_noise_covariance, measurement_noise_covariance, 
                            measurement, timespan, dynamics_args, measurement_args)      
                anterior_estimate, anterior_covariance, posterior_estimate, posterior_covariance, innovations, denominator, exponent = iterate_GM_kernel(*EKF_inputs)

                anterior_estimates[:, mode_index] = anterior_estimate
                posterior_estimates[:, mode_index] = posterior_estimate
                anterior_covariances[:, :, mode_index] = anterior_covariance
                posterior_covariances[:, :, mode_index] = posterior_covariance
                # innovations_vals[:, time_index, mode_index] = innovations
                denominators[mode_index] = denominator
                exponents[mode_index] = exponent

        new_mode_probabilities = np.empty(num_modes)        
        normalized_denominators = denominators / denominators.min()
        normalized_exponents = exponents - exponents.max()
        
        measurement_probabilities = 1 / normalized_denominators * np.exp(normalized_exponents)
        raw_mode_probabilities[mode_index] = measurement_probabilities*individual_total_conditional_probabilities
        new_mode_probabilities = raw_mode_probabilities / np.sum(raw_mode_probabilities)
        mode_probability_vals[:, time_index] = new_mode_probabilities

        anterior_estimate_vals[:, time_index-1, :] = anterior_estimates
        posterior_estimate_vals[:, time_index, :] = posterior_estimates
        anterior_covariance_vals[:, :, time_index-1, :] = anterior_covariances
        posterior_covariance_vals[:, :, time_index, :] = posterior_covariances

        previous_time = current_time
        previous_posterior_estimates = posterior_estimates
        previous_posterior_covariances = posterior_covariances
        previous_mode_probabilities = new_mode_probabilities

    return GM_FilterResults(time_vals, anterior_estimate_vals, posterior_estimate_vals, anterior_covariance_vals, posterior_covariance_vals, 0, mode_probability_vals)