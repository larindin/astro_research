

import numpy as np
from helper_functions import *
from EKF import *

class GM_FilterResults:
    def __init__(self, time_vals, anterior_estimate_vals, posterior_estimate_vals, anterior_covariance_vals, posterior_covariance_vals, innovations_vals, weight_vals):
        self.t = time_vals
        self.anterior_estimate_vals = anterior_estimate_vals
        self.posterior_estimate_vals = posterior_estimate_vals
        self.anterior_covariance_vals = anterior_covariance_vals
        self.posterior_covariance_vals = posterior_covariance_vals
        self.innovations_vals = innovations_vals
        self.weight_vals = weight_vals

def iterate_GM_kernel(time_index, previous_posterior_estimate, previous_posterior_covariance, 
                dynamics_equation, measurement_equation, individual_measurement_size, 
                process_noise_covariance, measurement_noise_covariance, measurement, timespan, 
                dynamics_args, measurement_args):
    
    state_size = np.size(previous_posterior_estimate, 0)
    initial_conditions = np.concatenate((previous_posterior_estimate, previous_posterior_covariance.flatten()))
    args = dynamics_args + (process_noise_covariance,)
    propagation = scipy.integrate.solve_ivp(dynamics_equation, np.array([0, timespan]), initial_conditions, args=args, atol=1e-12, rtol=1e-12)

    output = propagation.y
    anterior_estimate = output[0:state_size, -1]
    anterior_covariance = output[state_size:(state_size+state_size**2), -1].reshape((state_size, state_size))
    anterior_covariance = enforce_symmetry(anterior_covariance)

    measurement, valid_indices = assess_measurement(measurement, individual_measurement_size)

    predicted_measurement, measurement_jacobian = measurement_equation(time_index, anterior_estimate, *measurement_args)
    predicted_measurement, measurement_jacobian = parse_measurement(predicted_measurement, measurement_jacobian, individual_measurement_size, valid_indices)

    measurement_noise_covariance = scipy.linalg.block_diag(*(measurement_noise_covariance, )*len(valid_indices))
    innovations_covariance = measurement_jacobian @ anterior_covariance @ measurement_jacobian.T + measurement_noise_covariance
    innovations_covariance = enforce_symmetry(innovations_covariance)
    cross_covariance = anterior_covariance @ measurement_jacobian.T
    gain_matrix = cross_covariance @ np.linalg.inv(innovations_covariance)

    innovations = measurement - predicted_measurement
    innovations = check_innovations(innovations)

    denominator, exponent = assess_measurement_probability(innovations, innovations_covariance)

    # posterior_estimate = anterior_estimate + gain_matrix @ innovations * np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    posterior_estimate = anterior_estimate + gain_matrix @ innovations
    posterior_covariance = anterior_covariance - cross_covariance @ gain_matrix.T - gain_matrix @ cross_covariance.T + gain_matrix @ innovations_covariance @ gain_matrix.T
    posterior_covariance = enforce_symmetry(posterior_covariance)

    return anterior_estimate, anterior_covariance, posterior_estimate, posterior_covariance, innovations, denominator, exponent

def run_GM_EKF(initial_estimates, initial_covariances, initial_weights,
               dynamics_equation, measurement_equation, measurements,
               process_noise_covariance, measurement_noise_covariance,
               dynamics_args, measurement_args):
    
    time_vals = measurements.t
    measurement_vals = measurements.measurements
    individual_measurement_size = measurements.individual_measurement_size

    state_size = np.size(initial_estimates, 0)
    num_kernels = np.size(initial_estimates, 1)
    measurement_size = np.size(measurement_vals, 0)
    num_measurements = np.size(measurement_vals, 1)

    weight_vals = np.empty((num_kernels, num_measurements+1))
    anterior_estimate_vals = np.empty((state_size, num_measurements, num_kernels))
    posterior_estimate_vals = np.empty((state_size, num_measurements+1, num_kernels))
    anterior_covariance_vals = np.empty((state_size, state_size, num_measurements, num_kernels))
    posterior_covariance_vals = np.empty((state_size, state_size, num_measurements+1, num_kernels))
    innovations_vals = np.empty((measurement_size, num_measurements, num_kernels))
    
    weight_vals[:, 0] = initial_weights
    posterior_estimate_vals[:, 0, :] = initial_estimates
    posterior_covariance_vals[:, :, 0, :] = initial_covariances
    # for kernel_index in np.arange(num_kernels): 
    #     posterior_estimate_vals[:, 0, kernel_index] = initial_estimates[:, kernel_index]
    #     posterior_covariance_vals [:, :, 0, kernel_index] = initial_covariances[:, :, kernel_index]

    previous_time = 0
    previous_posterior_estimates = initial_estimates
    previous_posterior_covariances = initial_covariances
    previous_weights = initial_weights
    
    denominators = np.empty(num_kernels)
    exponents = np.empty(num_kernels)

    for time_index in np.arange(1, num_measurements + 1):

        measurement = measurement_vals[:, time_index-1]
        current_time = time_vals[time_index-1]
        timespan = current_time - previous_time

        if np.array_equal(measurement, np.empty(measurement_size)*np.nan, equal_nan=True):
            for kernel_index in np.arange(num_kernels):

                previous_posterior_estimate = previous_posterior_estimates[:, kernel_index]
                previous_posterior_covariance = previous_posterior_covariances[:, :, kernel_index]

                propagation_inputs = (previous_posterior_estimate, previous_posterior_covariance, dynamics_equation,
                                    process_noise_covariance, timespan, dynamics_args, measurement_size)        
                anterior_estimate, anterior_covariance, posterior_estimate, posterior_covariance, innovations = propagate_EKF(*propagation_inputs)

                anterior_estimate_vals[:, time_index-1, kernel_index] = anterior_estimate
                posterior_estimate_vals[:, time_index, kernel_index] = posterior_estimate
                anterior_covariance_vals[:, :, time_index-1, kernel_index] = anterior_covariance
                posterior_covariance_vals[:, :, time_index, kernel_index] = posterior_covariance
                # innovations_vals[:, time_index, kernel_index] = innovations
                denominators[kernel_index] = 1
                exponents[kernel_index] = 1
        else:
            for kernel_index in np.arange(num_kernels):

                previous_posterior_estimate = previous_posterior_estimates[:, kernel_index]
                previous_posterior_covariance = previous_posterior_covariances[:, :, kernel_index]

                EKF_inputs = (time_index, previous_posterior_estimate, previous_posterior_covariance, 
                            dynamics_equation, measurement_equation, individual_measurement_size, 
                            process_noise_covariance, measurement_noise_covariance, 
                            measurement, timespan, dynamics_args, measurement_args)      
                anterior_estimate, anterior_covariance, posterior_estimate, posterior_covariance, innovations, denominator, exponent = iterate_GM_kernel(*EKF_inputs)

                anterior_estimate_vals[:, time_index-1, kernel_index] = anterior_estimate
                posterior_estimate_vals[:, time_index, kernel_index] = posterior_estimate
                anterior_covariance_vals[:, :, time_index-1, kernel_index] = anterior_covariance
                posterior_covariance_vals[:, :, time_index, kernel_index] = posterior_covariance
                # innovations_vals[:, time_index, kernel_index] = innovations
                denominators[kernel_index] = denominator
                exponents[kernel_index] = exponent
        
        new_weights = np.empty(num_kernels)
        normalized_denominators = denominators / denominators.min()
        normalized_exponents = exponents - exponents.max()
        measurement_probabilities = 1 / normalized_denominators * np.exp(normalized_exponents)
        probability_sum = np.sum(previous_weights * measurement_probabilities)
        for kernel_index in np.arange(num_kernels):
            new_weights[kernel_index] = previous_weights[kernel_index]*measurement_probabilities[kernel_index]/probability_sum
        weight_vals[:, time_index] = new_weights


        previous_time = current_time
        previous_posterior_estimate = posterior_estimate
        previous_posterior_covariance = posterior_covariance
        previous_weights = new_weights
    
    return GM_FilterResults(time_vals, anterior_estimate_vals, posterior_estimate_vals, anterior_covariance_vals, posterior_covariance_vals, innovations_vals, weight_vals)