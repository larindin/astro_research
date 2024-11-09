

import numpy as np
from EKF import *
from helper_functions import *


def run_dual_filter(initial_estimate, initial_covariance, 
            first_dynamics_equation, first_measurement_equation, 
            second_dynamics_equation, second_measurement_equation, measurements, 
            first_process_noise_covariance, first_measurement_noise_covariance,
            second_process_noise_covariance, second_measurement_noise_covariance, 
            first_dynamics_args, first_measurement_args,
            second_dynamics_args, second_measurement_args,
            timeout_count, switching_count, filter_index):

    time_vals = measurements.t
    measurement_vals = measurements.measurements
    individual_measurement_size = measurements.individual_measurement_size

    state_size = np.size(initial_estimate, 0)
    measurement_size = np.size(measurement_vals, 0)
    num_measurements = np.size(measurement_vals, 1)

    anterior_estimate_vals = np.empty((state_size, num_measurements))
    posterior_estimate_vals = np.empty((state_size, num_measurements))
    anterior_covariance_vals = np.empty((state_size, state_size, num_measurements))
    posterior_covariance_vals = np.empty((state_size, state_size, num_measurements))
    innovations_vals = np.empty((measurement_size, num_measurements))

    anterior_estimate_vals[:, 0] = initial_estimate
    anterior_covariance_vals[:, :, 0] = initial_covariance
    posterior_estimate_vals[:, 0] = initial_estimate
    posterior_covariance_vals[:, :, 0] = initial_covariance

    previous_time = 0
    previous_posterior_estimate = initial_estimate
    previous_posterior_covariance = initial_covariance

    observable_count = 0
    unobservable_count = 0

    active_filter_index = filter_index

    for time_index in np.arange(1, num_measurements):

        measurement = measurement_vals[:, time_index]
        current_time = time_vals[time_index]
        timespan = current_time - previous_time

        if np.array_equal(measurement, np.empty(measurement_size)*np.nan, equal_nan=True):
            unobservable_count += 1 * (unobservable_count < timeout_count)
            observable_count -= 1 * (observable_count > 0)

            if unobservable_count == timeout_count:
                active_filter_index = 1

            propagation_inputs = (previous_posterior_estimate, previous_posterior_covariance, first_dynamics_equation,
                                  first_process_noise_covariance, timespan, first_dynamics_args, measurement_size)        
            anterior_estimate, anterior_covariance, posterior_estimate, posterior_covariance, innovations = propagate_EKF(*propagation_inputs)
        else:
            observable_count += 1 * (observable_count < switching_count)
            unobservable_count -= 1 * (unobservable_count > 0)

            if observable_count == switching_count:
                active_filter_index = 0

            if active_filter_index == 0:
                dynamics_equation = first_dynamics_equation
                dynamics_args = first_dynamics_args
                measurement_equation = first_measurement_equation
                measurement_args = first_measurement_args
                process_noise_covariance = first_process_noise_covariance
                measurement_noise_covariance = first_measurement_noise_covariance

                EKF_inputs = (time_index, previous_posterior_estimate, previous_posterior_covariance, 
                        dynamics_equation, measurement_equation, individual_measurement_size, 
                        process_noise_covariance, measurement_noise_covariance, 
                        measurement, timespan, dynamics_args, measurement_args)      
                anterior_estimate, anterior_covariance, posterior_estimate, posterior_covariance, innovations = iterate_EKF(*EKF_inputs)
            else:
                dynamics_equation = second_dynamics_equation
                dynamics_args = second_dynamics_args
                measurement_equation = second_measurement_equation
                measurement_args = second_measurement_args
                process_noise_covariance = second_process_noise_covariance
                measurement_noise_covariance = second_measurement_noise_covariance
            
                EKF_inputs = (time_index, previous_posterior_estimate[0:6], previous_posterior_covariance[0:6, 0:6], 
                        dynamics_equation, measurement_equation, individual_measurement_size, 
                        process_noise_covariance, measurement_noise_covariance, 
                        measurement, timespan, dynamics_args, measurement_args)      
                anterior_estimate, anterior_covariance, posterior_estimate, posterior_covariance, innovations = iterate_EKF(*EKF_inputs)
                
                anterior_estimate = np.concatenate((anterior_estimate, previous_posterior_estimate[6:12]))
                anterior_covariance = scipy.linalg.block_diag(anterior_covariance, previous_posterior_covariance[6:12, 6:12])
                posterior_estimate = np.concatenate((posterior_estimate, previous_posterior_estimate[6:12]))
                posterior_covariance = scipy.linalg.block_diag(posterior_covariance, previous_posterior_covariance[6:12, 6:12])

            
        
        anterior_estimate_vals[:, time_index] = anterior_estimate
        posterior_estimate_vals[:, time_index] = posterior_estimate
        anterior_covariance_vals[:, :, time_index] = anterior_covariance
        posterior_covariance_vals[:, :, time_index] = posterior_covariance
        # innovations_vals[:, time_index] = innovations

        previous_time = current_time
        previous_posterior_estimate = posterior_estimate
        previous_posterior_covariance = posterior_covariance

    return FilterResults(time_vals, anterior_estimate_vals, posterior_estimate_vals, anterior_covariance_vals, posterior_covariance_vals, innovations_vals)