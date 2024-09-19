

import numpy as np
import scipy
from EKF import *

def run_batch_processor(initial_estimate, initial_covariance, time_indices, 
                        measurements, dynamics_function, measurement_function,
                        dynamics_args, measurement_args, tolerance=1e-12, max_iterations=100):
    
    time_vals = measurements.t
    measurement_vals = measurements.measurements
    individual_measurement_size = measurements.individual_measurement_size

    timespan = time_vals[1] - time_vals[0]

    state_size = len(initial_estimate)
    num_measurements = np.size(measurement_vals, 1)
    initial_STM = np.eye(state_size)

    posterior_estimate = initial_estimate
    posterior_STM = initial_STM

    information_matrix = np.zeros((state_size, state_size))
    information_vector = np.zeros((state_size))

    estimates = np.empty((max_iterations, state_size))

    iteration_number = 1
    while iteration_number <= max_iterations:

        print(iteration_number)

        posterior_estimate = initial_estimate
        posterior_STM = initial_STM
        
        for measurement_index in np.arange(num_measurements):

            time_index = time_indices[measurement_index]

            propagation_ICs = np.concatenate((posterior_estimate, posterior_STM.flatten()))
            propagation = scipy.integrate.solve_ivp(dynamics_function, np.array([0, timespan]), propagation_ICs, args=dynamics_args, atol=1e-12, rtol=1e-12)
            output = propagation.y

            posterior_estimate = output[0:state_size, -1]
            posterior_STM = output[state_size:state_size**2 + state_size, -1].reshape((state_size, state_size))

            measurement = measurement_vals[:, measurement_index]
            measurement, valid_indices = assess_measurement(measurement, individual_measurement_size)

            predicted_measurement, measurement_jacobian = measurement_function(time_index, posterior_estimate, *measurement_args)
            predicted_measurement, measurement_jacobian = parse_measurement(predicted_measurement, measurement_jacobian, individual_measurement_size, valid_indices)

            innovations = measurement - predicted_measurement
            innovations = check_innovations(innovations)

            H = measurement_jacobian @ posterior_STM

            R = np.eye(len(innovations))

            factor = H.T @ np.linalg.inv(R)
            information_matrix += factor @ H
            information_vector += factor @ innovations
        
        covariance= np.linalg.inv(information_matrix)
        deviation = covariance @ information_vector

        iteration_number += 1

        print(deviation)
        
        if np.linalg.norm(deviation) < tolerance:
            print("successfully found solution")
            break
        
        initial_estimate += deviation

        estimates[iteration_number - 2, :] = initial_estimate
    
    return estimates, covariance