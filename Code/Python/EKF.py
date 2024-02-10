

import numpy as np
import scipy.integrate
from dynamics_functions import *

class FilterResults:
    def __init__(self, time_vals, anterior_estimate_vals, posterior_estimate_vals, anterior_covariance_vals, posterior_covariance_vals, innovations_vals):
        self.t = time_vals
        self.anterior_estimate_vals = anterior_estimate_vals
        self.posterior_estimate_vals = posterior_estimate_vals
        self.anterior_covariance_vals = anterior_covariance_vals
        self.posterior_covariance_vals = posterior_covariance_vals
        self.innovations_vals = innovations_vals

def enforce_symmetry(covariance_matrix: np.ndarray):
    fixed_matrix = (covariance_matrix + covariance_matrix.T) / 2
    return fixed_matrix
    
def check_innovations(innovations):
    for index, innovation in enumerate(innovations):
        if abs(innovation) > np.pi:
            innovations[index] = -np.sign(innovation)*(2*np.pi - abs(innovation))
    return innovations

def iterate_EKF(previous_posterior_estimate, previous_posterior_covariance, 
                dynamics_equation, measurement_equation, 
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

    predicted_measurement, measurement_jacobian = measurement_equation(anterior_estimate, *measurement_args)

    innovations_covariance = measurement_jacobian @ anterior_covariance @ measurement_jacobian.T + measurement_noise_covariance
    innovations_covariance = enforce_symmetry(innovations_covariance)
    cross_covariance = anterior_covariance @ measurement_jacobian.T
    gain_matrix = cross_covariance @ np.linalg.inv(innovations_covariance)

    innovations = measurement - predicted_measurement
    # innovations = check_innovations(innovations)

    posterior_estimate = anterior_estimate + gain_matrix @ innovations
    posterior_covariance = anterior_covariance - cross_covariance @ gain_matrix.T - gain_matrix @ cross_covariance.T + gain_matrix @ innovations_covariance @ gain_matrix.T
    posterior_covariance = enforce_symmetry(posterior_covariance)

    return anterior_estimate, anterior_covariance, posterior_estimate, posterior_covariance, innovations

def run_EKF(initial_estimate, initial_covariance, 
            dynamics_equation, measurement_equation, measurements, 
            process_noise_covariance, measurement_noise_covariance, 
            dynamics_args, measurement_args):

    time_vals = measurements.t
    measurement_vals = measurements.measurements

    state_size = np.size(initial_estimate, 0)
    measurement_size = np.size(measurement_vals, 0)
    num_measurements = np.size(measurement_vals, 1)

    anterior_estimate_vals = np.zeros((state_size, num_measurements))
    posterior_estimate_vals = np.zeros((state_size, num_measurements+1))
    anterior_covariance_vals = np.zeros((state_size, state_size, num_measurements))
    posterior_covariance_vals = np.zeros((state_size, state_size, num_measurements+1))
    innovations_vals = np.zeros((measurement_size, num_measurements))

    posterior_estimate_vals[:, 0] = initial_estimate
    posterior_covariance_vals[:, :, 0] = initial_covariance

    previous_time = 0
    previous_posterior_estimate = initial_estimate
    previous_posterior_covariance = initial_covariance

    for time_index in np.arange(num_measurements):
        
        measurement = measurement_vals[:, time_index]
        current_time = time_vals[time_index]
        timespan = current_time - previous_time

        EKF_inputs = (previous_posterior_estimate, previous_posterior_covariance, 
                      dynamics_equation, measurement_equation, 
                      process_noise_covariance, measurement_noise_covariance, 
                      measurement, timespan, 
                      dynamics_args, measurement_args)
        
        anterior_estimate, anterior_covariance, posterior_estimate, posterior_covariance, innovations = iterate_EKF(*EKF_inputs)
        
        anterior_estimate_vals[:, time_index] = anterior_estimate
        posterior_estimate_vals[:, time_index+1] = posterior_estimate
        anterior_covariance_vals[:, :, time_index] = anterior_covariance
        posterior_covariance_vals[:, :, time_index+1] = posterior_covariance
        innovations_vals[:, time_index] = innovations

        previous_time = current_time
        previous_posterior_estimate = posterior_estimate
        previous_posterior_covariance = posterior_covariance

    return FilterResults(time_vals, anterior_estimate_vals, posterior_estimate_vals, anterior_covariance_vals, posterior_covariance_vals, innovations_vals)