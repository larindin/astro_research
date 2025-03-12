

import numpy as np
import scipy.integrate
from helper_functions import *

class EKF_Results:
    def __init__(self, 
                 time_vals, 
                 anterior_estimate_vals, 
                 posterior_estimate_vals, 
                 anterior_covariance_vals, 
                 posterior_covariance_vals, 
                 innovations_vals):
        self.time_vals = time_vals
        self.anterior_estimate_vals = anterior_estimate_vals
        self.posterior_estimate_vals = posterior_estimate_vals
        self.anterior_covariance_vals = anterior_covariance_vals
        self.posterior_covariance_vals = posterior_covariance_vals
        self.innovations_vals = innovations_vals

class EKF():
    def __init__(self,
                 dynamics_function,
                 dynamics_function_args,
                 measurement_function,
                 measurement_function_args,
                 process_noise_covariance
                 ):
        
        self.dynamics_function = dynamics_function
        self.dynamics_function_args = dynamics_function_args
        self.measurement_function = measurement_function
        self.measurement_function_args = measurement_function_args
        self.process_noise_covariance = process_noise_covariance
    
    def run(self, initial_estimate, initial_covariance, time_vals, measurement_vals):

        state_size = np.size(initial_estimate, 0)
        measurement_size = np.size(measurement_vals, 0)
        num_measurements = np.size(measurement_vals, 1)

        anterior_estimate_vals = np.full((state_size, num_measurements), np.nan)
        posterior_estimate_vals = np.full((state_size, num_measurements), np.nan)
        anterior_covariance_vals = np.full((state_size, state_size, num_measurements), np.nan)
        posterior_covariance_vals = np.full((state_size, state_size, num_measurements), np.nan)
        innovations_vals = np.full((measurement_size, num_measurements), np.nan)

        anterior_estimate_vals[:, 0] = initial_estimate
        anterior_covariance_vals[:, :, 0] = initial_covariance

        posterior_estimate, posterior_covariance = self.measurement_update(0, initial_estimate, initial_covariance, measurement_vals[:, 0])
        posterior_estimate_vals[:, 0] = posterior_estimate
        posterior_covariance_vals[:, :, 0] = posterior_covariance
        
        previous_time = 0
        previous_posterior_estimate = posterior_estimate
        previous_posterior_covariance = posterior_covariance

        for time_index in range(1, num_measurements):
            
            current_time = time_vals[time_index]
            timespan = current_time - previous_time
            
            current_measurement = measurement_vals[:, time_index]

            anterior_estimate, anterior_covariance = self.time_update(time_index, previous_posterior_estimate, previous_posterior_covariance, timespan)

            posterior_estimate, posterior_covariance = self.measurement_update(time_index, anterior_estimate, anterior_covariance, current_measurement)

            anterior_estimate_vals[:, time_index] = anterior_estimate
            anterior_covariance_vals[:, :, time_index] = anterior_covariance
            posterior_estimate_vals[:, time_index] = posterior_estimate
            posterior_covariance_vals[:, :, time_index] = posterior_covariance

            previous_posterior_estimate, previous_posterior_covariance = posterior_estimate, posterior_covariance
            previous_time = current_time
        
        return EKF_Results(time_vals, anterior_estimate_vals, posterior_estimate_vals, anterior_covariance_vals, posterior_covariance_vals, innovations_vals)

    def measurement_update(self, time_index, anterior_estimate, anterior_covariance, measurement):

        measurement = measurement[np.isnan(measurement) == False]
        if len(measurement) == 0:
            return anterior_estimate, anterior_covariance

        predicted_measurement, measurement_jacobian, measurement_noise_covariance, rs = self.measurement_function(time_index, anterior_estimate, *self.measurement_function_args)

        innovations_covariance = measurement_jacobian @ anterior_covariance @ measurement_jacobian.T + measurement_noise_covariance
        innovations_covariance = enforce_symmetry(innovations_covariance)
        cross_covariance = anterior_covariance @ measurement_jacobian.T
        gain_matrix = cross_covariance @ np.linalg.inv(innovations_covariance)

        innovations = measurement - predicted_measurement
        # innovations = check_innovations(innovations)
        for sensor_index, r in enumerate(rs):
            innovations[sensor_index*3:(sensor_index+1)*3]*= r

        posterior_estimate = anterior_estimate + gain_matrix @ innovations
        posterior_covariance = enforce_symmetry(anterior_covariance - cross_covariance @ gain_matrix.T - gain_matrix @ cross_covariance.T + gain_matrix @ innovations_covariance @ gain_matrix.T)

        denominator, exponent = assess_measurement_likelihood(innovations, innovations_covariance)

        return posterior_estimate, posterior_covariance

    def time_update(self, time_index, posterior_estimate, posterior_covariance, timespan):

        state_size = len(posterior_estimate)

        ICs = np.concatenate((posterior_estimate, np.eye(state_size).flatten()))
        propagation = scipy.integrate.solve_ivp(self.dynamics_function, [0,timespan], ICs, args=self.dynamics_function_args, atol=1e-12, rtol=1e-12).y[:, -1]

        STM = propagation[state_size:state_size**2 + state_size].reshape((state_size, state_size))
        anterior_estimate = propagation[0:state_size]
        anterior_covariance = enforce_symmetry(STM @ posterior_covariance @ STM.T + self.process_noise_covariance)

        return anterior_estimate, anterior_covariance

def assess_measurement(measurement, individual_measurement_size):

    measurement_size = np.size(measurement, 0)
    num_measurements = int(measurement_size/individual_measurement_size)
    new_measurement = np.array([])
    valid_indices = []

    for checking_index in range(num_measurements):
        
        to_be_checked = measurement[checking_index*individual_measurement_size:(checking_index + 1)*individual_measurement_size]
        
        if not np.array_equal(to_be_checked, np.empty(individual_measurement_size)*np.nan, equal_nan=True):
            new_measurement = np.concatenate((new_measurement, to_be_checked))
            valid_indices.append(checking_index)
    
    return new_measurement, valid_indices