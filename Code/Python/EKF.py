

import numpy as np
import scipy.integrate
from helper_functions import *

class FilterResults:
    def __init__(self, time_vals, anterior_estimate_vals, posterior_estimate_vals, anterior_covariance_vals, posterior_covariance_vals, STM_vals, innovations_vals):
        self.t = time_vals
        self.anterior_estimate_vals = anterior_estimate_vals
        self.posterior_estimate_vals = posterior_estimate_vals
        self.anterior_covariance_vals = anterior_covariance_vals
        self.posterior_covariance_vals = posterior_covariance_vals
        self.STM_vals = STM_vals
        self.innovations_vals = innovations_vals

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

def parse_measurement(measurement, measurement_jacobian, individual_measurement_size, valid_indices):
    new_measurement = np.empty(len(valid_indices)*individual_measurement_size)
    new_measurement_jacobian = np.empty((len(valid_indices)*individual_measurement_size, np.size(measurement_jacobian, 1)))
    for index, valid_index in enumerate(valid_indices):
        new_measurement[index*individual_measurement_size:(index+1)*individual_measurement_size] = measurement[valid_index*individual_measurement_size:(valid_index+1)*individual_measurement_size]
        new_measurement_jacobian[index*individual_measurement_size:(index+1)*individual_measurement_size, :] = measurement_jacobian[valid_index*individual_measurement_size:(valid_index+1)*individual_measurement_size, :]
    
    return new_measurement, new_measurement_jacobian

def iterate_EKF(time_index, previous_posterior_estimate, previous_posterior_covariance, 
                dynamics_equation, measurement_equation, individual_measurement_size, 
                process_noise_covariance, measurement_noise_covariance, measurement, timespan, 
                dynamics_args, measurement_args):

    state_size = np.size(previous_posterior_estimate, 0)
    initial_conditions = np.concatenate((previous_posterior_estimate, np.eye(state_size).flatten()))
    args = dynamics_args
    propagation = scipy.integrate.solve_ivp(dynamics_equation, np.array([0, timespan]), initial_conditions, args=args, atol=1e-12, rtol=1e-12)

    output = propagation.y
    anterior_estimate = output[0:state_size, -1]
    STM = output[state_size:(state_size+state_size**2), -1].reshape((state_size, state_size))
    anterior_covariance = STM @ previous_posterior_covariance @ STM.T + process_noise_covariance
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

    posterior_estimate = anterior_estimate + gain_matrix @ innovations
    posterior_covariance = anterior_covariance - cross_covariance @ gain_matrix.T - gain_matrix @ cross_covariance.T + gain_matrix @ innovations_covariance @ gain_matrix.T
    posterior_covariance = enforce_symmetry(posterior_covariance)

    return anterior_estimate, anterior_covariance, posterior_estimate, posterior_covariance, STM, innovations

def propagate_EKF(previous_posterior_estimate, previous_posterior_covariance, 
                  dynamics_equation, process_noise_covariance, timespan, 
                  dynamics_args, measurement_size):
    
    state_size = np.size(previous_posterior_estimate, 0)
    initial_conditions = np.concatenate((previous_posterior_estimate, np.eye(state_size).flatten()))
    propagation = scipy.integrate.solve_ivp(dynamics_equation, np.array([0, timespan]), initial_conditions, args=dynamics_args, atol=1e-12, rtol=1e-12)

    output = propagation.y
    anterior_estimate = output[0:state_size, -1]
    STM = output[state_size:(state_size+state_size**2), -1].reshape((state_size, state_size))
    anterior_covariance = STM @ previous_posterior_covariance @ STM.T + process_noise_covariance
    anterior_covariance = enforce_symmetry(anterior_covariance)

    posterior_esimate = anterior_estimate
    posterior_covariance = anterior_covariance

    innovations = np.ones(measurement_size)*np.nan

    return anterior_estimate, anterior_covariance, posterior_esimate, posterior_covariance, STM, innovations

def run_EKF(initial_estimate, initial_covariance, 
            dynamics_equation, measurement_equation, measurements, 
            process_noise_covariance, measurement_noise_covariance, 
            dynamics_args, measurement_args):

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
    STM_vals = np.empty((state_size, state_size, num_measurements))
    innovations_vals = np.empty((measurement_size, num_measurements))

    anterior_estimate_vals[:, 0] = initial_estimate
    posterior_estimate_vals[:, 0] = initial_estimate
    anterior_covariance_vals[:, :, 0] = initial_covariance
    posterior_covariance_vals[:, :, 0] = initial_covariance
    STM_vals[:, :, 0] = np.eye(state_size)

    previous_time = 0
    previous_posterior_estimate = initial_estimate
    previous_posterior_covariance = initial_covariance

    for time_index in range(1, num_measurements):

        measurement = measurement_vals[:, time_index]
        current_time = time_vals[time_index]
        timespan = current_time - previous_time

        if np.array_equal(measurement, np.empty(measurement_size)*np.nan, equal_nan=True):
            propagation_inputs = (previous_posterior_estimate, previous_posterior_covariance, dynamics_equation,
                                  process_noise_covariance, timespan, dynamics_args, measurement_size)        
            anterior_estimate, anterior_covariance, posterior_estimate, posterior_covariance, STM, innovations = propagate_EKF(*propagation_inputs)
        else:
            EKF_inputs = (time_index, previous_posterior_estimate, previous_posterior_covariance, 
                        dynamics_equation, measurement_equation, individual_measurement_size, 
                        process_noise_covariance, measurement_noise_covariance, 
                        measurement, timespan, dynamics_args, measurement_args)      
            anterior_estimate, anterior_covariance, posterior_estimate, posterior_covariance, STM, innovations = iterate_EKF(*EKF_inputs)
        
        anterior_estimate_vals[:, time_index] = anterior_estimate
        posterior_estimate_vals[:, time_index] = posterior_estimate
        anterior_covariance_vals[:, :, time_index] = anterior_covariance
        posterior_covariance_vals[:, :, time_index] = posterior_covariance
        STM_vals[:, :, time_index] = STM
        # innovations_vals[:, time_index] = innovations

        previous_time = current_time
        previous_posterior_estimate = posterior_estimate
        previous_posterior_covariance = posterior_covariance

    return FilterResults(time_vals, anterior_estimate_vals, posterior_estimate_vals, anterior_covariance_vals, posterior_covariance_vals, STM_vals, innovations_vals)