

import numpy as np
import scipy.integrate
from joblib import Parallel, delayed, parallel_config
from helper_functions import *

class EKF_Results:
    def __init__(self, 
                 time_vals, 
                 anterior_estimate_vals, 
                 posterior_estimate_vals, 
                 anterior_covariance_vals, 
                 posterior_covariance_vals, 
                 innovations_vals):
        self.t = time_vals
        self.anterior_estimate_vals = anterior_estimate_vals
        self.posterior_estimate_vals = posterior_estimate_vals
        self.anterior_covariance_vals = anterior_covariance_vals
        self.posterior_covariance_vals = posterior_covariance_vals
        self.innovations_vals = innovations_vals

class EKF_MCResults:
    def __init__(self, EKF_results):

        num_results = len(EKF_results)
        anterior_estimates = []
        posterior_estimates = []
        anterior_covariances = []
        posterior_covariances = []
        innovations = []
        for result in EKF_results:
            anterior_estimates.append(result.anterior_estimate_vals)
            posterior_estimates.append(result.posterior_estimate_vals)
            anterior_covariances.append(result.anterior_covariance_vals)
            posterior_covariances.append(result.posterior_covariance_vals)
            innovations.append(result.innovations_vals)
        
        self.t = result.t
        self.anterior_estimates = anterior_estimates
        self.posterior_estimates = posterior_estimates
        self.anterior_covariances = anterior_covariances
        self.posterior_covariances = posterior_covariances
        self.innovations = innovations

class EKF():
    def __init__(self,
                 dynamics_function,
                 dynamics_function_args,
                 measurement_function,
                 process_noise_covariance
                 ):
        
        self.dynamics_function = dynamics_function
        self.dynamics_function_args = dynamics_function_args
        self.measurement_function = measurement_function
        self.process_noise_covariance = process_noise_covariance
    
    def run(self, initial_estimate, initial_covariance, time_vals, measurement_vals, measurement_function_args):

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

        posterior_estimate, posterior_covariance = self.measurement_update(0, initial_estimate, initial_covariance, measurement_function_args, measurement_vals[:, 0])
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

            posterior_estimate, posterior_covariance = self.measurement_update(time_index, anterior_estimate, anterior_covariance, measurement_function_args, current_measurement)

            anterior_estimate_vals[:, time_index] = anterior_estimate
            anterior_covariance_vals[:, :, time_index] = anterior_covariance
            posterior_estimate_vals[:, time_index] = posterior_estimate
            posterior_covariance_vals[:, :, time_index] = posterior_covariance

            previous_posterior_estimate, previous_posterior_covariance = posterior_estimate, posterior_covariance
            previous_time = current_time
        
        return EKF_Results(time_vals, anterior_estimate_vals, posterior_estimate_vals, anterior_covariance_vals, posterior_covariance_vals, innovations_vals)

    def measurement_update(self, time_index, anterior_estimate, anterior_covariance, measurement_function_args, measurement):

        measurement = measurement[np.isnan(measurement) == False]
        if len(measurement) == 0:
            return anterior_estimate, anterior_covariance

        predicted_measurement, measurement_jacobian, measurement_noise_covariance, rs = self.measurement_function(time_index, anterior_estimate, *measurement_function_args)

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
    
    def run_MC(self, initial_estimates, initial_covariance, time_vals, measurements, measurement_args):

        num_runs = len(initial_estimates)
        results = []

        with parallel_config(verbose=100, n_jobs=-1):
            results = Parallel()(delayed(self.run)(initial_estimates[run_index], initial_covariance, time_vals, measurements[run_index], measurement_args[run_index]) for run_index in range(num_runs))
        
        return EKF_MCResults(results)
