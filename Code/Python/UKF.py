

import numpy as np
import scipy.integrate
from joblib import Parallel, delayed, parallel_config
from helper_functions import *

class UKF_Results:
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

class UKF_MCResults:
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

class UKF():
    def __init__(self,
                 dynamics_function,
                 dynamics_function_args,
                 measurement_function,
                 process_noise_covariance,
                 ukf_parameters: tuple,
                 underweighting_ratio = 1.
                 ):
        
        assert underweighting_ratio > 0
        assert underweighting_ratio <= 1

        self.dynamics_function = dynamics_function
        self.dynamics_function_args = dynamics_function_args
        self.measurement_function = measurement_function
        self.process_noise_covariance = process_noise_covariance
        self.alpha = ukf_parameters[0]
        self.beta = ukf_parameters[1]
        self.kappa = ukf_parameters[2]
        self.underweighting_ratio = underweighting_ratio
    
    def run(self, initial_estimate, initial_covariance, time_vals, measurement_vals, measurement_function_args):

        state_size = np.size(initial_estimate, 0)
        measurement_size = np.size(measurement_vals, 0)
        num_measurements = np.size(measurement_vals, 1)

        if self.underweighting_ratio < 1:
            measurement_update = self.underweighted_update
        else:
            measurement_update = self.measurement_update

        anterior_estimate_vals = np.full((state_size, num_measurements), np.nan)
        posterior_estimate_vals = np.full((state_size, num_measurements), np.nan)
        anterior_covariance_vals = np.full((state_size, state_size, num_measurements), np.nan)
        posterior_covariance_vals = np.full((state_size, state_size, num_measurements), np.nan)
        innovations_vals = np.full((measurement_size, num_measurements), np.nan)

        anterior_estimate_vals[:, 0] = initial_estimate
        anterior_covariance_vals[:, :, 0] = initial_covariance
        posterior_estimate_vals[:, 0] = initial_estimate
        posterior_covariance_vals[:, :, 0] = initial_covariance
        
        previous_time = 0
        previous_posterior_estimate = initial_estimate
        previous_posterior_covariance = initial_covariance

        for time_index in range(1, num_measurements):
            
            current_time = time_vals[time_index]
            timespan = current_time - previous_time
            
            current_measurement = measurement_vals[:, time_index]

            initial_sigma_points, weights = self.generate_sigma_points(previous_posterior_estimate, previous_posterior_covariance)
            propagated_sigma_points = self.time_update(time_index, initial_sigma_points, timespan)

            anterior_estimate, anterior_covariance = self.calculate_mean_covariance(propagated_sigma_points, weights)

            posterior_estimate, posterior_covariance = measurement_update(time_index, anterior_estimate, anterior_covariance, measurement_function_args, current_measurement)

            anterior_estimate_vals[:, time_index] = anterior_estimate
            anterior_covariance_vals[:, :, time_index] = anterior_covariance
            posterior_estimate_vals[:, time_index] = posterior_estimate
            posterior_covariance_vals[:, :, time_index] = posterior_covariance

            previous_posterior_estimate, previous_posterior_covariance = posterior_estimate, posterior_covariance
            previous_time = current_time
        
        return UKF_Results(time_vals, anterior_estimate_vals, posterior_estimate_vals, anterior_covariance_vals, posterior_covariance_vals, innovations_vals)
    
    def generate_sigma_points(self, mean, covariance):

        state_dim = len(mean)
        num_sigma_points = 2*state_dim + 1
        
        l = self.alpha**2 * (state_dim + self.kappa) - state_dim
        beta = self.beta
        cov_sqrt = scipy.linalg.sqrtm((state_dim + l)*covariance)

        sigma_points = np.empty((state_dim, num_sigma_points))
        weights = np.empty(num_sigma_points)
        sigma_points[:, 0] = mean
        weights[0] = l / (state_dim + l)
        weights[1:] = 1 / (2 * (state_dim + l))

        for sigma_point_index in range(1, num_sigma_points):
            if sigma_point_index % 2 == 1:
                sigma_points[:, sigma_point_index] = mean + cov_sqrt[int((sigma_point_index - 1)/2), :]
            else:
                sigma_points[:, sigma_point_index] = mean - cov_sqrt[int((sigma_point_index - 1)/2), :]

        return sigma_points, weights
    
    def calculate_mean_covariance(self, sigma_points, weights):
        
        num_sigma_points = len(weights)
        state_size = len(sigma_points[:, 0])
        
        mean = np.zeros(state_size)
        for sigma_point_index in range(num_sigma_points):
            mean += sigma_points[:, sigma_point_index] * weights[sigma_point_index]
        
        covariance = np.zeros((state_size, state_size))
        for sigma_point_index in range(num_sigma_points):
            difference = sigma_points[:, sigma_point_index] - mean
            covariance += weights[sigma_point_index] * (difference[:, None] @ difference[None, :])
        covariance += self.process_noise_covariance

        return mean, covariance

    def time_update(self, time_index, sigma_points, timespan):

        num_sigma_points = np.size(sigma_points, axis=1)
        propagated_sigma_points = np.empty(np.shape(sigma_points))

        for sigma_point_index in range(num_sigma_points):
            ICs = sigma_points[:, sigma_point_index]
            propagated_sigma_points[:, sigma_point_index] = scipy.integrate.solve_ivp(self.dynamics_function, [0,timespan], ICs, args=self.dynamics_function_args, atol=1e-12, rtol=1e-12).y[:, -1]

        return propagated_sigma_points
    
    # def unscented_measurement_update(self, time_index, propagated_sigma_points, weights, measurement_function_args, measurement):

    #     measurement = measurement[np.isnan(measurement) == False]
    #     measurement_size = len(measurement)
    #     if measurement_size == 0:
    #         return self.calculate_mean_covariance(propagated_sigma_points, weights)
        
    #     num_sigma_points = np.size(propagated_sigma_points, axis=1)
    #     sigma_measurements = np.empty((measurement_size, num_sigma_points))

    #     for sigma_point_index in range(num_sigma_points):
    #         sigma_measurements[:, sigma_point_index], measurement_noise_covariance = self.measurement_function(time_index, propagated_sigma_points[:, sigma_point_index], *measurement_function_args)

    #     return posterior_estimate, posterior_covariance

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
    
    def underweighted_update(self, time_index, anterior_estimate, anterior_covariance, measurement_function_args, measurement):

        measurement = measurement[np.isnan(measurement) == False]
        if len(measurement) == 0:
            return anterior_estimate, anterior_covariance

        predicted_measurement, measurement_jacobian, measurement_noise_covariance, rs = self.measurement_function(time_index, anterior_estimate, *measurement_function_args)

        if np.trace(measurement_jacobian @ np.linalg.inv(anterior_covariance) @ measurement_jacobian.T) < (1 - self.underweighting_ratio)/self.underweighting_ratio * np.trace(np.linalg.inv(measurement_noise_covariance)):
            innovations_covariance = measurement_jacobian @ anterior_covariance @ measurement_jacobian.T/self.underweighting_ratio + measurement_noise_covariance
        else:
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
    
    def run_MC(self, initial_estimates, initial_covariance, time_vals, measurements, measurement_args):

        num_runs = len(initial_estimates)
        results = []

        with parallel_config(verbose=100, n_jobs=-1):
            results = Parallel()(delayed(self.run)(initial_estimates[run_index], initial_covariance, time_vals, measurements[run_index], measurement_args[run_index]) for run_index in range(num_runs))
        
        return UKF_MCResults(results)
