

import numpy as np
import scipy

class FilterResults:
    def __init__(self, time_vals, anterior_estimate_vals, posterior_estimate_vals, anterior_covariance_vals, posterior_covariance_vals):
        self.t = time_vals
        self.anterior_estimate_vals = anterior_estimate_vals
        self.posterior_estimate_vals = posterior_estimate_vals
        self.anterior_covariance_vals = anterior_covariance_vals
        self.posterior_covariance_vals = posterior_covariance_vals

class Measurements:
    def __init__(self, time_vals, measurement_vals):     
        assert np.size(time_vals, 0) == np.size(measurement_vals, 1)

        self.t = time_vals
        self.measurements = measurement_vals

def EKF(previous_posterior_estimate, previous_posterior_covariance, dynamics_equation, measurement_equation, measurement):

    return posterior_estimate, posterior_covariance

def run_EKF(initial_estimate, initial_covariance, dynamics_equation, measurement_equation, measurements: Measurements):

    time_vals = measurements.t
    measurement_vals = measurements.measurements

    state_size = np.size(initial_estimate, 0)
    measurement_size = np.size(measurement_vals, 0)
    num_measurements = np.size(measurement_vals, 1)

    anterior_estimate_vals = np.zeros((state_size, num_measurements))
    posterior_estimate_vals = np.zeros((state_size, num_measurements))
    anterior_covariance_vals = np.zeros((state_size, state_size, num_measurements))
    posterior_covariance_vals = np.zeros((state_size, state_size, num_measurements))

    previous_posterior_estimate = initial_estimate
    previous_posterior_covariance = initial_covariance

    for time_index in np.arange(num_measurements):
        
        integration_inputs = (dynamics_equation, )
        integration_settings = {"rtol":1e-12, "atol":1e-12}

        




    return 