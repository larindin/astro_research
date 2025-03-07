
import numpy as np
import scipy.integrate
from helper_functions import *
from EKF import *
from GM_EKF import *

class VSD_filter():

    def __init__(self,
                 quiescent_ODE,
                 quiescent_ODE_args,
                 quiescent_size,
                 maneuvering_ODE,
                 maneuvering_ODE_args,
                 maneuvering_size,
                 measurement_function,
                 measurement_function_args,
                 process_noise_covariances,
                 q2m_initialization,
                 memory_factor,
                 activation_significance
                 ):
        
        self.quiescent_ODE = quiescent_ODE
        self.quiescent_ODE_args = quiescent_ODE_args
        self.quiescent_size = quiescent_size
        self.maneuvering_ODE = maneuvering_ODE
        self.maneuvering_ODE_args = maneuvering_ODE_args
        self.maneuvering_size = maneuvering_size
        self.measurement_function = measurement_function
        self.measurement_function_args = measurement_function_args
        self.process_noise_covariances = process_noise_covariances
        self.q2m_initialization = q2m_initialization
        self.memory_factor = memory_factor
        self.activation_significance = activation_significance
    
    def run(self, initial_estimate, initial_covariance, time_vals, measurement_vals):
        
        quiescent_size = self.quiescent_size
        maneuvering_size = self.maneuvering_size
        memory_factor = self.memory_factor

        measurement_size = np.size(measurement_vals, 0)
        num_measurements = len(time_vals)
        window = round(1/ (1 - memory_factor))

        chi2_cutoff = get_chi2_cutoff(measurement_size*window, self.activation_significance)

        anterior_estimate_vals = np.full((maneuvering_size, num_measurements), np.nan)
        posterior_estimate_vals = np.full((maneuvering_size, num_measurements), np.nan)
        anterior_covariance_vals = np.full((maneuvering_size, maneuvering_size, num_measurements), np.nan)
        posterior_covariance_vals = np.full((maneuvering_size, maneuvering_size, num_measurements), np.nan)
        innovations_vals = np.full((measurement_size, num_measurements), np.nan)

        anterior_estimate_vals[0:quiescent_size, 0] = initial_estimate
        anterior_covariance_vals[0:quiescent_size, 0:quiescent_size, 0] = initial_covariance

        posterior_estimate, posterior_covariance, activation_metric = self.measurement_update(0, anterior_estimate_vals[:, 0], anterior_covariance_vals[:, :, 0], measurement_vals[:, 0], 0)
        posterior_estimate_vals[:, 0] = posterior_estimate
        posterior_covariance_vals[:, :, 0] = posterior_covariance

        previous_time = time_vals[0]
        previous_posterior_estimate = posterior_estimate
        previous_posterior_covariance = posterior_covariance
        active_filter = 0

        for time_index in range(1, num_measurements):
            
            current_time = time_vals[time_index]
            timespan = current_time - previous_time
            
            current_measurement = measurement_vals[:, time_index]

            anterior_estimate, anterior_covariance = self.time_update(time_index, previous_posterior_estimate, previous_posterior_covariance, timespan, active_filter)

            posterior_estimate, posterior_covariance, innovations_distance = self.measurement_update(time_index, anterior_estimate, anterior_covariance, current_measurement, active_filter)

            activation_metric = innovations_distance + memory_factor*activation_metric

            if activation_metric > chi2_cutoff and active_filter == 0:
                active_filter = 1
                activation_metric = 0
                print("maneuvering ", current_time)

                previous_posterior_estimate = posterior_estimate_vals[:, time_index-window-1]
                previous_posterior_covariance = posterior_covariance_vals[:, :, time_index-window-1]
                previous_time = time_vals[time_index-window-1]

                previous_posterior_estimate, previous_posterior_covariance = self.q2m_initialization(previous_posterior_estimate, previous_posterior_covariance)

                for repair_index in range(time_index-window-1, time_index+1):
                    
                    current_time = time_vals[repair_index]
                    timespan = current_time - previous_time

                    current_measurement = measurement_vals[:, repair_index]

                    anterior_estimate, anterior_covariance = self.time_update(repair_index, previous_posterior_estimate, previous_posterior_covariance, timespan, active_filter)
                    
                    posterior_estimate, posterior_covariance, innovations_distance = self.measurement_update(repair_index, anterior_estimate, anterior_covariance, current_measurement, active_filter)

                    anterior_estimate_vals[:, repair_index] = anterior_estimate
                    anterior_covariance_vals[:, :, repair_index] = anterior_covariance
                    posterior_estimate_vals[:, repair_index] = posterior_estimate
                    posterior_covariance_vals[:, :, repair_index] = posterior_covariance
            
            elif activation_metric > chi2_cutoff and active_filter == 1:
                active_filter = 0
                activation_metric = 0
                print("quiescent ", current_time)

            anterior_estimate_vals[:, time_index] = anterior_estimate
            anterior_covariance_vals[:, :, time_index] = anterior_covariance
            posterior_estimate_vals[:, time_index] = posterior_estimate
            posterior_covariance_vals[:, :, time_index] = posterior_covariance

            previous_posterior_estimate, previous_posterior_covariance = posterior_estimate, posterior_covariance
            
            previous_time = current_time
        
        return FilterResults(time_vals, anterior_estimate_vals, posterior_estimate_vals, anterior_covariance_vals, posterior_covariance_vals, 0, 0)

    def measurement_update(self, time_index, anterior_estimate, anterior_covariance, measurement, active_filter):

        measurement = measurement[np.isnan(measurement) == False]
        posterior_estimate = np.full(len(anterior_estimate), np.nan)
        posterior_covariance = np.full(np.shape(anterior_covariance), np.nan)

        # predicted_measurement, measurement_jacobian, measurement_noise_covariance = self.measurement_function(time_index, anterior_estimate, *self.measurement_function_args)
        predicted_measurement, measurement_jacobian, measurement_noise_covariance, rs = self.measurement_function(time_index, anterior_estimate, *self.measurement_function_args)

        if active_filter == 0:
            size = self.quiescent_size
            anterior_covariance = anterior_covariance[0:size, 0:size]
            anterior_estimate = anterior_estimate[0:size]
        elif active_filter == 1:
            size = self.maneuvering_size

        innovations_covariance = measurement_jacobian @ anterior_covariance @ measurement_jacobian.T + measurement_noise_covariance
        innovations_covariance = enforce_symmetry(innovations_covariance)
        cross_covariance = anterior_covariance @ measurement_jacobian.T
        gain_matrix = cross_covariance @ np.linalg.inv(innovations_covariance)

        innovations = measurement - predicted_measurement
        for sensor_index, r in enumerate(rs):
            innovations[sensor_index*3:(sensor_index+1)*3]*= r
        # innovations = check_innovations(innovations)

        posterior_estimate[0:size] = anterior_estimate + gain_matrix @ innovations
        posterior_covariance [0:size, 0:size]= enforce_symmetry(anterior_covariance - cross_covariance @ gain_matrix.T - gain_matrix @ cross_covariance.T + gain_matrix @ innovations_covariance @ gain_matrix.T)

        innovations_distance = innovations.T @ np.linalg.inv(innovations_covariance) @ innovations

        return posterior_estimate, posterior_covariance, innovations_distance

    def time_update(self, time_index, posterior_estimate, posterior_covariance, timespan, active_filter):

        anterior_estimate = np.full(len(posterior_estimate), np.nan)
        anterior_covariance = np.full(np.shape(posterior_covariance), np.nan)

        if active_filter == 0:
            size = self.quiescent_size
            
            ICs = np.concatenate((posterior_estimate[0:size], np.eye(size).flatten()))
            propagation = scipy.integrate.solve_ivp(self.quiescent_ODE, [0,timespan], ICs, args=self.quiescent_ODE_args, atol=1e-12, rtol=1e-12).y[:, -1]
            
            posterior_covariance = posterior_covariance[0:size, 0:size]

        elif active_filter == 1:
            size = self.maneuvering_size

            ICs = np.concatenate((posterior_estimate, np.eye(size).flatten()))
            propagation = scipy.integrate.solve_ivp(self.maneuvering_ODE, [0,timespan], ICs, args=self.maneuvering_ODE_args, atol=1e-12, rtol=1e-12).y[:, -1]

        STM = propagation[size:size**2 + size].reshape((size, size))
        anterior_estimate[0:size] = propagation[0:size]
        anterior_covariance[0:size, 0:size] = enforce_symmetry(STM @ posterior_covariance @ STM.T + self.process_noise_covariances[active_filter])

        return anterior_estimate, anterior_covariance

