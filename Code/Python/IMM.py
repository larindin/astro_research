
import numpy as np
from helper_functions import *
from EKF import *
from GM_EKF import *

class IMM_FilterResults:
    def __init__(self, 
                 time_vals, 
                 anterior_estimate_vals, 
                 posterior_estimate_vals, 
                 output_estimate_vals,
                 anterior_covariance_vals, 
                 posterior_covariance_vals, 
                 output_covariance_vals,
                 innovations_vals, 
                 mode_probability_vals):
        self.t = time_vals
        self.anterior_estimate_vals = anterior_estimate_vals
        self.posterior_estimate_vals = posterior_estimate_vals
        self.output_estimate_vals = output_estimate_vals
        self.anterior_covariance_vals = anterior_covariance_vals
        self.posterior_covariance_vals = posterior_covariance_vals
        self.output_covariance_vals = output_covariance_vals
        self.innovations_vals = innovations_vals
        self.mode_probability_vals = mode_probability_vals

class IMM_filter():

    def __init__(self,
                 dynamics_functions,
                 dynamics_functions_args,
                 measurement_function,
                 measurement_function_args,
                 process_noise_covariances,
                 mode_transition_matrix
                 ):
        
        self.dynamics_functions = dynamics_functions
        self.dynamics_functions_args = dynamics_functions_args
        self.measurement_function = measurement_function
        self.measurement_function_args = measurement_function_args
        self.process_noise_covariances = process_noise_covariances
        self.mode_transition_matrix = mode_transition_matrix
    
    def run(self, initial_estimate, initial_covariance, initial_mode_probabilities, time_vals, measurement_vals):

        state_size = np.size(initial_estimate, 0)
        num_modes = np.size(initial_mode_probabilities, 0)
        measurement_size = np.size(measurement_vals, 0)
        num_measurements = np.size(measurement_vals, 1)

        anterior_estimate_vals = np.full((state_size, num_measurements, num_modes), np.nan)
        posterior_estimate_vals = np.full((state_size, num_measurements, num_modes), np.nan)
        output_estimate_vals = np.full((state_size, num_measurements), np.nan)
        anterior_covariance_vals = np.full((state_size, state_size, num_measurements, num_modes), np.nan)
        posterior_covariance_vals = np.full((state_size, state_size, num_measurements, num_modes), np.nan)
        output_covariance_vals = np.full((state_size, state_size, num_measurements), np.nan)
        innovations_vals = np.full((measurement_size, num_measurements, num_modes), np.nan)
        mode_probability_vals = np.full((num_modes, num_measurements), np.nan)

        denominators = np.empty(num_modes)
        exponents = np.empty(num_modes)

        mode_probability_vals[:, 0] = initial_mode_probabilities
        for mode_index in range(num_modes):
            anterior_estimate_vals[:, 0, mode_index] = initial_estimate
            anterior_covariance_vals[:, :, 0, mode_index] = initial_covariance

        for mode_index in range(num_modes):
            posterior_estimate, posterior_covariance, denominator, exponent  = self.measurement_update(0, initial_estimate, initial_covariance, measurement_vals[:, 0])
            posterior_estimate_vals[:, 0, mode_index] = posterior_estimate
            posterior_covariance_vals[:, :, 0, mode_index] = posterior_covariance
            denominators[mode_index], exponents[mode_index] = denominator, exponent
        
        mode_probability_vals[:, 0] = self.mode_probability_update(initial_mode_probabilities, denominators, exponents)
        output_estimate, output_covariance = self.mixed_outputs(mode_probability_vals[:, 0], posterior_estimate_vals[:, 0, :], posterior_covariance_vals[:, :, 0, :])
        output_estimate_vals[:, 0], output_covariance_vals[:, :, 0] = output_estimate, output_covariance

        previous_time = 0
        previous_posterior_estimates = posterior_estimate_vals[:, 0, :]
        previous_posterior_covariances = posterior_covariance_vals[:, :, 0, :]
        previous_mode_probabilities = mode_probability_vals[:, 0]

        original_umax = self.dynamics_functions_args[1][1]

        for time_index in range(1, num_measurements):
            
            current_time = time_vals[time_index]
            timespan = current_time - previous_time

            self.dynamics_functions_args[1][1] = original_umax * previous_mode_probabilities[1]
            
            current_measurement = measurement_vals[:, time_index]

            for mode_index in range(num_modes):

                mixed_state, mixed_covariance = self.mixed_initial_conditions(mode_index, previous_mode_probabilities, previous_posterior_estimates, previous_posterior_covariances)

                anterior_estimate, anterior_covariance = self.time_update(time_index, mixed_state, mixed_covariance, timespan, mode_index)

                posterior_estimate, posterior_covariance, denominator, exponent = self.measurement_update(time_index, anterior_estimate, anterior_covariance, current_measurement)

                anterior_estimate_vals[:, time_index, mode_index] = anterior_estimate
                anterior_covariance_vals[:, :, time_index, mode_index] = anterior_covariance
                posterior_estimate_vals[:, time_index, mode_index] = posterior_estimate
                posterior_covariance_vals[:, :, time_index, mode_index] = posterior_covariance
                denominators[mode_index] = denominator
                exponents[mode_index] = exponent
            
            new_mode_probabilities = self.mode_probability_update(previous_mode_probabilities, denominators, exponents)
            mode_probability_vals[:, time_index] = new_mode_probabilities

            output_estimate, output_covariance = self.mixed_outputs(new_mode_probabilities, posterior_estimate_vals[:, time_index, :], posterior_covariance_vals[:, :, time_index, :])
            output_estimate_vals[:, time_index] = output_estimate
            output_covariance_vals[:, :, time_index] = output_covariance

            previous_posterior_estimates, previous_posterior_covariances = posterior_estimate_vals[:, time_index, :], posterior_covariance_vals[:, :, time_index, :]
            previous_mode_probabilities = new_mode_probabilities
            previous_time = current_time
        
        return IMM_FilterResults(time_vals, anterior_estimate_vals, posterior_estimate_vals, output_estimate_vals, anterior_covariance_vals, posterior_covariance_vals, output_covariance_vals, 0, mode_probability_vals)

    def measurement_update(self, time_index, anterior_estimate, anterior_covariance, measurement):

        measurement = measurement[np.isnan(measurement) == False]
        if len(measurement) == 0:
            return anterior_estimate, anterior_covariance, 1, 1
        
        posterior_estimate = np.full(len(anterior_estimate), np.nan)
        posterior_covariance = np.full(np.shape(anterior_covariance), np.nan)

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
        posterior_covariance= enforce_symmetry(anterior_covariance - cross_covariance @ gain_matrix.T - gain_matrix @ cross_covariance.T + gain_matrix @ innovations_covariance @ gain_matrix.T)

        denominator, exponent = assess_measurement_likelihood(innovations, innovations_covariance)

        return posterior_estimate, posterior_covariance, denominator, exponent

    def time_update(self, time_index, mixed_initial_conditions, mixed_initial_covariance, timespan, mode_index):

        state_size = len(mixed_initial_conditions)
        dynamics_eq = self.dynamics_functions[mode_index]
        args = self.dynamics_functions_args[mode_index]

        ICs = np.concatenate((mixed_initial_conditions, np.eye(state_size).flatten()))
        propagation = scipy.integrate.solve_ivp(dynamics_eq, [0,timespan], ICs, args=args, atol=1e-12, rtol=1e-12).y[:, -1]

        STM = propagation[state_size:state_size**2 + state_size].reshape((state_size, state_size))
        anterior_estimate = propagation[0:state_size]
        anterior_covariance = enforce_symmetry(STM @ mixed_initial_covariance @ STM.T + self.process_noise_covariances[mode_index])

        return anterior_estimate, anterior_covariance
    
    def mode_probability_update(self, previous_mode_probabilities, denominators, exponents):
        
        num_modes = len(denominators)
        new_mode_probabilities = np.empty(num_modes)
        normalized_denominators = denominators / denominators.min()
        normalized_exponents = exponents - exponents.max()

        anterior_mode_probabilities = np.empty(num_modes)
        for mode_index in range(num_modes):
            anterior_mode_probabilities[mode_index] = np.sum(previous_mode_probabilities * self.mode_transition_matrix[:, mode_index])
        
        measurement_likelihoods = 1 / normalized_denominators * np.exp(normalized_exponents)
        raw_mode_probabilities = measurement_likelihoods*anterior_mode_probabilities
        new_mode_probabilities = raw_mode_probabilities / np.sum(raw_mode_probabilities)
        
        return new_mode_probabilities
    
    def mixed_initial_conditions(self, mode, previous_mode_probabilities, previous_posterior_estimates, previous_posterior_covariances):
        
        raw_mixing_proportions = previous_mode_probabilities * self.mode_transition_matrix[:, mode]
        anterior_mode_probability = np.sum(previous_mode_probabilities * self.mode_transition_matrix[:, mode])
        mixing_proportions = raw_mixing_proportions / anterior_mode_probability

        num_modes = len(previous_mode_probabilities)
        mixed_state = np.zeros(previous_posterior_estimates[:, 0].shape)
        mixed_covariance = np.zeros(previous_posterior_covariances[:, :, 0].shape)

        for mode_index in range(num_modes):
            mixed_state += mixing_proportions[mode_index] * previous_posterior_estimates[:, mode_index]
        for mode_index in range(num_modes):
            difference = (previous_posterior_estimates[:, mode_index] - mixed_state)[:, None]
            mixed_covariance += mixing_proportions[mode_index] * (previous_posterior_covariances[:, :, mode_index] + difference @ difference.T)
        
        return mixed_state, mixed_covariance
    
    def mixed_outputs(self, mode_probabilities, posterior_estimates, posterior_covariances):

        num_modes = len(mode_probabilities)
        mixed_state = np.zeros(posterior_estimates[:, 0].shape)
        mixed_covariance = np.zeros(posterior_covariances[:, :, 0].shape)

        for mode_index in range(num_modes):
            mixed_state += mode_probabilities[mode_index] * posterior_estimates[:, mode_index]
        for mode_index in range(num_modes):
            difference = (posterior_estimates[:, mode_index] - mixed_state)[:, None]
            mixed_covariance += mode_probabilities[mode_index] * (posterior_covariances[:, :, mode_index] + difference @ difference.T)
        
        return mixed_state, mixed_covariance

def get_thrusting_indices(time_vals, mode_probability_vals, switching_cutoff):

    thrusting_boolean_array = mode_probability_vals[1, :] > 0.5

    thrusting_begins = False
    for time_index in range(len(time_vals)):
        
        if thrusting_begins == False and np.array_equal(thrusting_boolean_array[time_index:time_index+switching_cutoff], np.ones(switching_cutoff)):
            thrusting_begins = True
            thrust_start_index = time_index

        if thrusting_begins == True and np.array_equal(thrusting_boolean_array[time_index:time_index+switching_cutoff], np.zeros(switching_cutoff)):
            thrust_end_index = time_index
            break

    return np.array([thrust_start_index, thrust_end_index])