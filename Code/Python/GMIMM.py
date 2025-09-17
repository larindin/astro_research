
import numpy as np
from joblib import Parallel, delayed, parallel_config
from helper_functions import *
from EKF import *
from GM_EKF import *

class GMIMM_FilterResults:
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

class GMIMM_MCResults:
    def __init__(self, GMIMM_results):

        num_results = len(GMIMM_results)
        anterior_estimates = []
        posterior_estimates = []
        output_estimates = []
        anterior_covariances = []
        posterior_covariances = []
        output_covariances = []
        innovations = []
        mode_probabilities = []
        for result in GMIMM_results:
            anterior_estimates.append(result.anterior_estimate_vals)
            posterior_estimates.append(result.posterior_estimate_vals)
            output_estimates.append(result.output_estimate_vals)
            anterior_covariances.append(result.anterior_covariance_vals)
            posterior_covariances.append(result.posterior_covariance_vals)
            output_covariances.append(result.output_covariance_vals)
            innovations.append(result.innovations_vals)
            mode_probabilities.append(result.mode_probability_vals)
        
        self.t = result.t
        self.anterior_estimates = anterior_estimates
        self.posterior_estimates = posterior_estimates
        self.output_estimates = output_estimates
        self.anterior_covariances = anterior_covariances
        self.posterior_covariances = posterior_covariances
        self.output_covariances = output_covariances
        self.innovations = innovations
        self.mode_probabilities = mode_probabilities


class GMIMM_filter():

    def __init__(self,
                 dynamics_functions,
                 dynamics_functions_args,
                 measurement_function,
                 process_noise_covariances,
                 mode_transition_matrix,
                 underweighting_ratio = 1.,
                 kernels_per_mode = 2,
                 ):
        
        assert underweighting_ratio > 0 and underweighting_ratio <= 1
        assert type(kernels_per_mode) == int

        self.dynamics_functions = dynamics_functions
        self.dynamics_functions_args = dynamics_functions_args
        self.measurement_function = measurement_function
        self.process_noise_covariances = process_noise_covariances
        self.mode_transition_matrix = mode_transition_matrix
        self.underweighting_ratio = underweighting_ratio
        self.kernels_per_mode = kernels_per_mode
    
    def run(self, initial_estimates, initial_covariances, initial_kernel_probabilities, time_vals, measurement_vals, measurement_function_args):

        state_size = np.size(initial_estimates, 0)
        num_modes = len(self.dynamics_functions)
        kernels_per_mode = self.kernels_per_mode
        measurement_size = np.size(measurement_vals, 0)
        num_measurements = np.size(measurement_vals, 1)

        anterior_estimate_vals = np.full((state_size, num_measurements, num_modes, kernels_per_mode), np.nan)
        posterior_estimate_vals = np.full((state_size, num_measurements, num_modes, kernels_per_mode), np.nan)
        output_estimate_vals = np.full((state_size, num_measurements), np.nan)
        anterior_covariance_vals = np.full((state_size, state_size, num_measurements, num_modes, kernels_per_mode), np.nan)
        posterior_covariance_vals = np.full((state_size, state_size, num_measurements, num_modes, kernels_per_mode), np.nan)
        output_covariance_vals = np.full((state_size, state_size, num_measurements), np.nan)
        innovations_vals = np.full((measurement_size, num_measurements, num_modes, kernels_per_mode), np.nan)
        kernel_probabilitiy_vals = np.full((num_modes, kernels_per_mode, num_measurements), np.nan)

        denominators = np.empty((num_modes, kernels_per_mode))
        exponents = np.empty((num_modes, kernels_per_mode))
        anterior_kernel_probabilities = np.empty((num_modes, kernels_per_mode))

        if self.underweighting_ratio < 1:
            measurement_update = self.underweighted_update
            # measurement_update = self.constrained_measurement_update
        else:
            measurement_update = self.measurement_update

        kernel_probabilitiy_vals[:, :, 0] = initial_kernel_probabilities
        for mode_index in range(num_modes):
            anterior_estimate_vals[:, 0, mode_index, :] = initial_estimates[:, mode_index, :]
            anterior_covariance_vals[:, :, 0, mode_index, :] = initial_covariances[:, :, mode_index, :]

        for mode_index in range(num_modes):
            for kernel_index in range(kernels_per_mode):
                posterior_estimate, posterior_covariance, denominator, exponent = measurement_update(0, initial_estimates[:, mode_index, kernel_index], initial_covariances[:, :, mode_index, kernel_index], measurement_function_args, measurement_vals[:, 0])
                posterior_estimate_vals[:, 0, mode_index, kernel_index] = posterior_estimate
                posterior_covariance_vals[:, :, 0, mode_index, kernel_index] = posterior_covariance
                denominators[mode_index, kernel_index], exponents[mode_index, kernel_index] = denominator, exponent
        
        kernel_probabilitiy_vals[:, :, 0] = self.kernel_probability_update(initial_kernel_probabilities, denominators, exponents, measurement_vals[:, 0])
        output_estimate, output_covariance = self.mixed_outputs(kernel_probabilitiy_vals[:, :, 0], posterior_estimate_vals[:, 0, :, :], posterior_covariance_vals[:, :, 0, :, :])
        output_estimate_vals[:, 0], output_covariance_vals[:, :, 0] = output_estimate, output_covariance

        previous_time = time_vals[0]
        previous_posterior_estimates = posterior_estimate_vals[:, 0, :, :]
        previous_posterior_covariances = posterior_covariance_vals[:, :, 0, :, :]
        previous_kernel_probabilities = kernel_probabilitiy_vals[:, :, 0]

        for time_index in range(1, num_measurements):
            
            current_time = time_vals[time_index]
            timespan = current_time - previous_time
            
            current_measurement = measurement_vals[:, time_index]

            for mode_index in range(num_modes):

                initial_states, initial_covariances, mode_anterior_kernel_probabilities = self.mixed_initial_conditions(mode_index, previous_kernel_probabilities, previous_posterior_estimates, previous_posterior_covariances)
                
                for kernel_index in range(kernels_per_mode):
                    initial_state, initial_covariance = initial_states[:, kernel_index], initial_covariances[:, :, kernel_index]
                    anterior_estimate, anterior_covariance = self.time_update(time_index, initial_state, initial_covariance, timespan, mode_index, previous_mode_probabilities[mode_index])

                    posterior_estimate, posterior_covariance, denominator, exponent = measurement_update(time_index, anterior_estimate, anterior_covariance, measurement_function_args, current_measurement)

                    anterior_estimate_vals[:, time_index, mode_index, kernel_index] = anterior_estimate
                    anterior_covariance_vals[:, :, time_index, mode_index, kernel_index] = anterior_covariance
                    posterior_estimate_vals[:, time_index, mode_index, kernel_index] = posterior_estimate
                    posterior_covariance_vals[:, :, time_index, mode_index, kernel_index] = posterior_covariance
                    denominators[mode_index, kernel_index] = denominator
                    exponents[mode_index, kernel_index] = exponent
                anterior_kernel_probabilities[mode_index, :] = mode_anterior_kernel_probabilities
            
            new_mode_probabilities = self.kernel_probability_update(anterior_kernel_probabilities, denominators, exponents, current_measurement)
            mode_probability_vals[:, time_index] = new_mode_probabilities

            output_estimate, output_covariance = self.mixed_outputs(new_mode_probabilities, posterior_estimate_vals[:, time_index, :], posterior_covariance_vals[:, :, time_index, :])
            output_estimate_vals[:, time_index] = output_estimate
            output_covariance_vals[:, :, time_index] = output_covariance

            previous_posterior_estimates, previous_posterior_covariances = posterior_estimate_vals[:, time_index, :], posterior_covariance_vals[:, :, time_index, :]
            previous_mode_probabilities = new_mode_probabilities
            previous_time = current_time
        
        return GMIMM_FilterResults(time_vals, anterior_estimate_vals, posterior_estimate_vals, output_estimate_vals, anterior_covariance_vals, posterior_covariance_vals, output_covariance_vals, 0, mode_probability_vals)

    def measurement_update(self, time_index, anterior_estimate, anterior_covariance, measurement_function_args, measurement):

        measurement = measurement[np.isnan(measurement) == False]
        if len(measurement) == 0:
            return anterior_estimate, anterior_covariance, 1, 1
        
        posterior_estimate = np.full(len(anterior_estimate), np.nan)
        posterior_covariance = np.full(np.shape(anterior_covariance), np.nan)

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
        posterior_covariance= enforce_symmetry(anterior_covariance - cross_covariance @ gain_matrix.T - gain_matrix @ cross_covariance.T + gain_matrix @ innovations_covariance @ gain_matrix.T)

        denominator, exponent = assess_measurement_likelihood(innovations, innovations_covariance)

        return posterior_estimate, posterior_covariance, denominator, exponent
    
    # def constrained_measurement_update(self, time_index, anterior_estimate, anterior_covariance, measurement_function_args, measurement):

    #     measurement = measurement[np.isnan(measurement) == False]
    #     if len(measurement) == 0:
    #         return anterior_estimate, anterior_covariance, 1, 1
        
    #     posterior_estimate = np.full(len(anterior_estimate), np.nan)
    #     posterior_covariance = np.full(np.shape(anterior_covariance), np.nan)

    #     costate_norm = np.linalg.norm(anterior_estimate[6:12])

    #     predicted_measurement, measurement_jacobian, measurement_noise_covariance, rs = self.measurement_function(time_index, anterior_estimate, *measurement_function_args)

    #     if np.trace(measurement_jacobian @ np.linalg.inv(anterior_covariance) @ measurement_jacobian.T) < (1 - self.underweighting_ratio)/self.underweighting_ratio * np.trace(np.linalg.inv(measurement_noise_covariance)):
    #         innovations_covariance = measurement_jacobian @ anterior_covariance @ measurement_jacobian.T/self.underweighting_ratio + measurement_noise_covariance
    #         # print(f"{time_index/24} underweighting")
    #     else:
    #         innovations_covariance = measurement_jacobian @ anterior_covariance @ measurement_jacobian.T + measurement_noise_covariance
    #     innovations_covariance = enforce_symmetry(innovations_covariance)
    #     inverse_innovations_cov = np.linalg.inv(innovations_covariance)

    #     anterior_state_covariance = anterior_covariance[:, 0:6]
    #     anterior_costate_covariance = anterior_covariance[:, 6:12]

    #     state_gain_matrix = anterior_state_covariance.T @ measurement_jacobian.T @ inverse_innovations_cov
    #     unconstrained_costate_gain_matrix = anterior_costate_covariance.T @ measurement_jacobian.T @ inverse_innovations_cov



    #     innovations = measurement - predicted_measurement
    #     # innovations = check_innovations(innovations)
    #     for sensor_index, r in enumerate(rs):
    #         innovations[sensor_index*3:(sensor_index+1)*3]*= r

    #     posterior_estimate = anterior_estimate + gain_matrix @ innovations
    #     posterior_covariance= enforce_symmetry(anterior_covariance - cross_covariance @ gain_matrix.T - gain_matrix @ cross_covariance.T + gain_matrix @ innovations_covariance @ gain_matrix.T)
 
    #     denominator, exponent = assess_measurement_likelihood(innovations, innovations_covariance)

    #     return posterior_estimate, posterior_covariance, denominator, exponent

    
    def underweighted_update(self, time_index, anterior_estimate, anterior_covariance, measurement_function_args, measurement):

        measurement = measurement[np.isnan(measurement) == False]
        if len(measurement) == 0:
            return anterior_estimate, anterior_covariance, 1, 1

        posterior_estimate = np.full(len(anterior_estimate), np.nan)
        posterior_covariance = np.full(np.shape(anterior_covariance), np.nan)

        predicted_measurement, measurement_jacobian, measurement_noise_covariance, rs = self.measurement_function(time_index, anterior_estimate, *measurement_function_args)

        if np.trace(measurement_jacobian @ np.linalg.inv(anterior_covariance) @ measurement_jacobian.T) < (1 - self.underweighting_ratio)/self.underweighting_ratio * np.trace(np.linalg.inv(measurement_noise_covariance)):
            innovations_covariance = measurement_jacobian @ anterior_covariance @ measurement_jacobian.T/self.underweighting_ratio + measurement_noise_covariance
            # print(f"{time_index/24} underweighting")
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
        posterior_covariance= enforce_symmetry(anterior_covariance - cross_covariance @ gain_matrix.T - gain_matrix @ cross_covariance.T + gain_matrix @ innovations_covariance @ gain_matrix.T)

        denominator, exponent = assess_measurement_likelihood(innovations, innovations_covariance)

        return posterior_estimate, posterior_covariance, denominator, exponent

    def time_update(self, time_index, mixed_initial_conditions, mixed_initial_covariance, timespan, mode_index, mode_probability):

        state_size = len(mixed_initial_conditions)
        dynamics_eq = self.dynamics_functions[mode_index]
        args = self.dynamics_functions_args[mode_index]

        # print(time_index)

        # if mode_index != 0:
        #     args = list(args)
        #     args[1] *= mode_probability

        ICs = np.concatenate((mixed_initial_conditions, np.eye(state_size).flatten()))
        propagation = scipy.integrate.solve_ivp(dynamics_eq, [0,timespan], ICs, args=args, atol=1e-12, rtol=1e-12).y[:, -1]

        STM = propagation[state_size:state_size**2 + state_size].reshape((state_size, state_size))
        anterior_estimate = propagation[0:state_size]
        anterior_covariance = enforce_symmetry(STM @ mixed_initial_covariance @ STM.T + self.process_noise_covariances[mode_index])

        return anterior_estimate, anterior_covariance
    
    def kernel_probability_update(self, anterior_kernel_probabilities, denominators, exponents, current_measurement):

        # if len(current_measurement[np.isnan(current_measurement) == False]) == 0:
        #     return previous_mode_probabilities
        
        num_modes = np.size(denominators, 0)
        kernels_per_mode = np.size(denominators, 1)
        new_kernel_probabilities = np.empty(num_modes, kernels_per_mode)
        normalized_denominators = denominators / denominators.min()
        normalized_exponents = exponents - exponents.max()
        
        measurement_likelihoods = 1 / normalized_denominators * np.exp(normalized_exponents)
        raw_mode_probabilities = measurement_likelihoods*anterior_kernel_probabilities
        new_kernel_probabilities = raw_mode_probabilities / np.sum(raw_mode_probabilities)
        
        return new_kernel_probabilities
    
    def mixed_initial_conditions(self, mode, previous_kernel_probabilities, previous_posterior_estimates, previous_posterior_covariances):
        
        num_modes = np.size(previous_kernel_probabilities, 0)
        kernels_per_mode = np.size(previous_kernel_probabilities, 1)
        state_size = np.size(previous_posterior_estimates, 0)
        mode_transition_matrix = self.mode_transition_matrix

        raw_mixing_proportions = np.empty((num_modes, kernels_per_mode))
        for mode_index in range(num_modes):
            for kernel_index in range(kernels_per_mode):
                raw_mixing_proportions[mode_index, kernel_index] = previous_kernel_probabilities[mode_index, kernel_index] * mode_transition_matrix[mode_index, mode]
        
        anterior_mode_probability = np.sum(raw_mixing_proportions)
        mixing_proportions = raw_mixing_proportions / anterior_mode_probability

        ind = np.argpartition(mixing_proportions.flatten(), -(kernels_per_mode-1))[-(kernels_per_mode-1):]

        initial_states = np.zeros((state_size, kernels_per_mode))
        initial_covariances = np.zeros((state_size, state_size, kernels_per_mode))
        anterior_kernel_probabilities = np.zeros(kernels_per_mode)
        mixed_state = np.zeros(state_size)
        mixed_covariance = np.zeros((state_size, state_size))
        
        overall_index = 0
        IC_index = 0
        for mode_index in range(num_modes):
            for kernel_index in range(kernels_per_mode):
                if overall_index in ind:
                    initial_states[:, IC_index] = previous_posterior_estimates[:, mode_index, kernel_index]
                    initial_covariances[:, :, IC_index] = previous_posterior_covariances[:, :, mode_index, kernel_index]
                    anterior_kernel_probabilities[IC_index] = raw_mixing_proportions[mode_index, kernel_index]
                    IC_index += 1
                else:
                    mixed_state += mixing_proportions[mode_index, kernel_index] * previous_posterior_estimates[:, mode_index, kernel_index]
                    anterior_kernel_probabilities[-1] += raw_mixing_proportions[mode_index, kernel_index]
                overall_index += 1

        overall_index = 0
        for mode_index in range(num_modes):
            for kernel_index in range(kernels_per_mode):
                if overall_index not in ind:
                    difference = (previous_posterior_estimates[:, mode_index, kernel_index] - mixed_state)[:, None]
                    mixed_covariance += mixing_proportions[mode_index, kernel_index] * (previous_posterior_covariances[:, :, mode_index, kernel_index] + difference @ difference.T)
        
        initial_states[:, -1] = mixed_state
        initial_covariances[:, :, -1] = mixed_covariance
        
        return initial_states, initial_covariances, anterior_kernel_probabilities
    
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
    
    def run_MC(self, initial_estimates, initial_covariance, initial_mode_probabilities, time_vals, measurements, measurement_args):

        num_runs = len(initial_estimates)
        results = []

        with parallel_config(verbose=100, n_jobs=-1):
            results = Parallel()(delayed(self.run)(initial_estimates[run_index], initial_covariance, initial_mode_probabilities, time_vals, measurements[run_index], measurement_args[run_index]) for run_index in range(num_runs))
        
        return GMIMM_MCResults(results)

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