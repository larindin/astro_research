

import numpy as np
from joblib import Parallel, delayed
from helper_functions import *
from EKF import *

class ParticleFilterResults:
    def __init__(self, time_vals, estimate_vals, innovations_vals, anterior_weight_vals, weight_vals):
        self.t = time_vals
        self.estimate_vals = estimate_vals
        self.innovations_vals = innovations_vals
        self.anterior_weight_vals = anterior_weight_vals
        self.weight_vals = weight_vals

def iterate_particle(previous_estimate, time_index, dynamics_equation, 
                     measurement_equation, individual_measurement_size, measurement_noise_covariance, 
                     measurement, timespan, dynamics_args, measurement_args):
    
    state_size = np.size(previous_estimate, 0)
    initial_conditions = previous_estimate
    propagation = scipy.integrate.solve_ivp(dynamics_equation, np.array([0, timespan]), initial_conditions, args=dynamics_args, atol=1e-12, rtol=1e-12)

    estimate = propagation.y[:, -1]

    measurement, valid_indices = assess_measurement(measurement, individual_measurement_size)

    predicted_measurement, measurement_jacobian = measurement_equation(time_index, estimate, *measurement_args)
    predicted_measurement, measurement_jacobian = parse_measurement(predicted_measurement, measurement_jacobian, individual_measurement_size, valid_indices)

    measurement_noise_covariance = scipy.linalg.block_diag(*(measurement_noise_covariance, )*len(valid_indices))

    innovations = measurement - predicted_measurement
    innovations = check_innovations(innovations)

    denominator, exponent = assess_measurement_probability(innovations, measurement_noise_covariance)

    return [estimate, innovations, denominator, exponent]

def propagate_particle(previous_estimate, dynamics_equation, timespan, dynamics_args, measurement_size):
    
    initial_conditions = previous_estimate
    propagation = scipy.integrate.solve_ivp(dynamics_equation, np.array([0, timespan]), initial_conditions, args=dynamics_args, atol=1e-12, rtol=1e-12)

    estimate = propagation.y[:, -1]

    innovations = np.ones(measurement_size)*np.nan

    return [estimate, innovations]

def run_particle_filter(initial_estimates, initial_weights, dynamics_equation, 
                        measurement_equation, measurements, measurement_noise_covariance, 
                        dynamics_args, measurement_args, roughening_cov, seed):
    
    time_vals = measurements.t
    measurement_vals = measurements.measurements
    individual_measurement_size = measurements.individual_measurement_size

    state_size = np.size(initial_estimates, 0)
    num_particles = np.size(initial_estimates, 1)
    measurement_size = np.size(measurement_vals, 0)
    num_measurements = np.size(measurement_vals, 1)

    weight_vals = np.empty((num_particles, num_measurements))
    anterior_weight_vals = np.empty(np.shape(weight_vals))
    estimate_vals = np.empty((state_size, num_measurements, num_particles))
    innovations_vals = np.empty((measurement_size, num_measurements, num_particles))
    
    weight_vals[:, 0] = initial_weights
    estimate_vals[:, 0, :] = initial_estimates

    previous_time = time_vals[0]
    previous_estimates = initial_estimates
    previous_weights = initial_weights
    
    denominators = np.empty(num_particles)
    exponents = np.empty(num_particles)
    generator = np.random.default_rng(seed)

    initial_conditions = initial_estimates.copy()

    for time_index in range(1, num_measurements):
        print(time_index)

        measurement = measurement_vals[:, time_index]
        current_time = time_vals[time_index]
        timespan = current_time - previous_time
        
        estimates = np.empty((state_size, num_particles))

        if np.array_equal(measurement, np.empty(measurement_size)*np.nan, equal_nan=True):

            propagation_inputs = (dynamics_equation, timespan, dynamics_args, measurement_size)
            particle_propagations = Parallel(n_jobs=8)(delayed(propagate_particle)(previous_estimates[:, particle_index], *propagation_inputs) for particle_index in range(num_particles))

            for particle_index in range(num_particles):
                estimates[:, particle_index] = particle_propagations[particle_index][0]
                innovations_vals[:, time_index, particle_index] = particle_propagations[particle_index][1]

            anterior_weight_vals[:, time_index] = previous_weights
            weight_vals[:, time_index] = previous_weights

            estimate_vals[:, time_index, :] = estimates

            previous_time = current_time
            previous_estimates = estimates
            previous_weights = previous_weights
        
        else:
            iteration_inputs = (time_index, dynamics_equation, measurement_equation, individual_measurement_size, 
                            measurement_noise_covariance, measurement, timespan, dynamics_args, measurement_args)
            particle_iterations = Parallel(n_jobs=8)(delayed(iterate_particle)(previous_estimates[:, particle_index], *iteration_inputs) for particle_index in range(num_particles))
            
            for particle_index in range(num_particles):
                estimates[:, particle_index] = particle_iterations[particle_index][0]
                innovations_vals[:, time_index, particle_index] = particle_iterations[particle_index][1]
                denominators[particle_index] = particle_iterations[particle_index][2]
                exponents[particle_index] = particle_iterations[particle_index][3]

            new_weights = np.empty(num_particles)
            normalized_denominators = denominators / denominators.min()
            normalized_exponents = exponents - exponents.max()
            measurement_probabilities = 1 / normalized_denominators * np.exp(normalized_exponents)
            raw_weights = previous_weights*measurement_probabilities
            new_weights = raw_weights/np.sum(raw_weights)

            anterior_weight_vals[:, time_index] = new_weights.copy()

            # if N_eff(new_weights, 0.25):
            #     new_indices = stratified_resampling(new_weights, generator)

            #     unique_new_indices, counts = np.unique(new_indices, return_counts=True)
            #     num_unique_resampled = len(unique_new_indices)
                
            #     new_estimates = np.empty(np.shape(estimates))
            #     new_initial_conditions = np.empty(np.shape(initial_conditions))

            #     assignment_index = 0
                
            #     for resampled_particle_index in range(num_unique_resampled):
            #         new_estimates[:, assignment_index] = estimates[:, unique_new_indices[resampled_particle_index]]
            #         new_initial_conditions[:, assignment_index] = initial_conditions[:, unique_new_indices[resampled_particle_index]]
            #         assignment_index += 1
                    
            #         duplicate_initial_conditions = np.empty((12, counts[resampled_particle_index]-1))
            #         roughening_vals = generator.multivariate_normal(np.zeros(12), roughening_cov, counts[resampled_particle_index]-1)

            #         for duplicate_index in range(counts[resampled_particle_index]-1):
            #             duplicate_initial_conditions[:, duplicate_index] = initial_conditions[:, unique_new_indices[resampled_particle_index]] + roughening_vals[duplicate_index]
                        
            #         propagation_inputs = (dynamics_equation, current_time, dynamics_args, measurement_size)
            #         particle_propagations = Parallel(n_jobs=8)(delayed(propagate_particle)(duplicate_initial_conditions[:, duplicate_index], *propagation_inputs) for duplicate_index in range(counts[resampled_particle_index]-1))
                    
            #         for duplicate_index in range(counts[resampled_particle_index]-1):
            #             new_estimates[:, assignment_index] = particle_propagations[duplicate_index][0]
            #             new_initial_conditions[:, assignment_index] = duplicate_initial_conditions[:, duplicate_index]
            #             assignment_index += 1
                        
                
            #     initial_conditions = new_initial_conditions
            #     estimates = new_estimates
            
            if N_eff(new_weights, 0.25):
                new_indices = stratified_resampling(new_weights, generator)

                unique_new_indices, counts = np.unique(new_indices, return_counts=True)
                num_unique_resampled = len(unique_new_indices)
                
                new_estimates = np.empty(np.shape(estimates))

                assignment_index = 0
                roughening_vals = generator.multivariate_normal(np.zeros(12), roughening_cov, num_particles)

                for resampled_particle_index in range(num_unique_resampled):
                    new_estimates[:, assignment_index] = estimates[:, unique_new_indices[resampled_particle_index]]
                    assignment_index += 1
                    
                    for duplicate_index in range(counts[resampled_particle_index]-1):
                        new_estimates[:, assignment_index] = estimates[:, unique_new_indices[resampled_particle_index]] + roughening_vals[assignment_index]
                        assignment_index += 1
                
                estimates = new_estimates
                new_weights = np.ones(num_particles)/num_particles

            
            weight_vals[:, time_index] = new_weights
            estimate_vals[:, time_index, :] = estimates

            previous_estimates = estimates
            previous_weights = new_weights
            previous_time = current_time

    return ParticleFilterResults(time_vals, estimate_vals, innovations_vals, anterior_weight_vals, weight_vals)

def N_eff(weights, proportion):
    return 1 / np.sum(weights**2) < len(weights) * proportion

def multinomial_resampling(weights, generator):

    num_particles = len(weights)
    cumulative = np.concatenate(([0], np.cumsum(weights)))

    new_indices = np.searchsorted(cumulative, generator.uniform(0, 1, num_particles), side="right") - 1

    return new_indices

def stratified_resampling(weights, generator):

    num_particles = len(weights)
    cumulative = np.concatenate(([0], np.cumsum(weights)))

    u0_vals = generator.uniform(0, 1/num_particles, num_particles)
    new_indices = np.searchsorted(cumulative, u0_vals+np.arange(num_particles)/num_particles, side="right") - 1

    return new_indices

def calculate_N_eff_vals(weight_vals):
    return 1 / np.sum(weight_vals**2, axis=0)