

import numpy as np
from joblib import Parallel, delayed
from helper_functions import *
from EKF import *

class ParticleFilterResults:
    def __init__(self, time_vals, estimate_vals, innovations_vals, weight_vals):
        self.t = time_vals
        self.estimate_vals = estimate_vals
        self.innovations_vals = innovations_vals
        self.weight_vals = weight_vals

def iterate_particle(time_index, previous_estimate, 
                dynamics_equation, measurement_equation, individual_measurement_size, 
                measurement_noise_covariance, measurement, timespan, 
                dynamics_args, measurement_args):
    
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

def run_particle_filter(initial_estimates, initial_weights,
               dynamics_equation, measurement_equation, measurements,
               measurement_noise_covariance, dynamics_args,
               measurement_args, resampling_args):
    
    time_vals = measurements.t
    measurement_vals = measurements.measurements
    individual_measurement_size = measurements.individual_measurement_size

    state_size = np.size(initial_estimates, 0)
    num_particles = np.size(initial_estimates, 1)
    measurement_size = np.size(measurement_vals, 0)
    num_measurements = np.size(measurement_vals, 1)

    weight_vals = np.empty((num_particles, num_measurements))
    estimate_vals = np.empty((state_size, num_measurements, num_particles))
    innovations_vals = np.empty((measurement_size, num_measurements, num_particles))
    
    weight_vals[:, 0] = initial_weights
    estimate_vals[:, 0, :] = initial_estimates

    previous_time = time_vals[0]
    previous_estimates = initial_estimates
    previous_weights = initial_weights
    
    denominators = np.empty(num_particles)
    exponents = np.empty(num_particles)

    for time_index in np.arange(1, num_measurements):
        print(time_index)

        measurement = measurement_vals[:, time_index]
        current_time = time_vals[time_index]
        timespan = current_time - previous_time
        
        estimates = np.empty((state_size, num_particles))

        if np.array_equal(measurement, np.empty(measurement_size)*np.nan, equal_nan=True):
            propagation_input_list = []
            for particle_index in np.arange(num_particles):
                previous_estimate = previous_estimates[:, particle_index]

                propagation_input_list.append((previous_estimate, dynamics_equation, timespan, dynamics_args, measurement_size))

            particle_propagations = Parallel(n_jobs=8)(delayed(propagate_particle)(*propagation_inputs) for propagation_inputs in propagation_input_list)

            for particle_index in np.arange(num_particles):
                estimates[:, particle_index] = particle_propagations[particle_index][0]
                innovations_vals[:, time_index, particle_index] = particle_propagations[particle_index][1]

            weight_vals[:, time_index] = previous_weights

            estimate_vals[:, time_index, :] = estimates

            previous_time = current_time
            previous_estimates = estimates
            previous_weights = previous_weights
        
        else:
            iteration_input_list = []
            for particle_index in np.arange(num_particles):
                
                previous_estimate = previous_estimates[:, particle_index]
                iteration_input_list.append((time_index, previous_estimate, 
                            dynamics_equation, measurement_equation, individual_measurement_size, 
                            measurement_noise_covariance, 
                            measurement, timespan, dynamics_args, measurement_args))
                
            particle_iterations = Parallel(n_jobs=8)(delayed(iterate_particle)(*iteration_inputs) for iteration_inputs in iteration_input_list)
            
            for particle_index in np.arange(num_particles):
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
            
            new_weights, estimates = resample_N_eff(new_weights, estimates, *resampling_args)
            
            weight_vals[:, time_index] = new_weights
            estimate_vals[:, time_index, :] = estimates

            previous_estimates = estimates
            previous_weights = new_weights
            previous_time = current_time

    return ParticleFilterResults(time_vals, estimate_vals, innovations_vals, weight_vals)

def resample_N_eff(weights, estimates, roughening_cov):

    N_eff = 1 / np.sum(weights**2)
    num_particles = len(weights)

    if N_eff < num_particles/4:

        generator = np.random.default_rng(0)
        cumulative = np.concatenate((np.zeros(1), np.cumsum(weights)))

        new_indices = np.searchsorted(cumulative, generator.uniform(0, 1, num_particles), side="right") - 1

        new_estimates = np.copy(estimates)*0

        roughening_noise = generator.multivariate_normal(np.zeros(12), roughening_cov, num_particles)
        for index in np.arange(num_particles):
            new_estimates[:, index] = estimates[:, new_indices[index]] + roughening_noise[index]
        
        new_weights = np.ones(num_particles) / num_particles

        return new_weights, new_estimates
    
    else:
        return weights, estimates