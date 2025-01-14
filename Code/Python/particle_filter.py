

import numpy as np
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

    return estimate, innovations, denominator, exponent

def propagate_particle(previous_estimate, dynamics_equation, timespan, dynamics_args, measurement_size):
    
    initial_conditions = previous_estimate
    propagation = scipy.integrate.solve_ivp(dynamics_equation, np.array([0, timespan]), initial_conditions, args=dynamics_args, atol=1e-12, rtol=1e-12)

    estimate = propagation.y[:, -1]

    innovations = np.ones(measurement_size)*np.nan

    return estimate, innovations

def run_particle_filter(initial_estimates, initial_weights,
               dynamics_equation, measurement_equation, measurements,
               measurement_noise_covariance,
               dynamics_args, measurement_args):
    
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
            for particle_index in np.arange(num_particles):

                previous_estimate = previous_estimates[:, particle_index]

                propagation_inputs = (previous_estimate, dynamics_equation, timespan, dynamics_args, measurement_size)
                estimate, innovations = propagate_particle(*propagation_inputs)

                estimates[:, particle_index] = estimate
                innovations_vals[:, time_index, particle_index] = innovations
                denominators[particle_index] = 1
                exponents[particle_index] = 1
        else:
            for particle_index in np.arange(num_particles):

                previous_estimate = previous_estimates[:, particle_index]

                interation_inputs = (time_index, previous_estimate, 
                            dynamics_equation, measurement_equation, individual_measurement_size, 
                            measurement_noise_covariance, 
                            measurement, timespan, dynamics_args, measurement_args)      
                estimate, innovations, denominator, exponent = iterate_particle(*interation_inputs)

                estimates[:, particle_index] = estimate
                innovations_vals[:, time_index, particle_index] = innovations
                denominators[particle_index] = denominator
                exponents[particle_index] = exponent
        
        new_weights = np.empty(num_particles)
        normalized_denominators = denominators / denominators.min()
        normalized_exponents = exponents - exponents.max()
        measurement_probabilities = 1 / normalized_denominators * np.exp(normalized_exponents)
        raw_weights = previous_weights*measurement_probabilities
        new_weights = raw_weights/np.sum(raw_weights)
        weight_vals[:, time_index] = new_weights

        estimate_vals[:, time_index, :] = estimates

        previous_time = current_time
        previous_estimates = estimates
        previous_weights = new_weights

    return ParticleFilterResults(time_vals, estimate_vals, innovations_vals, weight_vals)