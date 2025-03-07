
import numpy as np
import scipy
from CR3BP_pontryagin import *
from CR3BP_pontryagin_reformulated import *
from measurement_functions import *


def state_cost_reformulated(guess, magnitude, observation_times, observation_states, observation_covariances, propagation_args):
    
    # print(guess)

    tspan = np.array([observation_times[0], observation_times[-1]])
    initial_conditions = np.concatenate((guess, np.array([magnitude])))
    propagation = scipy.integrate.solve_ivp(reformulated_min_fuel_ODE, tspan, initial_conditions, args=propagation_args, t_eval=observation_times, atol=1e-12, rtol=1e-12).y

    num_steps = len(observation_times)
    cost = 0
    for timestep_index in range(num_steps):
        propagated = propagation[0:6, timestep_index]
        mean = observation_states[0:6, timestep_index]
        covariance = observation_covariances[0:6, 0:6, timestep_index]
        cost += scipy.spatial.distance.mahalanobis(propagated, mean, np.linalg.inv(covariance))

    return cost

def measurement_cost_reformulated(guess, magnitude, measurement_times, measurements, individual_measurement_covariance, obs_positions, check_results, propagation_args):
    
    # print(guess)

    tspan = np.array([measurement_times[0], measurement_times[-1]])
    initial_conditions = np.concatenate((guess, np.array([magnitude])))
    propagation = scipy.integrate.solve_ivp(reformulated_min_fuel_ODE, tspan, initial_conditions, args=propagation_args, t_eval=measurement_times, atol=1e-12, rtol=1e-12).y

    num_steps = len(measurement_times)
    num_sensors = np.size(check_results, axis=0)
    individual_measurement_size = np.size(individual_measurement_covariance, 0)

    propagated_measurements = generate_sensor_measurements(measurement_times, propagation, az_el_sensor, 2, np.zeros((2, 2)), obs_positions, check_results, 0).measurements

    cost = 0
    for timestep_index in range(num_steps):
        for sensor_index in range(num_sensors):
            if check_results[sensor_index, timestep_index] == 1:
                propagated_measurement = propagated_measurements[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size, timestep_index]
                true_measurement = measurements[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size, timestep_index]
                cost += scipy.spatial.distance.mahalanobis(propagated_measurement, true_measurement, np.linalg.inv(individual_measurement_covariance))

    return cost

def measurement_cost_costate(guess, state, magnitude, measurement_times, measurements, individual_measurement_covariance, obs_positions, check_results, propagation_args):
    
    # print(guess)

    tspan = np.array([measurement_times[0], measurement_times[-1]])
    initial_conditions = np.concatenate((state, guess, np.array([magnitude])))
    propagation = scipy.integrate.solve_ivp(reformulated_min_fuel_ODE, tspan, initial_conditions, args=propagation_args, t_eval=measurement_times, atol=1e-12, rtol=1e-12).y

    num_steps = len(measurement_times)
    num_sensors = np.size(check_results, axis=0)
    individual_measurement_size = np.size(individual_measurement_covariance, 0)

    propagated_measurements = generate_sensor_measurements(measurement_times, propagation, az_el_sensor, 2, np.zeros((2, 2)), obs_positions, check_results, 0).measurements

    cost = 0
    for timestep_index in range(num_steps):
        for sensor_index in range(num_sensors):
            if check_results[sensor_index, timestep_index] == 1:
                propagated_measurement = propagated_measurements[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size, timestep_index]
                true_measurement = measurements[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size, timestep_index]
                cost += scipy.spatial.distance.mahalanobis(propagated_measurement, true_measurement, np.linalg.inv(individual_measurement_covariance))

    return cost

def measurement_lstsqr_standard(guess, measurement_times, measurements, individual_measurement_covariance, obs_positions, check_results, propagation_args):
    
    # print(guess)

    tspan = np.array([measurement_times[0], measurement_times[-1]])
    propagation = scipy.integrate.solve_ivp(minimum_fuel_ODE, tspan, guess, args=propagation_args, t_eval=measurement_times, atol=1e-12, rtol=1e-12).y

    num_steps = len(measurement_times)
    num_sensors = np.size(check_results, axis=0)
    individual_measurement_size = np.size(individual_measurement_covariance, 0)
    measurement_std = np.sqrt(individual_measurement_covariance[0, 0])

    propagated_measurements = generate_sensor_measurements(measurement_times, propagation, az_el_sensor, 2, np.zeros((2, 2)), obs_positions, check_results, 0).measurements

    residuals = []
    for timestep_index in range(num_steps):
        for sensor_index in range(num_sensors):
            if check_results[sensor_index, timestep_index] == 1:
                for measurement_index in range(individual_measurement_size):
                    propagated_measurement = propagated_measurements[sensor_index*individual_measurement_size+measurement_index, timestep_index]
                    true_measurement = measurements[sensor_index*individual_measurement_size+measurement_index, timestep_index]
                    residuals.append((propagated_measurement - true_measurement)/measurement_std)

    return np.array(residuals)

def measurement_lstsqr_reformulated(guess, magnitude, measurement_times, measurements, individual_measurement_covariance, obs_positions, check_results, propagation_args):
    
    # print(guess)

    tspan = np.array([measurement_times[0], measurement_times[-1]])
    initial_conditions = np.concatenate((guess, np.array([magnitude])))
    propagation = scipy.integrate.solve_ivp(reformulated_min_fuel_ODE, tspan, initial_conditions, args=propagation_args, t_eval=measurement_times, atol=1e-12, rtol=1e-12).y

    num_steps = len(measurement_times)
    num_sensors = np.size(check_results, axis=0)
    individual_measurement_size = np.size(individual_measurement_covariance, 0)
    measurement_std = np.sqrt(individual_measurement_covariance[0, 0])

    propagated_measurements = generate_sensor_measurements(measurement_times, propagation, az_el_sensor, 2, np.zeros((2, 2)), obs_positions, check_results, 0).measurements

    residuals = []
    for timestep_index in range(num_steps):
        for sensor_index in range(num_sensors):
            if check_results[sensor_index, timestep_index] == 1:
                for measurement_index in range(individual_measurement_size):
                    propagated_measurement = propagated_measurements[sensor_index*individual_measurement_size+measurement_index, timestep_index]
                    true_measurement = measurements[sensor_index*individual_measurement_size+measurement_index, timestep_index]
                    residuals.append((propagated_measurement - true_measurement)/measurement_std)

    return np.array(residuals)

def measurement_lstsqr_reformulated_mag(guess, measurement_times, measurements, individual_measurement_covariance, obs_positions, check_results, propagation_args):
    
    # print(guess)

    tspan = np.array([measurement_times[0], measurement_times[-1]])
    propagation = scipy.integrate.solve_ivp(reformulated_min_fuel_ODE, tspan, guess, args=propagation_args, t_eval=measurement_times, atol=1e-12, rtol=1e-12).y

    num_steps = len(measurement_times)
    num_sensors = np.size(check_results, axis=0)
    individual_measurement_size = np.size(individual_measurement_covariance, 0)
    measurement_std = np.sqrt(individual_measurement_covariance[0, 0])

    propagated_measurements = generate_sensor_measurements(measurement_times, propagation, az_el_sensor, 2, np.zeros((2, 2)), obs_positions, check_results, 0).measurements

    residuals = []
    for timestep_index in range(num_steps):
        for sensor_index in range(num_sensors):
            if check_results[sensor_index, timestep_index] == 1:
                for measurement_index in range(individual_measurement_size):
                    propagated_measurement = propagated_measurements[sensor_index*individual_measurement_size+measurement_index, timestep_index]
                    true_measurement = measurements[sensor_index*individual_measurement_size+measurement_index, timestep_index]
                    residuals.append((propagated_measurement - true_measurement)/measurement_std)

    return np.array(residuals)

def measurement_lstsqr_costate(guess, state, magnitude, measurement_times, measurements, individual_measurement_covariance, obs_positions, check_results, propagation_args):
    
    # print(guess)

    tspan = np.array([measurement_times[0], measurement_times[-1]])
    initial_conditions = np.concatenate((state, guess, np.array([magnitude])))
    propagation = scipy.integrate.solve_ivp(reformulated_min_fuel_ODE, tspan, initial_conditions, args=propagation_args, t_eval=measurement_times, atol=1e-12, rtol=1e-12).y

    num_steps = len(measurement_times)
    num_sensors = np.size(check_results, axis=0)
    individual_measurement_size = np.size(individual_measurement_covariance, 0)
    measurement_std = np.sqrt(individual_measurement_covariance[0, 0])

    propagated_measurements = generate_sensor_measurements(measurement_times, propagation, az_el_sensor, 2, np.zeros((2, 2)), obs_positions, check_results, 0).measurements

    residuals = []
    for timestep_index in range(num_steps):
        for sensor_index in range(num_sensors):
            if check_results[sensor_index, timestep_index] == 1:
                for measurement_index in range(individual_measurement_size):
                    propagated_measurement = propagated_measurements[sensor_index*individual_measurement_size+measurement_index, timestep_index]
                    true_measurement = measurements[sensor_index*individual_measurement_size+measurement_index, timestep_index]
                    residuals.append((propagated_measurement - true_measurement)/measurement_std)

    return np.array(residuals)

