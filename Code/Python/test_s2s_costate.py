

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from configuration_costate import *
from CR3BP import *
from CR3BP_pontryagin import *
from EKF import *
from dynamics_functions import *
from helper_functions import *
from measurement_functions import *
from plotting import *

time_vals = np.arange(0, final_time, dt)
tspan = np.array([time_vals[0], time_vals[-1]])
truth_propagation = scipy.integrate.solve_ivp(dynamics_equation, tspan, initial_truth, args=(mu, umax), t_eval=time_vals, atol=1e-12, rtol=1e-12)
truth_vals = truth_propagation.y

sensor_position_vals = generate_sensor_positions(sensor_dynamics_equation, sensor_initial_conditions, (mu,), time_vals)

num_sensors = int(np.size(sensor_position_vals, 0)/3)
earth_vectors = np.empty((3*num_sensors, len(time_vals)))
moon_vectors = np.empty((3*num_sensors, len(time_vals)))
sun_vectors = np.empty((3*num_sensors, len(time_vals)))
for sensor_index in np.arange(num_sensors):
    sensor_positions = sensor_position_vals[sensor_index*3:(sensor_index + 1)*3, :]
    earth_vectors[sensor_index*3:(sensor_index + 1)*3, :] = generate_earth_vectors(time_vals, sensor_positions)
    moon_vectors[sensor_index*3:(sensor_index + 1)*3, :] = generate_moon_vectors(time_vals, sensor_positions)
    sun_vectors[sensor_index*3:(sensor_index + 1)*3, :] = generate_sun_vectors(time_vals, 0)

earth_results = np.empty((num_sensors, len(time_vals)))
moon_results = np.empty((num_sensors, len(time_vals)))
sun_results = np.empty((num_sensors, len(time_vals)))
check_results = np.empty((num_sensors, len(time_vals)))
for sensor_index in np.arange(num_sensors):
    sensor_positions = sensor_position_vals[sensor_index*3:(sensor_index + 1)*3, :]
    earth_results[sensor_index, :] = check_validity(time_vals, truth_vals[0:3, :], sensor_positions, earth_vectors[sensor_index*3:(sensor_index+1)*3, :], check_exclusion, (earth_exclusion_angle,))
    moon_results[sensor_index, :] = check_validity(time_vals, truth_vals[0:3, :], sensor_positions, moon_vectors[sensor_index*3:(sensor_index+1)*3, :], check_exclusion_dynamic, (9.0400624349e-3, moon_additional_angle))
    sun_results[sensor_index, :] = check_validity(time_vals, truth_vals[0:3, :], sensor_positions, sun_vectors[sensor_index*3:(sensor_index+1)*3, :], check_exclusion, (sun_exclusion_angle,))
    check_results[sensor_index, :] = earth_results[sensor_index, :] * moon_results[sensor_index, :] * sun_results[sensor_index, :]

check_results[0, -100:] = 0
check_results[1, -100:] = 0

# check_results[0, :25] = 0
check_results[1, :25] = 1

check_results[:, :] = 1

measurements = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, measurement_noise_covariance, sensor_position_vals, check_results, seed)

dynamics_args = (mu, umax)
measurement_args = (mu, sensor_position_vals, individual_measurement_size)

def EKF_dynamics_equation(t, X, mu, umax, process_noise_covariance):

    state = X[0:6]
    costate = X[6:12]
    covariance = X[12:156].reshape((12, 12))

    jacobian = CR3BP_costate_jacobian(state, costate, mu, umax)

    ddt_state = minimum_energy_ODE(0, X[0:12], mu, umax)
    ddt_covariance = jacobian @ covariance + covariance @ jacobian.T + process_noise_covariance

    return np.concatenate((ddt_state, ddt_covariance.flatten()))
    
def EKF_measurement_equation(time_index, X, mu, sensor_position_vals, individual_measurement_size):

    num_sensors = int(np.size(sensor_position_vals, 0)/3)
    measurement = np.empty(num_sensors*individual_measurement_size)
    measurement_jacobian = np.empty((num_sensors*individual_measurement_size, 12))

    for sensor_index in np.arange(num_sensors):
        sensor_position = sensor_position_vals[sensor_index*3:(sensor_index+1)*3, time_index+1]
        
        measurement[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor(X, sensor_position)
        measurement_jacobian[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor_jacobian_costate(X, sensor_position)

    return measurement, measurement_jacobian


filter_output = run_EKF(initial_estimate, initial_covariance,
                        EKF_dynamics_equation, EKF_measurement_equation,
                        measurements, process_noise_covariance,
                        measurement_noise_covariance, 
                        dynamics_args, measurement_args)

filter_time = filter_output.t
posterior_estimate_vals = filter_output.posterior_estimate_vals
posterior_covariance_vals = filter_output.posterior_covariance_vals
anterior_estimate_vals = filter_output.anterior_estimate_vals
anterior_covariance_vals = filter_output.anterior_covariance_vals
innovations = filter_output.innovations_vals

ax = plt.figure().add_subplot()
ax.plot(measurements.t, measurements.measurements[0])
ax.plot(measurements.t, measurements.measurements[1])
ax.plot(measurements.t, measurements.measurements[2])
ax.plot(measurements.t, measurements.measurements[3])

ax = plt.figure().add_subplot()
ax.plot(time_vals, check_results[0], alpha=0.25)
ax.plot(time_vals, earth_results[0], alpha=0.25)
ax.plot(time_vals, moon_results[0], alpha=0.25)
ax.plot(time_vals, sun_results[0], alpha=0.25)

ax = plt.figure().add_subplot()
ax.plot(time_vals, check_results[1], alpha=0.25)
ax.plot(time_vals, earth_results[1], alpha=0.25)
ax.plot(time_vals, moon_results[1], alpha=0.25)
ax.plot(time_vals, sun_results[1], alpha=0.25)

ax = plt.figure().add_subplot(projection="3d")
ax.set_aspect("equal")
ax.plot(sensor_position_vals[0], sensor_position_vals[1], sensor_position_vals[2])
ax.plot(sensor_position_vals[3], sensor_position_vals[4], sensor_position_vals[5])
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
ax.set_aspect("equal")

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
ax.plot(posterior_estimate_vals[0], posterior_estimate_vals[1], posterior_estimate_vals[2])
ax.set_aspect("equal")

posterior_estimates = [posterior_estimate_vals]
posterior_covariances = [posterior_covariance_vals]

estimation_errors = compute_estimation_errors(truth_vals, posterior_estimates, 12)
three_sigmas = compute_3sigmas(posterior_covariances, 12)
plot_3sigma(time_vals, estimation_errors, three_sigmas, 1, 1, 6)
plot_3sigma_costate(time_vals, estimation_errors, three_sigmas, 1, 1, 6)

plt.show()