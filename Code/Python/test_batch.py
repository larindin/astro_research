

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from configuration_costate import *
from CR3BP import *
from CR3BP_pontryagin import *
from EKF import *
from batch import *
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

check_results[:, :] = 0
check_results[:, 0:5] = 1

measurements = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, measurement_noise_covariance, sensor_position_vals, check_results, seed)

dynamics_args = (mu, umax)
measurement_args = (mu, sensor_position_vals, individual_measurement_size)

def dynamics_function(t, X, mu, umax):

    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    jacobian = CR3BP_costate_jacobian(state, costate, mu, umax)

    ddt_state = minimum_energy_ODE(0, X[0:12], mu, umax)
    ddt_STM = jacobian @ STM

    return np.concatenate((ddt_state, ddt_STM.flatten()))

def measurement_function(time_index, X, mu, sensor_position_vals, individual_measurement_size):

    num_sensors = int(np.size(sensor_position_vals, 0)/3)
    measurement = np.empty(num_sensors*individual_measurement_size)
    measurement_jacobian = np.empty((num_sensors*individual_measurement_size, 12))

    for sensor_index in np.arange(num_sensors):
        sensor_position = sensor_position_vals[sensor_index*3:(sensor_index+1)*3, time_index+1]
        
        measurement[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor(X, sensor_position)
        measurement_jacobian[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor_jacobian_costate(X, sensor_position)

    return measurement, measurement_jacobian

how_many = 10
time_indices = np.array(np.arange(how_many))
measurements.t = measurements.t[0:how_many]
measurements.measurements = measurements.measurements[:, 0:how_many]

initial_estimate = initial_truth

estimates, initial_covariance = run_batch_processor(initial_estimate, initial_covariance, time_indices,
                                                           measurements, dynamics_function, measurement_function, 
                                                           dynamics_args, measurement_args, max_iterations = 100)


ax = plt.figure().add_subplot(projection="3d")
ax.set_aspect("equal")
ax.plot(truth_vals[0, 0:how_many], truth_vals[1, 0:how_many], truth_vals[2, 0:how_many])
for estimate_index in np.arange(np.size(estimates, 0)):
    estimate = estimates[estimate_index, :]
    propagation = scipy.integrate.solve_ivp(minimum_energy_ODE, np.array([0, how_many*0.01]), estimate, args=dynamics_args, atol=1e-12, rtol=1e-12)
    output = propagation.y
    ax.plot(output[0], output[1], output[2])

plt.show()

quit()

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

truth_control = get_min_energy_control(truth_vals[6:12, :], umax)
posterior_control = get_min_energy_control(posterior_estimate_vals[6:12, :], umax)
ax = plt.figure().add_subplot()
ax.plot(time_vals, np.linalg.norm(truth_control, axis=0))
ax.plot(time_vals, np.linalg.norm(posterior_control, axis=0))

fig = plt.figure()
ax = fig.add_subplot(131)
ax.plot(time_vals, truth_control[0])
ax.plot(time_vals, posterior_control[0])
ax = fig.add_subplot(132)
ax.plot(time_vals, truth_control[1])
ax.plot(time_vals, posterior_control[1])
ax = fig.add_subplot(133)
ax.plot(time_vals, truth_control[2])
ax.plot(time_vals, posterior_control[2])

plt.show()