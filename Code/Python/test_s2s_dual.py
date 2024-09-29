

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from configuration_costate_dual import *
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

# check_results[0, -100:] = 0
# check_results[1, -100:] = 0

# check_results[0, :25] = 0
# check_results[1, :25] = 1

# check_results[:, 0:100] = 1
# check_results[:, 100:] = 0

check_results[:, :] = 1

measurements = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, measurement_noise_covariance, sensor_position_vals, check_results, seed)

def EKF_dynamics_equation(t, X, mu, process_noise_covariance):

    state = X[0:6]
    covariance = X[6:42].reshape((6, 6))

    jacobian = CR3BP_jacobian(state, mu)

    ddt_state = CR3BP_DEs(t, state, mu)
    ddt_covariance = jacobian @ covariance + covariance @ jacobian.T + process_noise_covariance

    return np.concatenate((ddt_state, ddt_covariance.flatten()))
    
def EKF_measurement_equation(time_index, X, mu, sensor_position_vals, individual_measurement_size):

    num_sensors = int(np.size(sensor_position_vals, 0)/3)
    measurement = np.empty(num_sensors*individual_measurement_size)
    measurement_jacobian = np.empty((num_sensors*individual_measurement_size, 6))

    for sensor_index in np.arange(num_sensors):
        sensor_position = sensor_position_vals[sensor_index*3:(sensor_index+1)*3, time_index]
        
        measurement[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor(X, sensor_position)
        measurement_jacobian[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor_jacobian(X, sensor_position)

    return measurement, measurement_jacobian

def costate_dynamics_equation(t, X, mu, umax, process_noise_covariance):

    state = X[0:6]
    costate = X[6:12]
    covariance = X[12:156].reshape((12, 12))

    jacobian = CR3BP_costate_jacobian(state, costate, mu, umax)

    ddt_state = minimum_energy_ODE(0, X[0:12], mu, umax)
    ddt_covariance = jacobian @ covariance + covariance @ jacobian.T + process_noise_covariance

    return np.concatenate((ddt_state, ddt_covariance.flatten()))
    
def costate_measurement_equation(time_index, X, mu, sensor_position_vals, individual_measurement_size):

    num_sensors = int(np.size(sensor_position_vals, 0)/3)
    measurement = np.empty(num_sensors*individual_measurement_size)
    measurement_jacobian = np.empty((num_sensors*individual_measurement_size, 12))

    for sensor_index in np.arange(num_sensors):
        sensor_position = sensor_position_vals[sensor_index*3:(sensor_index+1)*3, time_index]
        
        measurement[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor(X, sensor_position)
        measurement_jacobian[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor_jacobian_costate(X, sensor_position)

    return measurement, measurement_jacobian

the_factor = 5

first_dynamics_args = (mu,)
second_dynamics_args = (mu, umax)
first_measurement_args = (mu, sensor_position_vals[:, 0:the_factor+1], individual_measurement_size)
second_measurement_args = (mu, sensor_position_vals[:, the_factor:], individual_measurement_size)

first_measurements = Measurements(measurements.t[0:the_factor], measurements.measurements[:, 0:the_factor], 2)
second_measurements = Measurements(measurements.t[the_factor:], measurements.measurements[:, the_factor:], 2)

first_output = run_EKF(initial_estimate[0:6], initial_covariance[0:6, 0:6],
                       EKF_dynamics_equation, EKF_measurement_equation,
                       first_measurements, EKF_process_noise_covriance,
                       filter_measurement_covariance,
                       first_dynamics_args, first_measurement_args)

first_filter_time = first_output.t
first_anterior_estimate_vals = first_output.anterior_estimate_vals
first_posterior_estimate_vals = first_output.posterior_estimate_vals
first_posterior_covariance_vals = first_output.posterior_covariance_vals

updated_estimate = first_posterior_estimate_vals[:, -1]
updated_covariance = first_posterior_covariance_vals[:, :, -1]

updated_estimate = np.concatenate((updated_estimate, initial_estimate[6:12]))
updated_covariance = np.vstack((np.hstack((updated_covariance, np.zeros((6, 6)))), np.hstack((np.zeros((6, 6)), initial_covariance[6:12, 6:12]))))
# updated_covariance = initial_covariance

second_output = run_EKF(updated_estimate, updated_covariance,
                        costate_dynamics_equation, costate_measurement_equation,
                        second_measurements, process_noise_covariance,
                        filter_measurement_covariance, 
                        second_dynamics_args, second_measurement_args)

second_filter_time = second_output.t
second_anterior_estimate_vals = second_output.anterior_estimate_vals
second_posterior_estimate_vals = second_output.posterior_estimate_vals
second_posterior_covariance_vals = second_output.posterior_covariance_vals

first_anterior_estimate_vals = np.vstack((first_anterior_estimate_vals, 0*first_anterior_estimate_vals))
first_posterior_estimate_vals = np.vstack((first_posterior_estimate_vals, 0*first_posterior_estimate_vals))
first_posterior_estimate_vals = first_posterior_estimate_vals[:, 0:-1]
new_thing = np.empty((12, 12, len(first_filter_time) + 1))
for index in np.arange(len(first_filter_time) + 1):
    new_thing[:, :, index] = np.vstack((np.hstack((first_posterior_covariance_vals[:, :, index], np.zeros((6, 6)))), np.zeros((6, 12))))
first_posterior_covariance_vals = new_thing[:, :, 0:-1]
# first_posterior_covariance_vals = new_thing

anterior_estimate_vals = np.concatenate((first_anterior_estimate_vals, second_anterior_estimate_vals), 1)

posterior_estimate_vals = np.concatenate((first_posterior_estimate_vals, second_posterior_estimate_vals), 1)
posterior_covariance_vals = np.concatenate((first_posterior_covariance_vals, second_posterior_covariance_vals), 2)

print(np.size(first_posterior_estimate_vals, 1))
print(np.size(first_posterior_covariance_vals, 2))

ax = plt.figure().add_subplot()
ax.scatter(first_measurements.t, first_measurements.measurements[0])
ax.scatter(first_measurements.t, first_measurements.measurements[1])
ax.scatter(first_measurements.t, first_measurements.measurements[2])
ax.scatter(first_measurements.t, first_measurements.measurements[3])
ax.scatter(second_measurements.t, second_measurements.measurements[0])
ax.scatter(second_measurements.t, second_measurements.measurements[1])
ax.scatter(second_measurements.t, second_measurements.measurements[2])
ax.scatter(second_measurements.t, second_measurements.measurements[3])

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
ax.scatter(truth_vals[0], truth_vals[1], truth_vals[2])
ax.scatter(posterior_estimate_vals[0], posterior_estimate_vals[1], posterior_estimate_vals[2])
ax.scatter(anterior_estimate_vals[0], anterior_estimate_vals[1], anterior_estimate_vals[2])
# ax.scatter(truth_vals[0, 5:], truth_vals[1, 5:], truth_vals[2, 5:])
# ax.scatter(second_posterior_estimate_vals[0], second_posterior_estimate_vals[1], second_posterior_estimate_vals[2])
# ax.scatter(second_anterior_estimate_vals[0], second_anterior_estimate_vals[1], second_anterior_estimate_vals[2])
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
ax = fig.add_subplot(311)
ax.plot(time_vals, truth_control[0])
ax.plot(time_vals, posterior_control[0])
ax = fig.add_subplot(312)
ax.plot(time_vals, truth_control[1])
ax.plot(time_vals, posterior_control[1])
ax = fig.add_subplot(313)
ax.plot(time_vals, truth_control[2])
ax.plot(time_vals, posterior_control[2])

plt.show()