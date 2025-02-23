

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from configuration_costate_fuel import *
from CR3BP import *
from CR3BP_pontryagin import *
from EKF import *
from dual_filter import *
from helper_functions import *
from measurement_functions import *
from plotting import *

time_vals = np.arange(0, final_time, dt)
tspan = np.array([time_vals[0], time_vals[-1]])
truth_propagation = scipy.integrate.solve_ivp(dynamics_equation, tspan, initial_truth, args=(mu, umax, truth_rho), t_eval=time_vals, atol=1e-12, rtol=1e-12)
truth_vals = truth_propagation.y
origin_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, tspan, initial_truth[0:6], args=(mu,), atol=1e-12, rtol=1e-12)
origin_vals = origin_propagation.y

sensor_position_vals = generate_sensor_positions(sensor_dynamics_equation, sensor_initial_conditions, (mu,), time_vals)

num_sensors = int(np.size(sensor_position_vals, 0)/3)
earth_vectors = np.empty((3*num_sensors, len(time_vals)))
moon_vectors = np.empty((3*num_sensors, len(time_vals)))
sun_vectors = np.empty((3*num_sensors, len(time_vals)))
for sensor_index in range(num_sensors):
    sensor_positions = sensor_position_vals[sensor_index*3:(sensor_index + 1)*3, :]
    earth_vectors[sensor_index*3:(sensor_index + 1)*3, :] = generate_earth_vectors(time_vals, sensor_positions)
    moon_vectors[sensor_index*3:(sensor_index + 1)*3, :] = generate_moon_vectors(time_vals, sensor_positions)
    sun_vectors[sensor_index*3:(sensor_index + 1)*3, :] = generate_sun_vectors(time_vals, 0)

earth_results = np.empty((num_sensors, len(time_vals)))
moon_results = np.empty((num_sensors, len(time_vals)))
sun_results = np.empty((num_sensors, len(time_vals)))
check_results = np.empty((num_sensors, len(time_vals)))
for sensor_index in range(num_sensors):
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

# check_results[:, 100:] = 0

measurements = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, measurement_noise_covariance, sensor_position_vals, check_results, seed)

def EKF_dynamics_equation(t, X, mu, process_noise_covariance):

    state = X[0:6]
    STM = X[6:42].reshape((6, 6))

    jacobian = CR3BP_jacobian(state, mu)

    ddt_state = CR3BP_DEs(t, state, mu)
    ddt_covariance = jacobian @ STM
    return np.concatenate((ddt_state, ddt_covariance.flatten()))
    
def EKF_measurement_equation(time_index, X, mu, sensor_position_vals, individual_measurement_size):

    num_sensors = int(np.size(sensor_position_vals, 0)/3)
    measurement = np.empty(num_sensors*individual_measurement_size)
    measurement_jacobian = np.empty((num_sensors*individual_measurement_size, 6))

    for sensor_index in range(num_sensors):
        sensor_position = sensor_position_vals[sensor_index*3:(sensor_index+1)*3, time_index]
        
        measurement[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor(X, sensor_position)
        measurement_jacobian[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor_jacobian(X, sensor_position)

    return measurement, measurement_jacobian

def costate_dynamics_equation(t, X, mu, umax, rho):

    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    jacobian = minimum_fuel_jacobian(state, costate, mu, umax, rho)

    ddt_state = minimum_fuel_ODE(0, X[0:12], mu, umax, rho)
    ddt_covariance = jacobian @ STM

    return np.concatenate((ddt_state, ddt_covariance.flatten()))
    
def costate_measurement_equation(time_index, X, mu, sensor_position_vals, individual_measurement_size):

    num_sensors = int(np.size(sensor_position_vals, 0)/3)
    measurement = np.empty(num_sensors*individual_measurement_size)
    measurement_jacobian = np.empty((num_sensors*individual_measurement_size, 12))

    for sensor_index in range(num_sensors):
        sensor_position = sensor_position_vals[sensor_index*3:(sensor_index+1)*3, time_index]
        
        measurement[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor(X, sensor_position)
        measurement_jacobian[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor_jacobian_costate(X, sensor_position)

    return measurement, measurement_jacobian


EKF_dynamics_args = (mu,)
costate_dynamics_args = (mu, umax, filter_rho)
EKF_measurement_args = (mu, sensor_position_vals, individual_measurement_size)
costate_measurement_args = (mu, sensor_position_vals, individual_measurement_size)

output = run_dual_filter(initial_estimate, initial_covariance, 
                         costate_dynamics_equation, costate_measurement_equation,
                         EKF_dynamics_equation, EKF_measurement_equation, measurements,
                         process_noise_covariance, filter_measurement_covariance,
                         EKF_process_noise_covriance, filter_measurement_covariance,
                         costate_dynamics_args, costate_measurement_args,
                         EKF_dynamics_args, EKF_measurement_args,
                         timeout_count, switching_count, filter_index)

anterior_estimate_vals = output.anterior_estimate_vals
posterior_estimate_vals = output.posterior_estimate_vals
posterior_covariance_vals = output.posterior_covariance_vals

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
ax.plot(sensor_position_vals[0], sensor_position_vals[1], sensor_position_vals[2])
ax.plot(sensor_position_vals[3], sensor_position_vals[4], sensor_position_vals[5])
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
ax.set_aspect("equal")

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
ax.plot(posterior_estimate_vals[0], posterior_estimate_vals[1], posterior_estimate_vals[2])
ax.plot(anterior_estimate_vals[0], anterior_estimate_vals[1], anterior_estimate_vals[2])
ax.set_aspect("equal")

posterior_estimates = [posterior_estimate_vals]
posterior_covariances = [posterior_covariance_vals]

estimation_errors = compute_estimation_errors(truth_vals, posterior_estimates, 12)
three_sigmas = compute_3sigmas(posterior_covariances, 12)
plot_3sigma(time_vals, estimation_errors, three_sigmas, 6, [-0.5, 0.5], 0.25)
plot_3sigma_costate(time_vals, estimation_errors, three_sigmas, 6, [-5, 5], 0.25)

truth_control = get_min_fuel_control(truth_vals[6:12, :], umax, truth_rho)
posterior_control = get_min_fuel_control(posterior_estimate_vals[6:12, :], umax, filter_rho)
ax = plt.figure().add_subplot()
ax.step(time_vals, np.linalg.norm(truth_control, axis=0))
ax.step(time_vals, np.linalg.norm(posterior_control, axis=0))

fig = plt.figure()
ax = fig.add_subplot(311)
ax.step(time_vals, truth_control[0])
ax.step(time_vals, posterior_control[0])
ax = fig.add_subplot(312)
ax.step(time_vals, truth_control[1])
ax.step(time_vals, posterior_control[1])
ax = fig.add_subplot(313)
ax.step(time_vals, truth_control[2])
ax.step(time_vals, posterior_control[2])

fig = plt.figure()
ax = fig.add_subplot(231)
ax.plot(time_vals, truth_vals[6])
ax.plot(time_vals, posterior_estimate_vals[6])
ax = fig.add_subplot(232)
ax.plot(time_vals, truth_vals[7])
ax.plot(time_vals, posterior_estimate_vals[7])
ax = fig.add_subplot(233)
ax.plot(time_vals, truth_vals[8])
ax.plot(time_vals, posterior_estimate_vals[8])
ax = fig.add_subplot(234)
ax.plot(time_vals, truth_vals[9])
ax.plot(time_vals, posterior_estimate_vals[9])
ax = fig.add_subplot(235)
ax.plot(time_vals, truth_vals[10])
ax.plot(time_vals, posterior_estimate_vals[10])
ax = fig.add_subplot(236)
ax.plot(time_vals, truth_vals[11])
ax.plot(time_vals, posterior_estimate_vals[11])

plt.show()