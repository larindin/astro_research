

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from configuration_IMM_fuel import *
from CR3BP import *
from CR3BP_pontryagin import *
from IMM import *
from dual_filter import *
from helper_functions import *
from measurement_functions import *
from plotting import *

time_vals = np.arange(0, final_time, dt)
tspan = np.array([time_vals[0], time_vals[-1]])
truth_propagation = scipy.integrate.solve_ivp(dynamics_equation, tspan, initial_truth, args=truth_dynamics_args, t_eval=time_vals, atol=1e-12, rtol=1e-12)
truth_vals = truth_propagation.y

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

# check_results[:, 50:] = 0

measurements = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, measurement_noise_covariance, sensor_position_vals, check_results, seed)

def coasting_dynamics_equation(t, X, mu, umax, rho, process_noise_covariance):

    state = X[0:6]
    costate = X[6:12]
    covariance = X[12:156].reshape((12, 12))

    jacobian = minimum_fuel_jacobian(state, costate, mu, umax, rho)

    ddt_state = CR3BP_DEs(t, state, mu)
    ddt_costate = np.zeros(6)
    ddt_covariance = jacobian @ covariance + covariance @ jacobian.T + process_noise_covariance

    return np.concatenate((ddt_state, ddt_costate, ddt_covariance.flatten()))

def thrusting_dynamics_equation(t, X, mu, umax, rho, process_noise_covariance):

    state = X[0:6]
    costate = X[6:12]
    covariance = X[12:156].reshape((12, 12))

    jacobian = minimum_fuel_jacobian(state, costate, mu, umax, rho)

    ddt_state = minimum_fuel_ODE(0, X[0:12], mu, umax, rho)
    ddt_covariance = jacobian @ covariance + covariance @ jacobian.T + process_noise_covariance

    return np.concatenate((ddt_state, ddt_covariance.flatten()))
    
def filter_measurement_equation(time_index, X, mu, sensor_position_vals, individual_measurement_size):

    num_sensors = int(np.size(sensor_position_vals, 0)/3)
    measurement = np.empty(num_sensors*individual_measurement_size)
    measurement_jacobian = np.empty((num_sensors*individual_measurement_size, 12))

    for sensor_index in range(num_sensors):
        sensor_position = sensor_position_vals[sensor_index*3:(sensor_index+1)*3, time_index]
        
        measurement[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor(X, sensor_position)
        measurement_jacobian[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor_jacobian_costate(X, sensor_position)

    return measurement, measurement_jacobian


dynamics_args = (mu, umax, filter_rho)
measurement_args = (mu, sensor_position_vals, individual_measurement_size)
dynamics_equations = [coasting_dynamics_equation, thrusting_dynamics_equation]
num_modes = len(dynamics_equations)


output = run_IMM(initial_estimate, initial_covariance, initial_mode_probabilities,
                    dynamics_equations, filter_measurement_equation, measurements,
                    process_noise_covariances, filter_measurement_covariance,
                    dynamics_args, measurement_args, mode_transition_matrix)

anterior_estimate_vals = output.anterior_estimate_vals
posterior_estimate_vals = output.posterior_estimate_vals
posterior_covariance_vals = output.posterior_covariance_vals
weight_vals = output.weight_vals

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
for mode_index in range(num_modes):
    ax.plot(posterior_estimate_vals[0, :, mode_index], posterior_estimate_vals[1, :, mode_index], posterior_estimate_vals[2, :, mode_index], alpha=0.25)
ax.set_aspect("equal")

ax = plt.figure().add_subplot()
for mode_index in range(num_modes):
    ax.step(time_vals, weight_vals[mode_index, :], alpha=0.25)

fig = plt.figure()
ax = fig.add_subplot(231)
ax.plot(time_vals, truth_vals[6])
for index in range(num_modes):
    ax.step(time_vals, posterior_estimate_vals[6, :, index], alpha=0.25)
ax = fig.add_subplot(232)
ax.plot(time_vals, truth_vals[7])
for index in range(num_modes):
    ax.step(time_vals, posterior_estimate_vals[7, :, index], alpha=0.25)
ax = fig.add_subplot(233)
ax.plot(time_vals, truth_vals[8])
for index in range(num_modes):
    ax.step(time_vals, posterior_estimate_vals[8, :, index], alpha=0.25)
ax = fig.add_subplot(234)
ax.plot(time_vals, truth_vals[9])
for index in range(num_modes):
    ax.step(time_vals, posterior_estimate_vals[9, :, index], alpha=0.25)
ax = fig.add_subplot(235)
ax.plot(time_vals, truth_vals[10])
for index in range(num_modes):
    ax.step(time_vals, posterior_estimate_vals[10, :, index], alpha=0.25)
ax = fig.add_subplot(236)
ax.plot(time_vals, truth_vals[11])
for index in range(num_modes):
    ax.step(time_vals, posterior_estimate_vals[11, :, index], alpha=0.25)

truth_control = get_min_fuel_control(truth_vals[6:12, :], umax, truth_rho)
estimated_controls = []
for index in range(num_modes):
    estimated_control = get_min_fuel_control(posterior_estimate_vals[6:12, :, index], umax, filter_rho)
    estimated_controls.append(estimated_control)
fig = plt.figure()
ax = fig.add_subplot(311)
ax.step(time_vals, truth_control[0])
for index in range(num_modes):
    ax.step(time_vals, estimated_controls[index][0], alpha=0.25)
ax = fig.add_subplot(312)
ax.step(time_vals, truth_control[1])
for index in range(num_modes):
    ax.step(time_vals, estimated_controls[index][1], alpha=0.25)
ax = fig.add_subplot(313)
ax.step(time_vals, truth_control[2])
for index in range(num_modes):
    ax.step(time_vals, estimated_controls[index][2], alpha=0.25)

plt.show()