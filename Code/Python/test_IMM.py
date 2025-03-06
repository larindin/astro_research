

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from configuration_IMM import *
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

measurements = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, 2, measurement_noise_covariance, sensor_position_vals, check_results, seed)
measurements = angles2PV(measurements)

def coasting_dynamics_equation(t, X, mu, umax, rho):

    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    jacobian = minimum_time_jacobian(state, costate, mu, umax)

    ddt_state = CR3BP_DEs(t, state, mu)
    ddt_costate = CR3BP_costate_DEs(0, state, costate, mu)
    # K = np.diag(np.ones(6)*1e1)
    # ddt_costate = -K @ costate
    ddt_STM = jacobian @ STM

    return np.concatenate((ddt_state, ddt_costate, ddt_STM.flatten()))

def min_time_dynamics_equation(t, X, mu, umax, rho):

    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    jacobian = minimum_time_jacobian(state, costate, mu, umax)
    
    ddt_state = minimum_time_ODE(0, X[0:12], mu, umax)
    ddt_STM = jacobian @ STM

    return np.concatenate((ddt_state, ddt_STM.flatten()))

def min_energy_dynamics_equation(t, X, mu, umax, rho):
    
    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    jacobian = minimum_energy_jacobian(state, costate, mu, umax)

    ddt_state = minimum_energy_ODE(0, X[0:12], mu, umax)
    ddt_STM = jacobian @ STM

    return np.concatenate((ddt_state, ddt_STM.flatten()))

def min_fuel_dynamics_equation(t, X, mu, umax, rho):

    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    jacobian = minimum_fuel_jacobian(state, costate, mu, umax, rho)

    ddt_state = minimum_fuel_ODE(0, X[0:12], mu, umax, rho)
    ddt_STM = jacobian @ STM

    return np.concatenate((ddt_state, ddt_STM.flatten()))
    
def angles_measurement_equation(time_index, X, mu, sensor_position_vals, individual_measurement_size, input_measurement_covariance):

    num_sensors = int(np.size(sensor_position_vals, 0)/3)
    measurement = np.empty(num_sensors*individual_measurement_size)
    measurement_jacobian = np.empty((num_sensors*individual_measurement_size, 12))

    for sensor_index in range(num_sensors):
        sensor_position = sensor_position_vals[sensor_index*3:(sensor_index+1)*3, time_index]
        
        measurement[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor(X, sensor_position)
        measurement_jacobian[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor_jacobian_costate(X, sensor_position)

    return measurement, measurement_jacobian, input_measurement_covariance

def PV_measurement_equation(time_index, X, input_measurement_covariance, sensor_position_vals):

    num_sensors = int(np.size(sensor_position_vals, 0)/3)
    measurement = np.empty(num_sensors*3)
    measurement_jacobian = np.empty((num_sensors*3, len(X)))
    measurement_noise_covariance = np.empty((num_sensors*3, num_sensors*3))
    rs = np.empty(num_sensors)

    for sensor_index in range(num_sensors):
        sensor_position = sensor_position_vals[sensor_index*3:(sensor_index+1)*3, time_index]
        
        az, el = az_el_sensor(X, sensor_position)
        pvx, pvy, pvz = pointing_vector(X, sensor_position)
        PV = np.array([[pvx, pvy, pvz]]).T

        r = np.linalg.norm(X[0:3] - sensor_position)
        v1 = np.array([[-np.cos(az)*np.sin(el), -np.sin(az)*np.sin(el), np.cos(el)]]).T
        v2 = np.array([[np.sin(az), -np.cos(az), 0]]).T

        angle_noise_variance = input_measurement_covariance[0, 0]
        range_noise_variance = input_measurement_covariance[2, 2]

        new_measurement_noise_covariance = angle_noise_variance*(v1@v1.T + v2@v2.T) + range_noise_variance*r**2*PV@PV.T
        measurement_noise_covariance[sensor_index*3:(sensor_index+1)*3, sensor_index*3:(sensor_index+1)*3] = new_measurement_noise_covariance

        measurement[sensor_index*3:(sensor_index+1)*3] = PV.flatten()
        measurement_jacobian[sensor_index*3:(sensor_index+1)*3] = pointing_vector_jacobian()
        rs[sensor_index] = r

    return measurement, measurement_jacobian, measurement_noise_covariance, rs

filter_measurement_equation = PV_measurement_equation

dynamics_args = (mu, umax, truth_rho)
measurement_args = (sensor_position_vals,)
dynamics_equations = [coasting_dynamics_equation, min_time_dynamics_equation]
num_modes = len(dynamics_equations)


output = run_IMM(initial_estimate, initial_covariance, initial_mode_probabilities,
                 dynamics_equations, filter_measurement_equation, measurements,
                 process_noise_covariances, IMM_measurement_covariance,
                 dynamics_args, measurement_args, mode_transition_matrix)

anterior_estimate_vals = output.anterior_estimate_vals
mode_estimate_vals = output.posterior_estimate_vals
mode_covariance_vals = output.posterior_covariance_vals
weight_vals = output.weight_vals

posterior_estimate_vals, posterior_covariance_vals = compute_IMM_output(mode_estimate_vals, mode_covariance_vals, weight_vals)

truth_control = get_min_fuel_control(truth_vals[6:12, :], umax, truth_rho)
truth_control_norms = np.linalg.norm(truth_control, axis=0)/umax
estimated_control = get_min_fuel_control(posterior_estimate_vals[6:12], umax, truth_rho)

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
ax.plot(posterior_estimate_vals[0], posterior_estimate_vals[1], posterior_estimate_vals[2], alpha=0.25)
ax.set_aspect("equal")

ax = plt.figure().add_subplot()
for mode_index in range(num_modes):
    ax.step(time_vals, weight_vals[mode_index, :], alpha=0.25)
ax.step(time_vals, truth_control_norms, alpha=0.75)

control_fig = plt.figure()
for ax_index in range(3):
    thing = int("31" + str(ax_index+1))
    ax = control_fig.add_subplot(thing)
    ax.step(time_vals, truth_control[ax_index], alpha=0.75)
    ax.step(time_vals, estimated_control[ax_index], alpha=0.5)

errors = compute_estimation_errors(truth_vals, [posterior_estimate_vals], 12)
three_sigmas = compute_3sigmas([posterior_covariance_vals], 12)

plot_3sigma(time_vals, errors, three_sigmas, 6)

# fig = plt.figure()
# ax = fig.add_subplot(231)
# ax.plot(time_vals, truth_vals[6])
# for index in range(num_modes):
#     ax.step(time_vals, posterior_estimate_vals[6, :, index], alpha=0.25)
# ax = fig.add_subplot(232)
# ax.plot(time_vals, truth_vals[7])
# for index in range(num_modes):
#     ax.step(time_vals, posterior_estimate_vals[7, :, index], alpha=0.25)
# ax = fig.add_subplot(233)
# ax.plot(time_vals, truth_vals[8])
# for index in range(num_modes):
#     ax.step(time_vals, posterior_estimate_vals[8, :, index], alpha=0.25)
# ax = fig.add_subplot(234)
# ax.plot(time_vals, truth_vals[9])
# for index in range(num_modes):
#     ax.step(time_vals, posterior_estimate_vals[9, :, index], alpha=0.25)
# ax = fig.add_subplot(235)
# ax.plot(time_vals, truth_vals[10])
# for index in range(num_modes):
#     ax.step(time_vals, posterior_estimate_vals[10, :, index], alpha=0.25)
# ax = fig.add_subplot(236)
# ax.plot(time_vals, truth_vals[11])
# for index in range(num_modes):
#     ax.step(time_vals, posterior_estimate_vals[11, :, index], alpha=0.25)

# estimated_controls = []
# for index in range(num_modes):
#     estimated_control = get_min_fuel_control(posterior_estimate_vals[6:12, :, index], umax, filter_rho)
#     estimated_controls.append(estimated_control)
# fig = plt.figure()
# ax = fig.add_subplot(311)
# ax.step(time_vals, truth_control[0])
# for index in range(num_modes):
#     ax.step(time_vals, estimated_controls[index][0], alpha=0.25)
# ax = fig.add_subplot(312)
# ax.step(time_vals, truth_control[1])
# for index in range(num_modes):
#     ax.step(time_vals, estimated_controls[index][1], alpha=0.25)
# ax = fig.add_subplot(313)
# ax.step(time_vals, truth_control[2])
# for index in range(num_modes):
#     ax.step(time_vals, estimated_controls[index][2], alpha=0.25)

plt.show()