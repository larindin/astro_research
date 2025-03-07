

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from configuration_initial_IMM import *
from CR3BP import *
from CR3BP_pontryagin import *
from EKF import *
from IMM import *
from helper_functions import *
from measurement_functions import *
from plotting import *

backprop_time_vals = -np.arange(0, backprop_time, dt)
forprop_time_vals = np.arange(0, final_time, dt)
backprop_tspan = np.array([backprop_time_vals[0], backprop_time_vals[-1]])
forprop_tspan = np.array([forprop_time_vals[0], forprop_time_vals[-1]])
back_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, backprop_tspan, initial_truth[0:6], args=(mu,), t_eval=backprop_time_vals, atol=1e-12, rtol=1e-12).y
back_propagation = np.vstack((back_propagation, np.full(np.shape(back_propagation), np.nan)))
back_propagation = np.flip(back_propagation, axis=1)
forward_propagation = scipy.integrate.solve_ivp(dynamics_equation, forprop_tspan, initial_truth, args=truth_dynamics_args, t_eval=forprop_time_vals, atol=1e-12, rtol=1e-12).y
truth_vals = np.concatenate((back_propagation[:, :-1], forward_propagation), axis=1)
time_vals = np.concatenate((np.flip(backprop_time_vals[1:]), forprop_time_vals)) + abs(backprop_time_vals[-1])

initial_estimate = generator.multivariate_normal(truth_vals[0:6, 0], initial_state_covariance)

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

check_results[:, :] = 1

measurements = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, measurement_noise_covariance, sensor_position_vals, check_results, seed)

dynamics_args = (mu,)
measurement_args = (mu, sensor_position_vals, individual_measurement_size)
    
def coasting_dynamics_equation(t, X, mu):

    state = X[0:6]
    STM = X[6:42].reshape((6, 6))

    jacobian = CR3BP_jacobian(state, mu)

    ddt_state = CR3BP_DEs(t, state, mu)
    ddt_STM = jacobian @ STM

    return np.concatenate((ddt_state, ddt_STM.flatten()))

maneuvering_dynamics_equation = coasting_dynamics_equation

def EKF_measurement_equation(time_index, X, mu, sensor_position_vals, individual_measurement_size):

    num_sensors = int(np.size(sensor_position_vals, 0)/3)
    measurement = np.empty(num_sensors*individual_measurement_size)
    measurement_jacobian = np.empty((num_sensors*individual_measurement_size, 6))

    for sensor_index in range(num_sensors):
        sensor_position = sensor_position_vals[sensor_index*3:(sensor_index+1)*3, time_index]
        
        measurement[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor(X, sensor_position)
        measurement_jacobian[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor_jacobian(X, sensor_position)

    return measurement, measurement_jacobian

dynamics_equations = [coasting_dynamics_equation, maneuvering_dynamics_equation]

filter_output = run_IMM(initial_estimate, initial_covariance, 
                        initial_mode_probabilities, dynamics_equations,
                        EKF_measurement_equation, measurements,
                        process_noise_covariances, measurement_noise_covariance,
                        dynamics_args, measurement_args, mode_transition_matrix)

filter_time = filter_output.t
posterior_estimate_vals = filter_output.posterior_estimate_vals
posterior_covariance_vals = filter_output.posterior_covariance_vals
anterior_estimate_vals = filter_output.anterior_estimate_vals
mode_probabilities = filter_output.weight_vals

output_estimate_vals, output_covariance_vals = compute_IMM_output(posterior_estimate_vals, posterior_covariance_vals, mode_probabilities)

truth_control = get_min_fuel_control(truth_vals[6:12, :], umax, truth_rho)
first_order = (posterior_estimate_vals[3:6, :] - anterior_estimate_vals[3:6, :])/dt
estimated_control = (posterior_estimate_vals[3:6, :-1] - anterior_estimate_vals[3:6, :-1])/2/dt + (posterior_estimate_vals[3:6, 1:] - anterior_estimate_vals[3:6, 1:])/2/dt

estimation_errors = compute_estimation_errors(truth_vals, [output_estimate_vals], 6)
three_sigmas = compute_3sigmas([output_covariance_vals], 6)

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
ax.plot(output_estimate_vals[0], output_estimate_vals[1], output_estimate_vals[2])
plot_moon(ax, mu)
ax.set_aspect("equal")

plot_3sigma(time_vals, estimation_errors, three_sigmas, 6)

ax = plt.figure().add_subplot()
ax.plot(time_vals, mode_probabilities[0])
ax.plot(time_vals, mode_probabilities[1])
ax.plot(time_vals, np.linalg.norm(truth_control, axis=0))

control_fig = plt.figure()
for ax_index in range(3):
    thing = int("31" + str(ax_index + 1))
    ax = control_fig.add_subplot(thing)
    # ax.plot(time_vals, truth_control[ax_index])
    ax.plot(time_vals[:-1], estimated_control[ax_index, :, 0], alpha=0.5)
    ax.plot(time_vals, first_order[ax_index, :, 0], alpha=0.5)
    
plt.show()