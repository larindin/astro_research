

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from configuration_IMM import *
from CR3BP import *
from CR3BP_pontryagin import *
from IMM import *
from helper_functions import *
from measurement_functions import *
from plotting import *

backprop_time_vals = -np.arange(0, backprop_time, dt)
forprop_time_vals = np.arange(0, final_time, dt)
backprop_tspan = np.array([backprop_time_vals[0], backprop_time_vals[-1]])
forprop_tspan = np.array([forprop_time_vals[0], forprop_time_vals[-1]])
back_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, backprop_tspan, initial_truth[0:6], args=(mu,), t_eval=backprop_time_vals, atol=1e-12, rtol=1e-12).y
back_propagation = np.vstack((back_propagation, np.full(np.shape(back_propagation), 1e-12)))
back_propagation = np.flip(back_propagation, axis=1)
forward_propagation = scipy.integrate.solve_ivp(dynamics_equation, forprop_tspan, initial_truth, args=truth_dynamics_args, t_eval=forprop_time_vals, atol=1e-12, rtol=1e-12).y
truth_vals = np.concatenate((back_propagation[:, :-1], forward_propagation), axis=1)
time_vals = np.concatenate((np.flip(backprop_time_vals[1:]), forprop_time_vals)) + abs(backprop_time_vals[-1])

# ax = plt.figure().add_subplot(projection="3d")
# ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
# plot_moon(ax, mu)
# ax.set_aspect("equal")

# plt.show()
# quit()

# time_vals = np.arange(0, final_time, dt)
# tspan = np.array([time_vals[0], time_vals[-1]])
# truth_propagation = scipy.integrate.solve_ivp(dynamics_equation, tspan, initial_truth, args=truth_dynamics_args, t_eval=time_vals, atol=1e-12, rtol=1e-12)
# truth_vals = truth_propagation.y

sensor_position_vals = generate_sensor_positions(sensor_dynamics_equation, sensor_initial_conditions, (mu,), time_vals)
# sensor_position_vals = np.zeros((3, len(time_vals)))
# sensor_position_vals[0] = L2

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
# check_results[:, 250:] = 1

measurements = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, measurement_noise_covariance, sensor_position_vals, check_results, seed)

def coasting_costate_dynamics_equation(t, X, mu, umax):

    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    ddt_state = CR3BP_DEs(t, state, mu)
    # ddt_costate = CR3BP_costate_DEs(0, state, costate, mu)
    # jacobian = minimum_energy_jacobian(state, costate, mu, umax)
    K = np.diag(np.full(6, 1e0))
    jacobian = coasting_costate_jacobian(state, mu, K)
    ddt_costate = -K @ costate
    ddt_STM = jacobian @ STM

    return np.concatenate((ddt_state, ddt_costate, ddt_STM.flatten()))

def min_energy_dynamics_equation(t, X, mu, umax):
    
    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    jacobian = minimum_energy_jacobian(state, costate, mu, umax)

    ddt_state = minimum_energy_ODE(0, X[0:12], mu, umax)
    ddt_STM = jacobian @ STM

    return np.concatenate((ddt_state, ddt_STM.flatten()))

def min_time_dynamics_equation(t, X, mu, umax):

    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    jacobian = minimum_time_jacobian(state, costate, mu, umax)

    ddt_state = minimum_time_ODE(0, X[0:12], mu, umax)
    ddt_STM = jacobian @ STM

    return np.concatenate((ddt_state, ddt_STM.flatten()))

def angles_measurement_equation(time_index, X, measurement_variances, sensor_position_vals, check_results):

    check_result = check_results[:, time_index]
    angle_noise_variance = measurement_variances[0]

    num_sensors = len(check_result)
    valid_sensors = np.arange(0, num_sensors)[check_result == 1]
    num_valid_sensors = len(valid_sensors)
    measurement = np.empty(num_valid_sensors*2)
    measurement_jacobian = np.empty((num_valid_sensors*2, len(X[np.isnan(X) == False])))
    measurement_noise_covariance = np.zeros((num_valid_sensors*2, num_valid_sensors*2))

    for assignment_index, valid_sensor_index in enumerate(valid_sensors):

        sensor_position = sensor_position_vals[valid_sensor_index*3:(valid_sensor_index+1)*3, time_index]

        new_jacobian = az_el_sensor_jacobian_costate(X, sensor_position)
        new_measurement_covariance = np.diag([angle_noise_variance, angle_noise_variance])
        if np.isnan(X[-1]):
            new_jacobian = new_jacobian[:, 0:6]

        measurement[assignment_index*2:(assignment_index+1)*2] = az_el_sensor(X, sensor_position)
        measurement_jacobian[assignment_index*2:(assignment_index+1)*2] = new_jacobian
        measurement_noise_covariance[assignment_index*2:(assignment_index+1)*2, assignment_index*2:(assignment_index+1)*2] = new_measurement_covariance
    
    return measurement, measurement_jacobian, measurement_noise_covariance, 0

def PV_measurement_equation(time_index, X, measurement_variances, sensor_position_vals, check_results):
    
    check_result = check_results[:, time_index]
    angle_noise_variance = measurement_variances[0]
    range_noise_variance = measurement_variances[1]

    num_sensors = len(check_result)
    valid_sensors = np.arange(0, num_sensors)[check_result == 1]
    num_valid_sensors = len(valid_sensors)
    measurement = np.empty(num_valid_sensors*3)
    measurement_jacobian = np.empty((num_valid_sensors*3, len(X[np.isnan(X) == False])))
    measurement_noise_covariance = np.zeros((num_valid_sensors*3, num_valid_sensors*3))
    rs = np.empty(num_valid_sensors)
    
    for assignment_index, valid_sensor_index in enumerate(valid_sensors):

        sensor_position = sensor_position_vals[valid_sensor_index*3:(valid_sensor_index+1)*3, time_index]
        
        az, el = az_el_sensor(X, sensor_position)
        pvx, pvy, pvz = pointing_vector(X, sensor_position)
        PV = np.array([[pvx, pvy, pvz]]).T

        r = np.linalg.norm(X[0:3] - sensor_position)
        v1 = np.array([[-np.cos(az)*np.sin(el), -np.sin(az)*np.sin(el), np.cos(el)]]).T
        v2 = np.array([[np.sin(az), -np.cos(az), 0]]).T

        new_measurement_noise_covariance = angle_noise_variance*(v1@v1.T + v2@v2.T) + range_noise_variance*r**2*PV@PV.T
        new_jacobian = pointing_vector_jacobian(len(X[np.isnan(X) == False]))
        
        measurement[assignment_index*3:(assignment_index+1)*3] = PV.flatten()
        measurement_jacobian[assignment_index*3:(assignment_index+1)*3] = new_jacobian
        measurement_noise_covariance[assignment_index*3:(assignment_index+1)*3, assignment_index*3:(assignment_index+1)*3] = new_measurement_noise_covariance
        rs[assignment_index] = r

    return measurement, measurement_jacobian, measurement_noise_covariance, rs


coasting_dynamics_equation = coasting_costate_dynamics_equation
maneuvering_dynamics_equation = min_time_dynamics_equation
initial_estimate = np.concatenate((generator.multivariate_normal(truth_vals[0:6, 0], initial_state_covariance), np.ones(6)*1e-6))
initial_covariance = scipy.linalg.block_diag(initial_state_covariance, initial_costate_covariance)
process_noise_covariances = [coasting_costate_process_noise_covariance, min_time_process_noise_covariance]

# filter_measurement_function = angles_measurement_equation
filter_measurement_function = PV_measurement_equation
measurements = angles2PV(measurements)

dynamics_args = (mu, umax)
measurement_args = (measurement_variances, sensor_position_vals, check_results)
dynamics_functions = [coasting_dynamics_equation, maneuvering_dynamics_equation]
dynamics_functions_args = [dynamics_args, [mu, umax]]


IMM = IMM_filter(dynamics_functions,
                 dynamics_functions_args,
                 filter_measurement_function,
                 measurement_args,
                 process_noise_covariances,
                 mode_transition_matrix)

filter_output = IMM.run(initial_estimate, 
                        initial_covariance,
                        initial_mode_probabilities,
                        time_vals,
                        measurements.measurements)

filter_time = filter_output.t
posterior_estimate_vals = filter_output.posterior_estimate_vals
posterior_covariance_vals = filter_output.posterior_covariance_vals
anterior_estimate_vals = filter_output.anterior_estimate_vals
mode_probability_vals = filter_output.mode_probability_vals
output_estimate_vals = filter_output.output_estimate_vals
output_covariance_vals = filter_output.output_covariance_vals

truth_control = get_min_fuel_control(truth_vals[6:12, :], umax, truth_rho)
posterior_control = get_min_time_control(output_estimate_vals[6:12, :], umax)
for index in range(3):
    posterior_control[index, :] *= mode_probability_vals[1, :]

truth_primer_vectors = compute_primer_vectors(truth_vals[9:12])
estimated_primer_vectors = compute_primer_vectors(output_estimate_vals[9:12])

estimation_errors = compute_estimation_errors(truth_vals, [output_estimate_vals], 12)
three_sigmas = compute_3sigmas([output_covariance_vals], 12)

control_error = posterior_control - truth_control
control_3sigmas = get_min_energy_control_accel_cov([output_covariance_vals])
control_3sigmas[0][0] *= np.nan
control_3sigmas[0][1] *= np.nan
control_3sigmas[0][2] *= np.nan

thrusting_bool = mode_probability_vals[1] > 0.5

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
ax.plot(output_estimate_vals[0], output_estimate_vals[1], output_estimate_vals[2])
plot_moon(ax, mu)
ax.set_aspect("equal")

plot_3sigma(time_vals, [estimation_errors[0][0:3]], [three_sigmas[0][0:3]], "position")
plot_3sigma(time_vals, [estimation_errors[0][3:6]], [three_sigmas[0][3:6]], "velocity")
plot_3sigma(time_vals, [control_error], control_3sigmas, "acceleration")
plot_3sigma(time_vals, [estimation_errors[0][0:3]], [three_sigmas[0][0:3]], "position", scale="linear")
plot_3sigma(time_vals, [estimation_errors[0][3:6]], [three_sigmas[0][3:6]], "velocity", scale="linear")
plot_3sigma(time_vals, [control_error], control_3sigmas, "acceleration", scale="linear")
# plot_3sigma(time_vals, [estimation_errors[0][6:9]], [three_sigmas[0][6:9]], "lambdar")
# plot_3sigma(time_vals, [estimation_errors[0][9:12]], [three_sigmas[0][9:12]], "lambdav")

plot_mode_probabilities(time_vals, mode_probability_vals, truth_control)

control_fig = plt.figure()
control_ax_labels = ["$u_1$", "$u_2$", "$u_3$"]
for ax_index in range(3):
    thing = int("31" + str(ax_index + 1))
    ax = control_fig.add_subplot(thing)
    ax.plot(time_vals, truth_control[ax_index], alpha=0.5)
    ax.plot(time_vals, posterior_control[ax_index], alpha=0.5)
    ax.set_ylabel(control_ax_labels[ax_index])
ax.set_xlabel("Time [TU]")
control_fig.legend(["Truth", "Estimated"])

primer_vector_fig = plt.figure()
for ax_index in range(3):
    thing = int("31" + str(ax_index + 1))
    ax = primer_vector_fig.add_subplot(thing)
    ax.plot(time_vals, truth_primer_vectors[ax_index], alpha=0.5)
    ax.plot(time_vals, estimated_primer_vectors[ax_index], alpha=0.5)

costate_fig = plt.figure()
for ax_index in range(6):
    thing = int("61" + str(ax_index+1))
    ax = costate_fig.add_subplot(thing)
    ax.plot(time_vals, output_estimate_vals[ax_index+6])

plt.show()