

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from configuration_IMM import *
from CR3BP import *
from CR3BP_pontryagin import *
from UKF import *
from smoothing import *
from helper_functions import *
from measurement_functions import *
from plotting import *

time_vals, truth_vals = generate_truth_vals(dynamics_equation, CR3BP_DEs, final_time, dt, initial_truth, backprop_time, additional_time, truth_dynamics_args)

def CR3BP_dynamics_equation(t, X, mu, umax):
    ddt_state = CR3BP_DEs(0, X, mu)
    return ddt_state

def CR3BP_dynamics_equation_STM(t, X, mu, umax):
    
    state = X[0:6]
    STM = X[6:42].reshape((6, 6))

    jacobian = CR3BP_jacobian(state, mu)

    ddt_state = CR3BP_DEs(0, state, mu)
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

initial_covariance = initial_state_covariance

initial_estimates = []
measurements = []
measurement_args = []
if vary_scenarios == False:
    sensor_phase = generator.uniform(0, 1)
    sun_phase = generator.uniform(0, 2*np.pi)

    sensor_position_vals = generate_sensor_positions(sensor_dynamics_equation, sensor_initial_conditions, (mu,), time_vals, sensor_phase, sensor_period)

    num_sensors = int(np.size(sensor_position_vals, 0)/3)
    earth_vectors = np.empty((3*num_sensors, len(time_vals)))
    moon_vectors = np.empty((3*num_sensors, len(time_vals)))
    sun_vectors = np.empty((3*num_sensors, len(time_vals)))
    for sensor_index in range(num_sensors):
        sensor_positions = sensor_position_vals[sensor_index*3:(sensor_index + 1)*3, :]
        earth_vectors[sensor_index*3:(sensor_index + 1)*3, :] = generate_earth_vectors(time_vals, sensor_positions)
        moon_vectors[sensor_index*3:(sensor_index + 1)*3, :] = generate_moon_vectors(time_vals, sensor_positions)
        sun_vectors[sensor_index*3:(sensor_index + 1)*3, :] = generate_sun_vectors(time_vals, sun_phase)

    earth_results = np.empty((num_sensors, len(time_vals)))
    moon_results = np.empty((num_sensors, len(time_vals)))
    sun_results = np.empty((num_sensors, len(time_vals)))
    check_results = np.empty((num_sensors, len(time_vals)))
    shadow_results = check_validity(time_vals, truth_vals[0:3, :], sensor_positions, sun_vectors[0:3, :], check_shadow, ())
    for sensor_index in range(num_sensors):
        sensor_positions = sensor_position_vals[sensor_index*3:(sensor_index + 1)*3, :]
        earth_results[sensor_index, :] = check_validity(time_vals, truth_vals[0:3, :], sensor_positions, earth_vectors[sensor_index*3:(sensor_index+1)*3, :], check_exclusion, (earth_exclusion_angle,))
        moon_results[sensor_index, :] = check_validity(time_vals, truth_vals[0:3, :], sensor_positions, moon_vectors[sensor_index*3:(sensor_index+1)*3, :], check_exclusion, (moon_exclusion_angle,))
        sun_results[sensor_index, :] = check_validity(time_vals, truth_vals[0:3, :], sensor_positions, sun_vectors[sensor_index*3:(sensor_index+1)*3, :], check_exclusion, (sun_exclusion_angle,))
        check_results[sensor_index, :] = earth_results[sensor_index, :] * moon_results[sensor_index, :] * sun_results[sensor_index, :] * shadow_results
        if gap == False:
            check_results[sensor_index, :] = 1

for run_index in range(num_runs):

    if vary_scenarios == True:
        sensor_phase = generator.uniform(0, 1)
        sun_phase = generator.uniform(0, 2*np.pi)

        sensor_position_vals = generate_sensor_positions(sensor_dynamics_equation, sensor_initial_conditions, (mu,), time_vals, sensor_phase, sensor_period)

        num_sensors = int(np.size(sensor_position_vals, 0)/3)
        earth_vectors = np.empty((3*num_sensors, len(time_vals)))
        moon_vectors = np.empty((3*num_sensors, len(time_vals)))
        sun_vectors = np.empty((3*num_sensors, len(time_vals)))
        for sensor_index in range(num_sensors):
            sensor_positions = sensor_position_vals[sensor_index*3:(sensor_index + 1)*3, :]
            earth_vectors[sensor_index*3:(sensor_index + 1)*3, :] = generate_earth_vectors(time_vals, sensor_positions)
            moon_vectors[sensor_index*3:(sensor_index + 1)*3, :] = generate_moon_vectors(time_vals, sensor_positions)
            sun_vectors[sensor_index*3:(sensor_index + 1)*3, :] = generate_sun_vectors(time_vals, sun_phase)

        earth_results = np.empty((num_sensors, len(time_vals)))
        moon_results = np.empty((num_sensors, len(time_vals)))
        sun_results = np.empty((num_sensors, len(time_vals)))
        check_results = np.empty((num_sensors, len(time_vals)))
        for sensor_index in range(num_sensors):
            sensor_positions = sensor_position_vals[sensor_index*3:(sensor_index + 1)*3, :]
            earth_results[sensor_index, :] = check_validity(time_vals, truth_vals[0:3, :], sensor_positions, earth_vectors[sensor_index*3:(sensor_index+1)*3, :], check_exclusion, (earth_exclusion_angle,))
            moon_results[sensor_index, :] = check_validity(time_vals, truth_vals[0:3, :], sensor_positions, moon_vectors[sensor_index*3:(sensor_index+1)*3, :], check_exclusion, (moon_exclusion_angle,))
            sun_results[sensor_index, :] = check_validity(time_vals, truth_vals[0:3, :], sensor_positions, sun_vectors[sensor_index*3:(sensor_index+1)*3, :], check_exclusion, (sun_exclusion_angle,))
            check_results[sensor_index, :] = earth_results[sensor_index, :] * moon_results[sensor_index, :] * sun_results[sensor_index, :]
    
    # check_results[:, :] = 1

    initial_estimates.append(generator.multivariate_normal(truth_vals[0:6, 0], initial_covariance))
    measurement_vals = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, measurement_noise_covariance, sensor_position_vals, check_results, generator)
    measurement_vals = angles2PV(measurement_vals)
    measurement_args.append((measurement_variances, sensor_position_vals, check_results))
    measurements.append(measurement_vals.measurements)

# filter_measurement_function = angles_measurement_equ`ation

dynamics_function = CR3BP_dynamics_equation
dynamics_function_args = (mu, umax)
filter_measurement_function = PV_measurement_equation


myUKF = UKF(dynamics_function,
            dynamics_function_args,
            filter_measurement_function,
            UKF_process_noise_covariance,
            ukf_parameters,
            0.5)

filter_output = myUKF.run_MC(initial_estimates,
                             initial_covariance,
                             time_vals,
                             measurements,
                             measurement_args)

filter_time = filter_output.t
anterior_estimates = filter_output.anterior_estimates
anterior_covariances = filter_output.anterior_covariances
posterior_estimates = filter_output.posterior_estimates
posterior_covariances = filter_output.posterior_covariances

anterior_estimate_vals = anterior_estimates[0]
posterior_estimate_vals = posterior_estimates[0]
anterior_covariance_vals = anterior_covariances[0]
posterior_covariance_vals = posterior_covariances[0]

maneuver_start_index = run_maneuver_detection_alg(CR3BP_dynamics_equation_STM,
                                                  dynamics_function_args,
                                                  filter_time,
                                                  anterior_estimate_vals,
                                                  posterior_estimate_vals,
                                                  anterior_covariance_vals,
                                                  posterior_covariance_vals,
                                                  horizon,
                                                  detection_threshold)

print(time_vals[maneuver_start_index] * NONDIM_TIME_DAYS)

estimation_errors = compute_estimation_errors(truth_vals, posterior_estimates, (0, 6))
three_sigmas = compute_3sigmas(posterior_covariances, (0, 6))

avg_error_vals = compute_avg_error(estimation_errors, (0, 6))
avg_error_vals[0:3] *= NONDIM_LENGTH
avg_error_vals[3:6] *= NONDIM_LENGTH*1e3/NONDIM_TIME

position_norm_errors = compute_norm_errors(estimation_errors, (0, 3))
velocity_norm_errors = compute_norm_errors(estimation_errors, (3, 6))

avg_position_norm_errors = compute_avg_error(position_norm_errors, (0, 1)) * NONDIM_LENGTH
avg_velocity_norm_errors = compute_avg_error(velocity_norm_errors, (0, 1)) * NONDIM_LENGTH*1e3/NONDIM_TIME

plot_time = time_vals * NONDIM_TIME_HR/24

plot_3sigma(time_vals, estimation_errors, three_sigmas, "position", scale="linear", alpha=0.25)
plot_3sigma(time_vals, estimation_errors, three_sigmas, "velocity", scale="linear", alpha=0.25)
# plot_3sigma(time_vals, estimation_errors, three_sigmas, "position", scale="log", alpha=0.25)
# plot_3sigma(time_vals, estimation_errors, three_sigmas, "velocity", scale="log", alpha=0.25)

mae_fig = plt.figure()
ax = mae_fig.add_subplot(211)
ax.plot(plot_time, avg_position_norm_errors[0])
ax.set_ylabel("position")
ax = mae_fig.add_subplot(212)
ax.plot(plot_time, avg_velocity_norm_errors[0])
ax.set_ylabel("velocity")

ax = plt.figure(layout="constrained").add_subplot()
for run_index in range(num_runs):
    ax.step(plot_time, np.sum(measurement_args[run_index][2], axis=0), alpha=0.25)
ax.set_ylabel("num sensors")

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
for run_index in range(num_runs):
    ax.plot(posterior_estimates[run_index][0], posterior_estimates[run_index][1], posterior_estimates[run_index][2], alpha=0.25)
plot_moon(ax, mu, "nd")
ax.set_aspect("equal")

plt.show(block=False)
plt.pause(0.001)
input("press [enter] to close")
plt.close("all")