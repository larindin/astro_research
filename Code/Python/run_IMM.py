

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
additional_time_vals = np.arange(forprop_time_vals[-1], forprop_time_vals[-1]+additional_time, dt)
backprop_tspan = np.array([backprop_time_vals[0], backprop_time_vals[-1]])
forprop_tspan = np.array([forprop_time_vals[0], forprop_time_vals[-1]])
additional_tspan = np.array([additional_time_vals[0], additional_time_vals[-1]])
back_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, backprop_tspan, initial_truth[0:6], args=(mu,), t_eval=backprop_time_vals, atol=1e-12, rtol=1e-12).y
back_propagation = np.vstack((back_propagation, np.full(np.shape(back_propagation), 1e-12)))
back_propagation = np.flip(back_propagation, axis=1)
forward_propagation = scipy.integrate.solve_ivp(dynamics_equation, forprop_tspan, initial_truth, args=truth_dynamics_args, t_eval=forprop_time_vals, atol=1e-12, rtol=1e-12).y
additional_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, additional_tspan, forward_propagation[0:6, -1], args=(mu,), t_eval=additional_time_vals, atol=1e-12, rtol=1e-12).y
additional_propagation = np.vstack((additional_propagation, np.full(np.shape(additional_propagation), 1e-12)))
truth_vals = np.concatenate((back_propagation[:, :-1], forward_propagation, additional_propagation[:, 1:]), axis=1)
time_vals = np.concatenate((np.flip(backprop_time_vals[1:]), forprop_time_vals, additional_time_vals[1:])) + abs(backprop_time_vals[-1])

# truth_vals = np.concatenate((back_propagation[:, :-1], forward_propagation), axis=1)
# time_vals = np.concatenate((np.flip(backprop_time_vals[1:]), forprop_time_vals)) + abs(backprop_time_vals[-1])

# truth_vals = back_propagation[:, :-1]
# time_vals = np.flip(backprop_time_vals[1:]) + abs(backprop_time_vals[-1])

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

def coasting_costate_dynamics_equation(t, X, mu, umax):

    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    ddt_state = CR3BP_DEs(t, state, mu)

    # natural costate dynamics
    ddt_costate = CR3BP_costate_DEs(0, state, costate, mu)
    jacobian = minimum_time_jacobian(state, costate, mu, umax)
    jacobian[0:6, 6:12] = 0

    # zero costate dynamics
    # K = np.diag(np.full(6, 0))
    # jacobian = coasting_costate_jacobian(state, mu, K)
    # ddt_costate = np.zeros(6)

    # # exponential decay
    # K = np.diag(np.full(6, 1e1))
    # jacobian = coasting_costate_jacobian(state, mu, K)
    # ddt_costate = -K @ costate
    
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
initial_covariance = scipy.linalg.block_diag(initial_state_covariance, initial_costate_covariance)
process_noise_covariances = [coasting_costate_process_noise_covariance, min_time_process_noise_covariance]

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
    check_results[:, :] = 1

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
        shadow_results = check_validity(time_vals, truth_vals[0:3, :], sensor_positions, sun_vectors[0:3, :], check_shadow)
        for sensor_index in range(num_sensors):
            sensor_positions = sensor_position_vals[sensor_index*3:(sensor_index + 1)*3, :]
            earth_results[sensor_index, :] = check_validity(time_vals, truth_vals[0:3, :], sensor_positions, earth_vectors[sensor_index*3:(sensor_index+1)*3, :], check_exclusion, (earth_exclusion_angle,))
            moon_results[sensor_index, :] = check_validity(time_vals, truth_vals[0:3, :], sensor_positions, moon_vectors[sensor_index*3:(sensor_index+1)*3, :], check_exclusion, (moon_exclusion_angle,))
            sun_results[sensor_index, :] = check_validity(time_vals, truth_vals[0:3, :], sensor_positions, sun_vectors[sensor_index*3:(sensor_index+1)*3, :], check_exclusion, (sun_exclusion_angle,))
            check_results[sensor_index, :] = earth_results[sensor_index, :] * moon_results[sensor_index, :] * sun_results[sensor_index, :] * shadow_results
    
    # check_results[:, :] = 1

    initial_estimates.append(np.concatenate((generator.multivariate_normal(truth_vals[0:6, 0], initial_state_covariance), np.ones(6)*1e0)))
    measurement_vals = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, measurement_noise_covariance, sensor_position_vals, check_results, generator)
    measurement_vals = angles2PV(measurement_vals)
    measurement_args.append((measurement_variances, sensor_position_vals, check_results))
    measurements.append(measurement_vals.measurements)

# filter_measurement_function = angles_measurement_equation
filter_measurement_function = PV_measurement_equation

dynamics_args = (mu, umax)
dynamics_functions = (coasting_dynamics_equation, maneuvering_dynamics_equation)
dynamics_functions_args = (dynamics_args, dynamics_args)


IMM = IMM_filter(dynamics_functions,
                 dynamics_functions_args,
                 filter_measurement_function,
                 process_noise_covariances,
                 mode_transition_matrix,
                 underweighting_ratio)

results = IMM.run_MC(initial_estimates,
                     initial_covariance,
                     initial_mode_probabilities,
                     time_vals,
                     measurements,
                     measurement_args)

filter_time = results.t
output_estimates = results.output_estimates
output_covariances = results.output_covariances
mode_probabilities = results.mode_probabilities

truth_control = get_min_fuel_control(truth_vals[6:12, :], umax, truth_rho)
estimated_controls = []
control_errors = []
control_covariances = []
for run_index in range(num_runs):
    posterior_control = get_min_time_control(output_estimates[run_index][6:12, :], umax)
    control_covariance_vals = get_min_time_ctrl_cov(output_estimates[run_index][6:12], output_covariances[run_index], umax)
    for index in range(3):
        posterior_control[index, :] *= mode_probabilities[run_index][1, :]
        # control_covariance_vals[index, index, :] *= mode_probabilities[run_index][1, :]**2
    estimated_controls.append(posterior_control)
    control_errors.append(posterior_control - truth_control)
    control_covariances.append(control_covariance_vals)
    

estimation_errors = compute_estimation_errors(truth_vals, output_estimates, (0, 12))
three_sigmas = compute_3sigmas(output_covariances, (0, 12))
control_3sigmas = compute_3sigmas(control_covariances, (0, 3))

position_norm_errors = compute_norm_errors(estimation_errors, (0, 3))
velocity_norm_errors = compute_norm_errors(estimation_errors, (3, 6))
ctrl_norm_errors = compute_norm_errors(control_errors, (0, 3))

avg_position_norm_errors = compute_avg_error(position_norm_errors, (0, 1)) * NONDIM_LENGTH
avg_velocity_norm_errors = compute_avg_error(velocity_norm_errors, (0, 1)) * NONDIM_LENGTH*1e3/NONDIM_TIME
avg_ctrl_norm_errors = compute_avg_error(ctrl_norm_errors, (0, 1)) * NONDIM_LENGTH*1e6/NONDIM_TIME**2

avg_error_vals = compute_avg_error(estimation_errors, (0, 6))
avg_error_vals[0:3] *= NONDIM_LENGTH
avg_error_vals[3:6] *= NONDIM_LENGTH*1e3/NONDIM_TIME
avg_ctrl_error_vals = compute_avg_error(control_errors, (0, 3))
avg_ctrl_error_vals *= NONDIM_LENGTH*1e6/NONDIM_TIME**2

output_estimated_control = np.array(estimated_controls) * NONDIM_LENGTH*1e6/NONDIM_TIME**2

avg_error_vals = np.vstack((avg_error_vals, avg_ctrl_error_vals))
avg_norm_error_vals = np.vstack((avg_position_norm_errors, avg_velocity_norm_errors, avg_ctrl_norm_errors))

lambda_ratios = np.empty((6, len(time_vals), num_runs))
for run_index in range(num_runs):
    lambda_ratios[:, :, run_index] = abs(estimation_errors[run_index][6:12, :] / three_sigmas[run_index][6:12, :])

if save == True:
    if gap == True:
        np.save("data/OCIMM_est_control1.npy", output_estimated_control)
        np.save("data/OCIMM_avg_error1.npy", avg_error_vals)
        np.save("data/OCIMM_avg_norm_error1.npy", avg_norm_error_vals)
        np.save("data/OCIMM_est_errors1.npy", estimation_errors)
        np.save("data/OCIMM_est_3sigmas1.npy", three_sigmas)
        np.save("data/OCIMM_ctrl_errors1.npy", control_errors)
        np.save("data/OCIMM_ctrl_3sigmas1.npy", control_3sigmas)
        np.save("data/OCIMM_mode_probabilities1.npy", mode_probabilities)
    elif gap == False:
        np.save("data/OCIMM_est_control.npy", output_estimated_control)
        np.save("data/OCIMM_avg_error.npy", avg_error_vals)
        np.save("data/OCIMM_avg_norm_error.npy", avg_norm_error_vals)
        np.save("data/OCIMM_est_errors.npy", estimation_errors)
        np.save("data/OCIMM_est_3sigmas.npy", three_sigmas)
        np.save("data/OCIMM_ctrl_errors.npy", control_errors)
        np.save("data/OCIMM_ctrl_3sigmas.npy", control_3sigmas)
        np.save("data/OCIMM_mode_probabilities.npy", mode_probabilities)

plot_3sigma(time_vals, estimation_errors, three_sigmas, "position", scale="linear", alpha=0.15)
plot_3sigma(time_vals, estimation_errors, three_sigmas, "velocity", scale="linear", alpha=0.15)
plot_3sigma(time_vals, control_errors, control_3sigmas, "control", scale="linear", alpha=0.15)
plot_3sigma(time_vals, estimation_errors, three_sigmas, "lambdar", scale="linear", alpha=0.15)
plot_3sigma(time_vals, estimation_errors, three_sigmas, "lambdav", scale="linear", alpha=0.15)
plot_3sigma(time_vals, estimation_errors, three_sigmas, "position", alpha=0.15)
plot_3sigma(time_vals, estimation_errors, three_sigmas, "velocity", alpha=0.15)
plot_3sigma(time_vals, control_errors, control_3sigmas, "control", alpha=0.15)
plot_3sigma(time_vals, estimation_errors, three_sigmas, "lambdar", alpha=0.15)
plot_3sigma(time_vals, estimation_errors, three_sigmas, "lambdav", alpha=0.15)

plot_time = time_vals * NONDIM_TIME_HR/24
# plot_time = time_vals

ax = plt.figure(layout="constrained").add_subplot()
for run_index in range(num_runs):
    ax.step(plot_time, np.sum(measurement_args[run_index][2], axis=0), alpha=0.25)
ax.set_ylabel("num sensors")

rmse_r_fig = plt.figure()
rmse_ax_labels = ["$x$", "$y$", "$z$"]
for ax_index in range(3):
    thing = int("31" + str(ax_index + 1))
    ax = rmse_r_fig.add_subplot(thing)
    ax.plot(plot_time, avg_error_vals[ax_index], alpha=0.75)
    ax.set_ylabel(rmse_ax_labels[ax_index])
ax.set_xlabel("Time [days]")

rmse_v_fig = plt.figure()
rmse_ax_labels = ["$v_x$", "$v_y$", "$v_z$"]
for ax_index in range(3):
    thing = int("31" + str(ax_index + 1))
    ax = rmse_v_fig.add_subplot(thing)
    ax.plot(plot_time, avg_error_vals[ax_index+3], alpha=0.75)
    ax.set_ylabel(rmse_ax_labels[ax_index])
ax.set_xlabel("Time [days]")

rmse_a_fig = plt.figure()
rmse_ax_labels = ["$a_x$", "$a_y$", "$a_z$"]
for ax_index in range(3):
    thing = int("31" + str(ax_index + 1))
    ax = rmse_a_fig.add_subplot(thing)
    ax.plot(plot_time, avg_ctrl_error_vals[ax_index], alpha=0.75)
    ax.set_ylabel(rmse_ax_labels[ax_index])
ax.set_xlabel("Time [days]")

control_fig = plt.figure()
control_ax_labels = ["$u_1$", "$u_2$", "$u_3$"]
for ax_index in range(3):
    thing = int("31" + str(ax_index + 1))
    ax = control_fig.add_subplot(thing)
    ax.scatter(plot_time, truth_control[ax_index], alpha=0.75, s=4)
    for run_index in range(num_runs):
        ax.scatter(plot_time, estimated_controls[run_index][ax_index], alpha=0.15, s=4)
    ax.set_ylabel(control_ax_labels[ax_index])
ax.set_xlabel("Time [days]")
control_fig.legend(["Truth", "Estimated"])

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2], alpha=0.75)
for run_index in range(num_runs):
    ax.plot(output_estimates[run_index][0], output_estimates[run_index][1], output_estimates[run_index][2], alpha=0.25)
plot_moon(ax, mu)
for sensor_index in range(3):
    ax.scatter(sensor_position_vals[sensor_index*3, 0], sensor_position_vals[sensor_index*3+1, 0], sensor_position_vals[sensor_index*3+2, 0])
ax.set_aspect("equal")

ax = plt.figure(layout="constrained").add_subplot()
for run_index in range(num_runs):
    ax.scatter(plot_time, mode_probabilities[run_index][0], color="black", alpha=0.15, s=4)
    ax.scatter(plot_time, mode_probabilities[run_index][1], color="red", alpha=0.15, s=4)

ax = plt.figure(layout="constrained").add_subplot()
for run_index in range(num_runs):
    ax.plot(plot_time, np.linalg.norm(output_estimates[run_index][6:9], axis=0))
ax.set_yscale("log")

ax = plt.figure(layout="constrained").add_subplot()
for run_index in range(num_runs):
    ax.plot(plot_time, np.linalg.norm(output_estimates[run_index][9:12], axis=0))
ax.set_yscale("log")

ax = plt.figure(layout="constrained").add_subplot()
for run_index in range(num_runs):
    ax.plot(plot_time, lambda_ratios[0, :, run_index])

ax = plt.figure(layout="constrained").add_subplot()
# ax.plot(plot_time, truth_vals[6])
# ax.plot(plot_time, truth_vals[7])
# ax.plot(plot_time, truth_vals[8])
for run_index in range(num_runs):
    ax.plot(plot_time, output_estimates[run_index][6])
    ax.plot(plot_time, output_estimates[run_index][7])
    ax.plot(plot_time, output_estimates[run_index][8])
ax.set_ylim(-200, 200)
# ax.set_yscale("log")

ax = plt.figure(layout="constrained").add_subplot()
# ax.plot(plot_time, truth_vals[9])
# ax.plot(plot_time, truth_vals[10])
# ax.plot(plot_time, truth_vals[11])
for run_index in range(num_runs):
    ax.plot(plot_time, output_estimates[run_index][9])
    ax.plot(plot_time, output_estimates[run_index][10])
    ax.plot(plot_time, output_estimates[run_index][11])
# ax.set_ylim(-5, 5)
# ax.set_yscale("log")

plt.show(block=False)
plt.pause(0.001) # Pause for interval seconds.
input("hit[enter] to end.")
plt.close('all')