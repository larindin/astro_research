
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from configuration_OCBE import *
from CR3BP import *
from CR3BP_pontryagin import *
from OCBE import *
from smoothing import *
from measurement_functions import *
from plotting import *

backprop_time_vals = -np.arange(0, backprop_time, dt)
forprop_time_vals = np.arange(0, final_time, dt)
backprop_tspan = np.array([backprop_time_vals[0], backprop_time_vals[-1]])
forprop_tspan = np.array([forprop_time_vals[0], forprop_time_vals[-1]])
back_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, backprop_tspan, initial_truth[0:6], args=(mu,), t_eval=backprop_time_vals, atol=1e-12, rtol=1e-12).y
back_propagation = np.vstack((back_propagation, np.ones(np.shape(back_propagation))*1e-9))
back_propagation = np.flip(back_propagation, axis=1)
forward_propagation = scipy.integrate.solve_ivp(dynamics_equation, forprop_tspan, initial_truth, args=truth_dynamics_args, t_eval=forprop_time_vals, atol=1e-12, rtol=1e-12).y
# truth_vals = np.concatenate((back_propagation, forward_propagation), axis=1)
# time_vals = np.arange(0, final_time+backprop_time, dt)
truth_vals = forward_propagation
time_vals = forprop_time_vals

initial_estimate = generator.multivariate_normal(truth_vals[0:6, 0], initial_covariance)

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
# check_results[:, 100:] = 1
# check_results[:, 150:] = 0

measurements = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, measurement_noise_covariance, sensor_position_vals, check_results, seed)

dynamics_args = (mu, control_noise_covariance)
measurement_args = (mu, sensor_position_vals, individual_measurement_size)

def OCBE_dynamics_equation(t, X, mu, control_noise_covariance):

    state = X[0:6]
    STM = X[6:150].reshape((12, 12))

    B = np.vstack((np.zeros((3, 3)), np.eye(3)))

    jacobian = CR3BP_jacobian(state, mu)
    big_A = np.block([[jacobian, -B@control_noise_covariance@B.T], [np.zeros((6, 6)), -jacobian.T]])

    ddt_state = CR3BP_DEs(t, state, mu)
    ddt_STM = big_A @ STM

    return np.concatenate((ddt_state, ddt_STM.flatten()))
    
def OCBE_measurement_equation(time_index, X, mu, sensor_position_vals, individual_measurement_size):

    num_sensors = int(np.size(sensor_position_vals, 0)/3)
    measurement = np.empty(num_sensors*individual_measurement_size)
    measurement_jacobian = np.empty((num_sensors*individual_measurement_size, 6))

    for sensor_index in range(num_sensors):
        sensor_position = sensor_position_vals[sensor_index*3:(sensor_index+1)*3, time_index]
        
        measurement[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor(X, sensor_position)
        measurement_jacobian[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor_jacobian(X, sensor_position)

    return measurement, measurement_jacobian


filter_output = run_OCBE(initial_estimate, initial_covariance,
                        OCBE_dynamics_equation, OCBE_measurement_equation,
                        measurements, control_noise_covariance,
                        measurement_noise_covariance, 
                        dynamics_args, measurement_args)

smoothed_estimates, smoothed_covariance, smoothed_costate, smoothed_control = run_OCBE_smoothing(filter_output, control_noise_covariance)

filter_time = filter_output.t
posterior_estimate_vals = filter_output.posterior_estimate_vals
posterior_covariance_vals = filter_output.posterior_covariance_vals
anterior_estimate_vals = filter_output.anterior_estimate_vals
anterior_covariance_vals = filter_output.anterior_covariance_vals
innovations = filter_output.innovations_vals
STM_vals = filter_output.STM_vals
control_vals = filter_output.control_vals
costate_vals = filter_output.costate_vals
# weight_vals = np.ones((1, len(filter_time)))

# plot_GM_heatmap(truth_vals, posterior_estimate_vals[:, :, np.newaxis], posterior_covariance_vals[:, :, :, np.newaxis], weight_vals, -1, xbounds=[0.75, 1.25], ybounds=[-0.25, 0.25], resolution=51)
# plot_GM_heatmap(truth_vals, posterior_estimate_vals[:, :, np.newaxis], posterior_covariance_vals[:, :, :, np.newaxis], weight_vals, -1, xbounds=[0.75, 1.25], ybounds=[-0.25, 0.25], resolution=51, state_indices=[0, 2])
# plot_GM_heatmap(truth_vals, posterior_estimate_vals[:, :, np.newaxis], posterior_covariance_vals[:, :, :, np.newaxis], weight_vals, -51, resolution=51)
# plot_GM_heatmap(truth_vals, posterior_estimate_vals[:, :, np.newaxis], posterior_covariance_vals[:, :, :, np.newaxis], weight_vals, -101, resolution=51)

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
ax.plot(posterior_estimate_vals[0], posterior_estimate_vals[1], posterior_estimate_vals[2])
ax.set_aspect("equal")

posterior_estimates = [posterior_estimate_vals]
posterior_covariances = [posterior_covariance_vals]

estimation_errors = compute_estimation_errors(truth_vals, posterior_estimates, 6)
three_sigmas = compute_3sigmas(posterior_covariances, 6)
plot_3sigma(time_vals, estimation_errors, three_sigmas, 6, alpha=0.75)

smoothed_estimation_errors = compute_estimation_errors(truth_vals, [smoothed_estimates], 6)
smoothed_three_sigmas = compute_3sigmas([smoothed_covariance], 6)
plot_3sigma(time_vals, smoothed_estimation_errors, smoothed_three_sigmas, 6, alpha=0.75)

truth_control = get_min_fuel_control(truth_vals[6:12, :], umax, truth_rho)
control_vals[:, 0] = 0
# control_vals = np.cumsum(control_vals, axis=1)
control_fig = plt.figure()
for ax_index in range(3):
    thing = int("31" + str(ax_index+1))
    ax = control_fig.add_subplot(thing)
    # ax.plot(time_vals, truth_control[ax_index])
    ax.plot(time_vals, control_vals[ax_index])
    ax.plot(time_vals, smoothed_control[ax_index])

control_fig = plt.figure()
for ax_index in range(3):
    thing = int("31" + str(ax_index+1))
    ax = control_fig.add_subplot(thing)
    ax.plot(time_vals, truth_control[ax_index])
    # ax.plot(time_vals, control_vals[ax_index])

costate_fig = plt.figure()
costate_vals[:, 0] = 0
# costate_vals = np.cumsum(costate_vals, axis=1)
for ax_index in range(6):
    thing = int("61" + str(ax_index+1))
    ax = costate_fig.add_subplot(thing)
    # ax.plot(time_vals, truth_control[ax_index])
    ax.plot(time_vals, costate_vals[ax_index])
    ax.plot(time_vals, smoothed_costate[ax_index])

plt.show()