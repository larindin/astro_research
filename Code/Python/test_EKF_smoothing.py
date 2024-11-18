
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from configuration import *
from CR3BP import *
from CR3BP_pontryagin import *
from EKF import *
from EKF_smoothing import *
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

# check_results[:, 50:] = 0
# check_results[:, 100:] = 1
# check_results[:, 150:] = 0

measurements = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, measurement_noise_covariance, sensor_position_vals, check_results, seed)

dynamics_args = (mu, )
measurement_args = (mu, sensor_position_vals, individual_measurement_size)

def EKF_dynamics_equation(t, X, mu):

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

    for sensor_index in np.arange(num_sensors):
        sensor_position = sensor_position_vals[sensor_index*3:(sensor_index+1)*3, time_index]
        
        measurement[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor(X, sensor_position)
        measurement_jacobian[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor_jacobian(X, sensor_position)

    return measurement, measurement_jacobian


filter_output = run_EKF(initial_estimate, initial_covariance,
                        EKF_dynamics_equation, EKF_measurement_equation,
                        measurements, process_noise_covariance,
                        measurement_noise_covariance, 
                        dynamics_args, measurement_args)

filter_time = filter_output.t
posterior_estimate_vals = filter_output.posterior_estimate_vals
posterior_covariance_vals = filter_output.posterior_covariance_vals
anterior_estimate_vals = filter_output.anterior_estimate_vals
anterior_covariance_vals = filter_output.anterior_covariance_vals
innovations = filter_output.innovations_vals
weight_vals = np.ones((1, len(filter_time)))

smoothed_results = run_EKF_smoothing(filter_output)

plot_GM_heatmap(truth_vals, posterior_estimate_vals[:, :, np.newaxis], posterior_covariance_vals[:, :, :, np.newaxis], weight_vals, -1, xbounds=[0.75, 1.25], ybounds=[-0.25, 0.25], resolution=51)
plot_GM_heatmap(truth_vals, posterior_estimate_vals[:, :, np.newaxis], posterior_covariance_vals[:, :, :, np.newaxis], weight_vals, -1, xbounds=[0.75, 1.25], ybounds=[-0.25, 0.25], resolution=51, state_indices=[0, 2])
# plot_GM_heatmap(truth_vals, posterior_estimate_vals[:, :, np.newaxis], posterior_covariance_vals[:, :, :, np.newaxis], weight_vals, -51, resolution=51)
# plot_GM_heatmap(truth_vals, posterior_estimate_vals[:, :, np.newaxis], posterior_covariance_vals[:, :, :, np.newaxis], weight_vals, -101, resolution=51)

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
ax.plot(posterior_estimate_vals[0], posterior_estimate_vals[1], posterior_estimate_vals[2])
ax.plot(smoothed_results[0, 1:], smoothed_results[1, 1:], smoothed_results[2, 1:])
ax.set_aspect("equal")

ax = plt.figure().add_subplot()
ax.plot(time_vals, 3*np.sqrt(posterior_covariance_vals[0, 0]))

ax = plt.figure().add_subplot()
ax.plot(measurements.t, anterior_estimate_vals[0])
ax.plot(measurements.t, anterior_estimate_vals[1])
ax.plot(measurements.t, anterior_estimate_vals[2])

posterior_estimates = [smoothed_results, posterior_estimate_vals]
posterior_covariances = [posterior_covariance_vals, posterior_covariance_vals]

estimation_errors = compute_estimation_errors(truth_vals, posterior_estimates, 6)
three_sigmas = compute_3sigmas(posterior_covariances, 6)
plot_3sigma(time_vals, estimation_errors, three_sigmas, 6, [-0.25, 0.25], 0.5)

truth_control = get_min_fuel_control(truth_vals[6:12, :], umax, truth_rho)
fig = plt.figure()
ax = fig.add_subplot(311)
ax.step(time_vals, truth_control[0])
ax = fig.add_subplot(312)
ax.step(time_vals, truth_control[1])
ax = fig.add_subplot(313)
ax.step(time_vals, truth_control[2])

plt.show()