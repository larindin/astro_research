
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from joblib import Parallel, delayed
from configuration_costate_reformulated import *
from CR3BP import *
from CR3BP_pontryagin_reformulated import *
from EKF import *
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

check_results[:, 50:] = 0
check_results[:, 125:] = 1
check_results[:, 150:] = 0

def thrusting_dynamics_equation(t, X, mu, umax, rho):

    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    jacobian = reforumlated_min_fuel_jacobian(state, costate, mu, umax, rho)

    ddt_state = reformulated_min_fuel_ODE(0, X[0:12], mu, umax, rho)
    ddt_STM = jacobian @ STM

    return np.concatenate((ddt_state, ddt_STM.flatten()))

def original_dynamics_equation(t, X, mu, umax, rho):

    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    jacobian = minimum_fuel_jacobian(state, costate, mu, umax, rho)

    ddt_state = minimum_fuel_ODE(0, X[0:12], mu, umax, rho)
    ddt_STM = jacobian @ STM

    return np.concatenate((ddt_state, ddt_STM.flatten()))

def filter_measurement_equation(time_index, X, mu, sensor_position_vals, individual_measurement_size):

    num_sensors = int(np.size(sensor_position_vals, 0)/3)
    measurement = np.empty(num_sensors*individual_measurement_size)
    measurement_jacobian = np.empty((num_sensors*individual_measurement_size, 12))

    for sensor_index in range(num_sensors):
        sensor_position = sensor_position_vals[sensor_index*3:(sensor_index+1)*3, time_index]
        
        measurement[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor(X, sensor_position)
        measurement_jacobian[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor_jacobian_costate(X, sensor_position)

    return measurement, measurement_jacobian

def big_function(seed):

    print("run number ", seed)

    initial_estimate = np.random.default_rng(seed).multivariate_normal(initial_truth, initial_kernel_covariance)
    # initial_estimate[9:12] = np.array([0, 0, 1e-3])
    original_initial_estimate = initial_estimate.copy()
    original_initial_estimate[6:12] = reformulated2standard(original_initial_estimate[6:12])

    measurements = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, measurement_noise_covariance, sensor_position_vals, check_results, seed)

    filter_output = run_EKF(initial_estimate, initial_kernel_covariance, 
                            thrusting_dynamics_equation, filter_measurement_equation, 
                            measurements, kernel_process_noise, measurement_noise_covariance,
                            dynamics_args, measurement_args)
    
    original_output = run_EKF(original_initial_estimate, original_initial_covariance,
                            original_dynamics_equation, filter_measurement_equation,
                            measurements, original_process_noise, measurement_noise_covariance,
                            dynamics_args, measurement_args)

    return [filter_output, original_output]

dynamics_args = (mu, umax, filter_rho)
measurement_args = (mu, sensor_position_vals, individual_measurement_size)

num_runs = 5
results = Parallel(n_jobs=6)(delayed(big_function)(seed) for seed in range(num_runs))

posterior_estimates = []
posterior_covariances = []
original_posterior_estimates = []
original_posterior_covariances = []

for result in results:
    posterior_estimates.append(result[0].posterior_estimate_vals.copy())
    posterior_covariances.append(result[0].posterior_covariance_vals.copy())
    original_posterior_estimates.append(result[1].posterior_estimate_vals.copy())
    original_posterior_covariances.append(result[1].posterior_covariance_vals.copy())

converted_posterior_estimates = []
converted_truth_vals = truth_vals.copy()
for val_index in range(len(time_vals)):
    converted_truth_vals[6:12, val_index] = reformulated2standard(converted_truth_vals[6:12, val_index])
for run_index in range(num_runs):
    converted_posterior_estimates.append(original_posterior_estimates[run_index].copy())
    for val_index in range(len(time_vals)):
        converted_posterior_estimates[run_index][6:12, val_index] = standard2reformulated(converted_posterior_estimates[run_index][6:12, val_index])

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2], alpha=0.75)
for run_index in range(num_runs):
    ax.plot(posterior_estimates[run_index][0], posterior_estimates[run_index][1], posterior_estimates[run_index][2], alpha=0.25, c="orange")
    ax.plot(original_posterior_estimates[run_index][0], original_posterior_estimates[run_index][1], original_posterior_estimates[run_index][2], alpha=0.25, c="green")
ax.set_aspect("equal")
plot_moon(ax, mu)

estimation_errors = compute_estimation_errors(truth_vals, posterior_estimates, 12)
three_sigmas = compute_3sigmas(posterior_covariances, 12)
original_estimation_errors = compute_estimation_errors(converted_truth_vals, original_posterior_estimates, 12)
original_three_sigmas = compute_3sigmas(original_posterior_covariances, 12)
plot_3sigma(time_vals, estimation_errors, three_sigmas, 6, bounds=(-0.5, 0.5))
plot_3sigma_costate(time_vals, estimation_errors, three_sigmas, 6, bounds=(-1, 1))
plot_3sigma(time_vals, original_estimation_errors, original_three_sigmas, 6, bounds=(-0.5, 0.5))
plot_3sigma_costate(time_vals, original_estimation_errors, original_three_sigmas, 6, bounds=(-1, 1))

truth_control = get_reformulated_min_fuel_control(truth_vals[6:12], umax, truth_rho)
estimated_controls = []
original_controls = []
for run_index in range(num_runs):
    estimated_controls.append(get_reformulated_min_fuel_control(posterior_estimates[run_index][6:12], umax, filter_rho))
    original_controls.append(get_min_fuel_control(original_posterior_estimates[run_index][6:12], umax, filter_rho))
fig = plt.figure()
for ax_index in range(3):
    thing = int("31" + str(ax_index + 1))
    ax = fig.add_subplot(thing)
    ax.step(time_vals, truth_control[ax_index], alpha=0.75)
    for run_index in range(num_runs):
        ax.step(time_vals, estimated_controls[run_index][ax_index], alpha=0.25, c="orange")
        ax.step(time_vals, original_controls[run_index][ax_index], alpha=0.25, c="green")

fig = plt.figure()
for ax_index in range(6):
    thing = int("23" + str(ax_index+1))
    ax = fig.add_subplot(thing)
    ax.plot(time_vals, truth_vals[6+ax_index], alpha=0.4)
    for run_index in range(num_runs):
        ax.plot(time_vals, posterior_estimates[run_index][6+ax_index], alpha=0.4)
        ax.plot(time_vals, converted_posterior_estimates[run_index][6+ax_index], alpha=0.4)

ax = plt.figure().add_subplot()
ax.step(time_vals, truth_vals[-1], alpha=0.75)
for run_index in range(num_runs):
    ax.step(time_vals, posterior_estimates[run_index][-1], alpha=0.25, c="orange")
    ax.step(time_vals, converted_posterior_estimates[run_index][-1], alpha=0.25, c="green")

plt.show()