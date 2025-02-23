

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from joblib import Parallel, delayed
from configuration_costate import *
from CR3BP import *
from CR3BP_pontryagin import *
from EKF import *
from helper_functions import *
from measurement_functions import *
from plotting import *

time_vals = np.arange(0, final_time, dt)
tspan = np.array([time_vals[0], time_vals[-1]])
truth_propagation = scipy.integrate.solve_ivp(dynamics_equation, tspan, initial_truth, args=(mu, umax), t_eval=time_vals, atol=1e-12, rtol=1e-12)
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

check_results[:, :] = 1

# check_results[:, 50:125] = 0
# check_results[:, 150:] = 0

check_results[:, 100:300] = 0
check_results[:, 325:] = 0

dynamics_args = (mu, umax)
measurement_args = (mu, sensor_position_vals, individual_measurement_size)

def EKF_dynamics_equation(t, X, mu, umax, process_noise_covariance):

    state = X[0:6]
    costate = X[6:12]
    covariance = X[12:156].reshape((12, 12))

    jacobian = minimum_energy_jacobian(state, costate, mu, umax)

    ddt_state = minimum_energy_ODE(0, X[0:12], mu, umax)
    ddt_covariance = jacobian @ covariance + covariance @ jacobian.T + process_noise_covariance

    return np.concatenate((ddt_state, ddt_covariance.flatten()))
    
def EKF_measurement_equation(time_index, X, mu, sensor_position_vals, individual_measurement_size):

    num_sensors = int(np.size(sensor_position_vals, 0)/3)
    measurement = np.empty(num_sensors*individual_measurement_size)
    measurement_jacobian = np.empty((num_sensors*individual_measurement_size, 12))

    for sensor_index in range(num_sensors):
        sensor_position = sensor_position_vals[sensor_index*3:(sensor_index+1)*3, time_index]
        
        measurement[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor(X, sensor_position)
        measurement_jacobian[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor_jacobian_costate(X, sensor_position)

    return measurement, measurement_jacobian

def big_function(seed):

    initial_estimate = np.concatenate((np.random.default_rng(seed).multivariate_normal(initial_truth[0:6], initial_covariance[0:6, 0:6]), np.array([0, 0, 0, 0, 0, 0])))

    measurements = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, measurement_noise_covariance, sensor_position_vals, check_results, seed)

    filter_output = run_EKF(initial_estimate, initial_covariance,
                            EKF_dynamics_equation, EKF_measurement_equation,
                            measurements, process_noise_covariance,
                            filter_measurement_covariance, 
                            dynamics_args, measurement_args)

    return filter_output

results = Parallel(n_jobs=6)(delayed(big_function)(seed) for seed in range(50))

posterior_estimates = []
posterior_covariances = []

for result in results:
    posterior_estimates.append(result.posterior_estimate_vals)
    posterior_covariances.append(result.posterior_covariance_vals)

estimation_errors = compute_estimation_errors(truth_vals, posterior_estimates, 12)
three_sigmas = compute_3sigmas(posterior_covariances, 12)
divergence_results = check_divergence(estimation_errors, three_sigmas)

print(np.count_nonzero(divergence_results))

divergence_results[:] = 1

ax = plt.figure().add_subplot(projection="3d")
ax.scatter(1-mu, 0, 0, "gray")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2], alpha=0.5)
for run_index in range(len(divergence_results)):
    if divergence_results[run_index]:
        vals = posterior_estimates[run_index]
        ax.plot(vals[0], vals[1], vals[2], alpha=0.25)
ax.set_aspect("equal")

plot_3sigma(time_vals, estimation_errors, three_sigmas, 6, bounds=(-1, 1), alpha=0.25)
plot_3sigma_costate(time_vals, estimation_errors, three_sigmas, 6, bounds=(-5, 5), alpha=0.25)

plt.show()