

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from joblib import Parallel, delayed
from configuration_initial_filter_algorithm import *
from CR3BP import *
from CR3BP_pontryagin import *
from CR3BP_pontryagin_reformulated import *
from EKF import *
from IMM import *
from minimization import *
from helper_functions import *
from measurement_functions import *
from plotting import *

backprop_time_vals = -np.arange(0, backprop_time, dt)
forprop_time_vals = np.arange(0, final_time, dt)
backprop_tspan = np.array([backprop_time_vals[0], backprop_time_vals[-1]])
forprop_tspan = np.array([forprop_time_vals[0], forprop_time_vals[-1]])
back_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, backprop_tspan, initial_truth[0:6], args=(mu,), t_eval=backprop_time_vals, atol=1e-12, rtol=1e-12).y
back_propagation = np.vstack((back_propagation, np.zeros(np.shape(back_propagation))))
back_propagation = np.flip(back_propagation, axis=1)
forward_propagation = scipy.integrate.solve_ivp(dynamics_equation, forprop_tspan, initial_truth, args=truth_dynamics_args, t_eval=forprop_time_vals, atol=1e-12, rtol=1e-12).y
truth_vals = np.concatenate((back_propagation, forward_propagation), axis=1)
time_vals = np.arange(0, final_time+backprop_time, dt)


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
# check_results[:, 125:] = 1
# check_results[:, 150:] = 0


def coasting_dynamics_equation(t, X, mu, umax):

    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    jacobian = minimum_energy_jacobian(state, costate, mu, umax)
    jacobian[0:6, 6:12] = 0

    ddt_state = CR3BP_DEs(t, state, mu)
    # ddt_costate = CR3BP_costate_DEs(0, state, costate, mu)
    K = np.diag(np.ones(6)*1e1)
    ddt_costate = -K @ costate
    ddt_STM = jacobian @ STM

    return np.concatenate((ddt_state, ddt_costate, ddt_STM.flatten()))

def maneuvering_dynamics_equation(t, X, mu, umax):
    
    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    jacobian = minimum_energy_jacobian(state, costate, mu, umax)

    ddt_state = minimum_energy_ODE(0, X[0:12], mu, umax)
    ddt_STM = jacobian @ STM

    return np.concatenate((ddt_state, ddt_STM.flatten()))

def filter_measurement_equation(time_index, X, mu, sensor_position_vals, individual_measurement_size):

    num_sensors = int(np.size(sensor_position_vals, 0)/3)
    measurement = np.empty(num_sensors*individual_measurement_size)
    measurement_jacobian = np.empty((num_sensors*individual_measurement_size, 12))

    for sensor_index in np.arange(num_sensors):
        sensor_position = sensor_position_vals[sensor_index*3:(sensor_index+1)*3, time_index]
        
        measurement[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor(X, sensor_position)
        measurement_jacobian[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor_jacobian_costate(X, sensor_position)

    return measurement, measurement_jacobian


dynamics_args = (mu, umax)
measurement_args = (mu, sensor_position_vals, individual_measurement_size)
dynamics_equations = [coasting_dynamics_equation, maneuvering_dynamics_equation]
num_modes = len(dynamics_equations)

def big_function(seed):

    print(seed)

    initial_estimate = np.concatenate((generator.multivariate_normal(truth_vals[0:6, 0], initial_state_covariance), np.zeros(6)))

    measurements = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, measurement_noise_covariance, sensor_position_vals, check_results, seed)

    filter_output = run_IMM(initial_estimate, initial_covariance, initial_mode_probabilities,
                            dynamics_equations, filter_measurement_equation, measurements,
                            process_noise_covariances, filter_measurement_covariance,
                            dynamics_args, measurement_args, mode_transition_matrix)

    return filter_output

num_runs = 50
results = Parallel(n_jobs=8)(delayed(big_function)(seed) for seed in range(num_runs))

posterior_estimates = []
posterior_covariances = []
posterior_controls = []
posterior_primer_vectors = []

for result in results:
    new_estimate_vals, new_covariance_vals = compute_IMM_output(result.posterior_estimate_vals, result.posterior_covariance_vals, result.weight_vals)
    posterior_estimates.append(new_estimate_vals)
    posterior_covariances.append(new_covariance_vals)
    posterior_controls.append(get_min_energy_control(new_estimate_vals[6:12, :], umax))
    posterior_primer_vectors.append(compute_primer_vectors(new_estimate_vals[9:12, :]))

truth_control = get_min_fuel_control(truth_vals[6:12, :], umax, truth_rho)
truth_primer_vectors = compute_primer_vectors(truth_vals[9:12])

estimation_errors = compute_estimation_errors(truth_vals, posterior_estimates, 12)
three_sigmas = compute_3sigmas(posterior_covariances, 12)

true_thrusting_bool = np.linalg.norm(truth_control, axis=0)
# thrusting_bool = mode_probabilities[1] > 0.5

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
for run_index in np.arange(num_runs):
    ax.plot(posterior_estimates[run_index][0], posterior_estimates[run_index][1], posterior_estimates[run_index][2], alpha=0.25)
plot_moon(ax, mu)
ax.set_aspect("equal")

plot_3sigma(time_vals, estimation_errors, three_sigmas, 6, alpha=0.25)
plot_3sigma_costate(time_vals, estimation_errors, three_sigmas, 6, alpha=0.25)

# ax = plt.figure().add_subplot()
# ax.plot(time_vals, mode_probabilities[0])
# ax.plot(time_vals, mode_probabilities[1])
# ax.plot(time_vals, true_thrusting_bool, alpha=0.5)
# ax.plot(time_vals, thrusting_bool*umax, alpha=0.5)

control_fig = plt.figure()
for ax_index in np.arange(3):
    thing = int("31" + str(ax_index + 1))
    ax = control_fig.add_subplot(thing)
    ax.plot(time_vals, truth_control[ax_index], alpha=0.75)
    ax.plot(time_vals, true_thrusting_bool, alpha=0.5)
    for run_index in np.arange(num_runs):
        ax.plot(time_vals, posterior_controls[run_index][ax_index], alpha=0.25)

primer_vector_fig = plt.figure()
for ax_index in np.arange(3):
    thing = int("31" + str(ax_index + 1))
    ax = primer_vector_fig.add_subplot(thing)
    ax.plot(time_vals, truth_primer_vectors[ax_index], alpha=0.75)
    ax.plot(time_vals, true_thrusting_bool, alpha=0.5)
    for run_index in np.arange(num_runs):
        ax.plot(time_vals, posterior_primer_vectors[run_index][ax_index], alpha=0.25)
    
plt.show()