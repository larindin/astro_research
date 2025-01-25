import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import time
from configuration_particle_filter import *
from CR3BP import *
from CR3BP_pontryagin_reformulated import *
from particle_filter import *
from helper_functions import *
from measurement_functions import *
from plotting import *

def min_fuel_costateSTM(t, X, mu, umax, rho):

    state = X[0:6]
    costate = X[6:12]
    STM = X[12:].reshape((6, 6))

    B = np.vstack((np.zeros((3, 3)), np.eye(3)))

    p = -B.T @ costate
    G = umax/2 * (1 + np.tanh((np.linalg.norm(p) - 1)/rho))
    control = G * p/np.linalg.norm(p)

    ddt_state_kepler = CR3BP_DEs(t, state, mu)
    ddt_state = ddt_state_kepler + B @ control

    ddt_costate = CR3BP_costate_DEs(t, state, costate, mu)

    ddt_STM = -CR3BP_jacobian(state, mu).T @ STM
    ddt_STM = ddt_STM.flatten()

    return np.concatenate((ddt_state, ddt_costate, ddt_STM))

def filter_measurement_equation(time_index, X, mu, sensor_position_vals, individual_measurement_size):

    num_sensors = int(np.size(sensor_position_vals, 0)/3)
    measurement = np.empty(num_sensors*individual_measurement_size)
    measurement_jacobian = np.empty((num_sensors*individual_measurement_size, 12))

    for sensor_index in np.arange(num_sensors):
        sensor_position = sensor_position_vals[sensor_index*3:(sensor_index+1)*3, time_index]
        
        measurement[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor(X, sensor_position)
        measurement_jacobian[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor_jacobian_costate(X, sensor_position)

    return measurement, measurement_jacobian

start_time = time.time()

time_vals = np.arange(0, final_time, dt)
tspan = np.array([time_vals[0], time_vals[-1]])
initial_truth = np.concatenate((initial_truth, np.eye(6).flatten()))
truth_propagation = scipy.integrate.solve_ivp(min_fuel_costateSTM, tspan, initial_truth, args=truth_dynamics_args, t_eval=time_vals, atol=1e-12, rtol=1e-12)
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

check_results[:, 100:] = 0
check_results[:, 125:] = 1
check_results[:, 250:] = 0

measurements = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, measurement_noise_covariance, sensor_position_vals, check_results, seed)

final_thrust_timestep = 16

total_STM = truth_vals[12:, final_thrust_timestep].reshape((6, 6))

initial_costate_original = truth_vals[6:12, 0].copy()
initial_costate = standard2reformulated(initial_costate_original)
final_costate_original = truth_vals[6:12, final_thrust_timestep].copy()
final_costate = standard2reformulated(final_costate_original)

costate_angle_covariance = np.eye(2)*np.deg2rad(1)**2
magnitude_error_std = 1
generator = np.random.default_rng(seed)
initial_costate_angle_errors = generator.multivariate_normal(np.zeros(2), costate_angle_covariance, num_particles)
final_costate_angle_errors = generator.multivariate_normal(np.zeros(2), costate_angle_covariance, num_particles)
initial_magnitude_errors = np.abs(generator.normal(0, magnitude_error_std, num_particles))
final_magnitude_errors = generator.normal(0, magnitude_error_std/5, num_particles)

initial_costate_angle_errors *= 0
final_costate_angle_errors *= 0
# initial_magnitude_errors *= 0
# final_magnitude_errors *= 0

STM_rr = total_STM[0:3, 0:3]
STM_rv = total_STM[0:3, 3:6]
STM_vr = total_STM[3:6, 0:3]
STM_vv = total_STM[3:6, 3:6]

costate_estimates = np.empty((6, num_particles*num_kernels))

total_index = 0
for particle_index in np.arange(num_particles):
    initial_lv = reformulated2standard(initial_costate + np.concatenate((np.zeros(3), initial_costate_angle_errors[particle_index], np.zeros(1))))[3:6]
    initial_lv_hat = initial_lv/np.linalg.norm(initial_lv)
    final_lv_error = np.concatenate((final_costate_angle_errors[particle_index], np.array([final_magnitude_errors[particle_index]])))
    final_lv = reformulated2standard(final_costate + np.concatenate((np.zeros(3), final_lv_error)))[3:6]
    
    for magnitude_index in np.arange(num_kernels):
        magnitude = magnitudes[magnitude_index] + initial_magnitude_errors[particle_index]
        initial_lv = initial_lv_hat * magnitude
        initial_lr = np.linalg.inv(STM_vr) @ (final_lv - STM_vv @ initial_lv)
        costate_estimates[:, total_index] = np.concatenate((initial_lr, initial_lv))
        total_index += 1

initial_estimates = np.empty((12, num_particles*num_kernels))
for particle_index in np.arange(num_particles*num_kernels):
    initial_estimates[:, particle_index] = np.concatenate((initial_truth[0:6], costate_estimates[:, particle_index]))

initial_weights = np.ones(num_particles*num_kernels)/(num_particles*num_kernels)

dynamics_args = (mu, umax, truth_rho)
measurement_args = (mu, sensor_position_vals, individual_measurement_size)
resampling_args = (roughening_cov,)

particle_filter_results = run_particle_filter(initial_estimates, initial_weights, minimum_fuel_ODE,
                                              filter_measurement_equation, measurements, filter_measurement_covariance,
                                              dynamics_args, measurement_args, resampling_args)

estimate_vals = particle_filter_results.estimate_vals
weight_vals = particle_filter_results.weight_vals
covariance_vals = np.zeros((12, 12, len(time_vals), num_kernels*num_particles))

estimate_vals, blanks = trim_zero_weights(estimate_vals, covariance_vals, weight_vals)

N_eff_vals = calculate_N_eff_vals(weight_vals)

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2], alpha=0.75)
for run_index in np.arange(num_particles*num_kernels):
    propagation = estimate_vals[:, :, run_index]
    ax.scatter(propagation[0], propagation[1], propagation[2], alpha=0.35)
ax.set_xlabel("X [LU]")
ax.set_ylabel("Y [LU]")
ax.set_zlabel("Z [LU]")
plot_moon(ax, mu)
ax.set_aspect("equal")

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[6], truth_vals[7], truth_vals[8], alpha=0.75)
for run_index in np.arange(num_particles*num_kernels):
    propagation = estimate_vals[:, :, run_index]
    ax.scatter(propagation[6], propagation[7], propagation[8], alpha=0.35)
ax.set_xlabel("l1")
ax.set_ylabel("l2")
ax.set_zlabel("l3")
ax.set_aspect("equal")

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[9], truth_vals[10], truth_vals[11], alpha=0.75)
for run_index in np.arange(num_particles*num_kernels):
    propagation = estimate_vals[:, :, run_index]
    ax.scatter(propagation[9], propagation[10], propagation[11], alpha=0.35)
ax.set_xlabel("l4")
ax.set_ylabel("l5")
ax.set_zlabel("l6")
ax.set_aspect("equal")

ax = plt.figure().add_subplot()
ax.plot(time_vals, N_eff_vals)

# truth_control = get_min_fuel_control(truth_vals[6:12, :], umax, truth_rho)
# estimated_controls = []
# for index in np.arange(num_particles*num_kernels):
#     estimated_control = get_min_fuel_control(estimate_vals[6:12, :, index], umax, filter_rho)
#     estimated_controls.append(estimated_control)
# fig = plt.figure()
# for ax_index in np.arange(3):
#     thing = int("31" + str(ax_index+1))
#     ax = fig.add_subplot(thing)
#     ax.plot(time_vals, truth_control[ax_index], alpha=0.75)
#     for index in np.arange(num_particles*num_kernels):
#         ax.plot(time_vals, estimated_controls[index][ax_index], alpha=0.35)

ax = plt.figure().add_subplot()
for index in np.arange(num_kernels*num_particles):
    ax.scatter(time_vals, weight_vals[index], alpha=0.2)
ax.set_ylim(-0.03, 1.03)

# fig = plt.figure()
# for ax_index in np.arange(6):
#     thing = int("61" + str(ax_index+1))
#     ax = fig.add_subplot(thing)
#     ax.plot(time_vals, truth_vals[ax_index+6])

end_time = time.time()
print(end_time - start_time)

plt.show()