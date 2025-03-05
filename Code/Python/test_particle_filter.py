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

def filter_measurement_equation(time_index, X, mu, sensor_position_vals, individual_measurement_size):

    num_sensors = int(np.size(sensor_position_vals, 0)/3)
    measurement = np.empty(num_sensors*individual_measurement_size)
    measurement_jacobian = np.empty((num_sensors*individual_measurement_size, 12))

    for sensor_index in range(num_sensors):
        sensor_position = sensor_position_vals[sensor_index*3:(sensor_index+1)*3, time_index]
        
        measurement[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor(X, sensor_position)
        measurement_jacobian[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor_jacobian_costate(X, sensor_position)

    return measurement, measurement_jacobian

start_time = time.time()

time_vals = np.arange(0, final_time, dt)
tspan = np.array([time_vals[0], time_vals[-1]])
# initial_truth = np.concatenate((initial_truth, np.eye(6).flatten()))
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

check_results[:, 100:] = 0
check_results[:, 200:] = 1
# check_results[:, 250:] = 0

measurements = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, measurement_noise_covariance, sensor_position_vals, check_results, seed)


# initial_estimates = np.empty((12, num_particles))
# for particle_index in range(num_particles*num_kernels):
#     initial_estimates[:, particle_index] = np.concatenate((initial_truth[0:6], costate_estimates[:, particle_index]))

initial_estimates = np.empty((12, num_particles))
for particle_index in range(num_particles):
    new_costate = generator.multivariate_normal(initial_truth[6:12], np.eye(6)*5e-3**2)
    initial_estimates[:, particle_index] = np.concatenate((initial_truth[0:6], new_costate))

initial_weights = np.ones(num_particles*num_kernels)/(num_particles*num_kernels)

dynamics_args = (mu, umax, truth_rho)
measurement_args = (mu, sensor_position_vals, individual_measurement_size)

particle_filter_results = run_particle_filter(initial_estimates, initial_weights, minimum_fuel_ODE, 
                                              filter_measurement_equation, measurements, filter_measurement_covariance, 
                                              dynamics_args, measurement_args, roughening_cov, seed)

estimate_vals = particle_filter_results.estimate_vals
anterior_weight_vals = particle_filter_results.anterior_weight_vals
weight_vals = particle_filter_results.weight_vals

# estimate_vals, blanks = trim_zero_weights(estimate_vals, covariance_vals, weight_vals)

N_eff_vals = calculate_N_eff_vals(weight_vals)

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2], alpha=0.75)
for run_index in range(num_particles*num_kernels):
    propagation = estimate_vals[:, :, run_index]
    ax.scatter(propagation[0], propagation[1], propagation[2], alpha=0.25, s=2)
ax.set_xlabel("X [LU]")
ax.set_ylabel("Y [LU]")
ax.set_zlabel("Z [LU]")
plot_moon(ax, mu)
ax.set_aspect("equal")

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[3], truth_vals[4], truth_vals[5], alpha=0.75)
for run_index in range(num_particles*num_kernels):
    propagation = estimate_vals[:, :, run_index]
    ax.scatter(propagation[3], propagation[4], propagation[5], alpha=0.25, s=2)
ax.set_xlabel("VX [LU]")
ax.set_ylabel("VY [LU]")
ax.set_zlabel("VZ [LU]")
ax.set_aspect("equal")

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[6], truth_vals[7], truth_vals[8], alpha=0.75)
for run_index in range(num_particles*num_kernels):
    propagation = estimate_vals[:, :, run_index]
    ax.scatter(propagation[6], propagation[7], propagation[8], alpha=0.25, s=2)
ax.set_xlabel("l1")
ax.set_ylabel("l2")
ax.set_zlabel("l3")
ax.set_aspect("equal")

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[9], truth_vals[10], truth_vals[11], alpha=0.75)
for run_index in range(num_particles*num_kernels):
    propagation = estimate_vals[:, :, run_index]
    ax.scatter(propagation[9], propagation[10], propagation[11], alpha=0.25, s=2)
ax.set_xlabel("l4")
ax.set_ylabel("l5")
ax.set_zlabel("l6")
ax.set_aspect("equal")

ax = plt.figure().add_subplot()
ax.plot(time_vals, N_eff_vals)

# truth_control = get_min_fuel_control(truth_vals[6:12, :], umax, truth_rho)
# estimated_controls = []
# for index in range(num_particles*num_kernels):
#     estimated_control = get_min_fuel_control(estimate_vals[6:12, :, index], umax, filter_rho)
#     estimated_controls.append(estimated_control)
# fig = plt.figure()
# for ax_index in range(3):
#     thing = int("31" + str(ax_index+1))
#     ax = fig.add_subplot(thing)
#     ax.plot(time_vals, truth_control[ax_index], alpha=0.75)
#     for index in range(num_particles*num_kernels):
#         ax.plot(time_vals, estimated_controls[index][ax_index], alpha=0.35)

ax = plt.figure().add_subplot()
for index in range(num_kernels*num_particles):
    ax.scatter(time_vals, anterior_weight_vals[index], alpha=0.3, s=4)
ax.set_ylim(-0.03, 1.03)

ax = plt.figure().add_subplot()
for index in range(num_kernels*num_particles):
    ax.scatter(time_vals, weight_vals[index], alpha=0.3, s=4)
ax.set_ylim(-0.03, 1.03)

# fig = plt.figure()
# for ax_index in range(6):
#     thing = int("61" + str(ax_index+1))
#     ax = fig.add_subplot(thing)
#     ax.plot(time_vals, truth_vals[ax_index+6])

end_time = time.time()
print(end_time - start_time)

plt.show()