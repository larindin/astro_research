

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from configuration_dual_GMIMM import *
from CR3BP import *
from CR3BP_pontryagin import *
from IMM import *
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

# check_results[:, 200:] = 0
# check_results[:, 300:] = 1

measurements = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, measurement_noise_covariance, sensor_position_vals, check_results, seed)

def coasting_dynamics_equation(t, X, mu, umax, rho):

    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    jacobian = CR3BP_costate_jacobian(state, costate, mu, umax)

    ddt_state = CR3BP_DEs(t, state, mu)
    ddt_costate = np.zeros(6)
    ddt_STM = jacobian @ STM

    return np.concatenate((ddt_state, ddt_costate, ddt_STM.flatten()))

def thrusting_dynamics_equation(t, X, mu, umax, rho):

    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    jacobian = CR3BP_costate_jacobian(state, costate, mu, umax)

    ddt_state = minimum_fuel_ODE(0, X[0:12], mu, umax, rho)
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


dynamics_args = (mu, umax, filter_rho)
measurement_args = (mu, sensor_position_vals, individual_measurement_size)
dynamics_equations = [coasting_dynamics_equation, thrusting_dynamics_equation]
num_modes = len(dynamics_equations)


IMM_output = run_IMM(initial_estimate, initial_covariance, initial_mode_probabilities,
                    dynamics_equations, filter_measurement_equation, measurements,
                    process_noise_covariances, filter_measurement_covariance,
                    dynamics_args, measurement_args, mode_transition_matrix)

IMM_t = IMM_output.t
IMM_posterior_estimate_vals = IMM_output.posterior_estimate_vals
IMM_posterior_covariance_vals = IMM_output.posterior_covariance_vals
IMM_weights = IMM_output.weight_vals

thrusting_indices = get_thrusting_indices(IMM_output, switching_cutoff)
start_index = thrusting_indices[0]
end_index = thrusting_indices[1]
duration = time_vals[end_index] - time_vals[start_index]

print(start_index)
print(end_index)
print(duration)

initial_state = IMM_posterior_estimate_vals[0:6, start_index, 1]
initial_covariance = IMM_posterior_covariance_vals[0:6, 0:6, start_index, 1]
initial_estimated_lv = IMM_posterior_estimate_vals[9:12, start_index, 1]
final_state = IMM_posterior_estimate_vals[0:6, end_index, 1]
final_covariance = IMM_posterior_covariance_vals[0:6, 0:6, end_index, 1]
final_estimated_lv = IMM_posterior_estimate_vals[9:12, end_index, 1]
initial_truth_state = truth_vals[0:6, start_index]
initial_truth_costate = truth_vals[6:12, start_index]
final_truth_state = truth_vals[0:6, end_index]
final_truth_costate = truth_vals[6:12, end_index]
initial_truth_lv = truth_vals[9:12, start_index]
final_truth_lv = truth_vals[9:12, end_index]

initial_costate_estimates = get_min_fuel_costates(initial_state, initial_estimated_lv, mu, umax, duration, magnitudes, "initial", "initial")

if False:
    
    measurements.t = measurements.t[start_index+1:]
    measurements.measurements = measurements.measurements[:, start_index+1:]
    new_time_vals = measurements.t
    tspan = np.array([time_vals[0], time_vals[-1]])
    propagations = []
    for kernel_index in np.arange(num_kernels):
        print(kernel_index)
        propagation_initial_conditions = np.concatenate((initial_state, initial_costate_estimates[:, kernel_index]))
        new_propagation = scipy.integrate.solve_ivp(dynamics_equation, tspan, propagation_initial_conditions, args=(mu, umax, truth_rho), t_eval=new_time_vals, atol=1e-12, rtol=1e-12)
        propagations.append(new_propagation)

    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
    for propagation in propagations:
        vals = propagation.y
        ax.plot(vals[0], vals[1], vals[2], alpha=0.25)
    ax.set_aspect("equal")

    truth_control = get_min_fuel_control(truth_vals[6:12, :], umax, truth_rho)
    propagation_controls = []
    for index in np.arange(num_kernels):
        propagation_control = get_min_fuel_control(propagations[index].y[6:12, :], umax, truth_rho)
        propagation_controls.append(propagation_control)
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.step(time_vals, truth_control[0])
    for index in np.arange(num_kernels):
        ax.step(new_time_vals, propagation_controls[index][0], alpha=0.25)
    ax = fig.add_subplot(312)
    ax.step(time_vals, truth_control[1])
    for index in np.arange(num_kernels):
        ax.step(new_time_vals, propagation_controls[index][1], alpha=0.25)
    ax = fig.add_subplot(313)
    ax.step(time_vals, truth_control[2])
    for index in np.arange(num_kernels):
        ax.step(new_time_vals, propagation_controls[index][2], alpha=0.25)

    fig = plt.figure()
    ax = fig.add_subplot(231)
    ax.plot(time_vals, truth_vals[6])
    for index in np.arange(num_kernels):
        ax.plot(new_time_vals, propagations[index].y[6], alpha=0.25)
    ax = fig.add_subplot(232)
    ax.plot(time_vals, truth_vals[7])
    for index in np.arange(num_kernels):
        ax.plot(new_time_vals, propagations[index].y[7], alpha=0.25)
    ax = fig.add_subplot(233)
    ax.plot(time_vals, truth_vals[8])
    for index in np.arange(num_kernels):
        ax.plot(new_time_vals, propagations[index].y[8], alpha=0.25)
    ax = fig.add_subplot(234)
    ax.plot(time_vals, truth_vals[9])
    for index in np.arange(num_kernels):
        ax.plot(new_time_vals, propagations[index].y[9], alpha=0.25)
    ax = fig.add_subplot(235)
    ax.plot(time_vals, truth_vals[10])
    for index in np.arange(num_kernels):
        ax.plot(new_time_vals, propagations[index].y[10], alpha=0.25)
    ax = fig.add_subplot(236)
    ax.plot(time_vals, truth_vals[11])
    for index in np.arange(num_kernels):
        ax.plot(new_time_vals, propagations[index].y[11], alpha=0.25)

    plt.show()

    quit()

initial_estimates = np.empty((12, num_kernels))
initial_covariances = np.empty((12, 12, num_kernels))
for kernel_index in np.arange(num_kernels):
    initial_estimates[:, kernel_index] = np.concatenate((initial_state, initial_costate_estimates[:, kernel_index]))
    initial_covariances[0:6, 0:6, kernel_index] = initial_covariance
    initial_covariances[6:12, 6:12, kernel_index] = initial_kernel_costate_covariance

# initial_estimates[:, 0] = truth_vals[:, end_index]
# initial_covariances[:, :, 0] = np.eye(12)*0.001**2


measurements.t = measurements.t[start_index:]
measurements.measurements = measurements.measurements[:, start_index:]

measurement_args = (mu, sensor_position_vals[:, start_index:], individual_measurement_size)

GM_output = run_GM_EKF(initial_estimates, initial_covariances, initial_weights,
                    thrusting_dynamics_equation, filter_measurement_equation, measurements,
                    kernel_process_noise, measurement_noise_covariance,
                    dynamics_args, measurement_args)

GM_time = GM_output.t
anterior_estimate_vals = GM_output.anterior_estimate_vals
posterior_estimate_vals = GM_output.posterior_estimate_vals
posterior_covariance_vals = GM_output.posterior_covariance_vals
weight_vals = GM_output.weight_vals


ax = plt.figure().add_subplot()
ax.step(IMM_t, IMM_weights[0], alpha=0.5)
ax.step(IMM_t, IMM_weights[1], alpha=0.5)

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
for mode_index in np.arange(num_kernels):
    ax.plot(posterior_estimate_vals[0, :, mode_index], posterior_estimate_vals[1, :, mode_index], posterior_estimate_vals[2, :, mode_index], alpha=0.25)
ax.set_aspect("equal")

ax = plt.figure().add_subplot()
ax.plot([0, final_time], [0.5, 0.5], alpha=0.5)
for mode_index in np.arange(num_kernels):
    ax.step(GM_time, weight_vals[mode_index, :], alpha=0.25)

fig = plt.figure()
ax = fig.add_subplot(231)
ax.plot(time_vals, truth_vals[6])
for index in np.arange(num_kernels):
    ax.step(GM_time, posterior_estimate_vals[6, :, index], alpha=0.25)
ax = fig.add_subplot(232)
ax.plot(time_vals, truth_vals[7])
for index in np.arange(num_kernels):
    ax.step(GM_time, posterior_estimate_vals[7, :, index], alpha=0.25)
ax = fig.add_subplot(233)
ax.plot(time_vals, truth_vals[8])
for index in np.arange(num_kernels):
    ax.step(GM_time, posterior_estimate_vals[8, :, index], alpha=0.25)
ax = fig.add_subplot(234)
ax.plot(time_vals, truth_vals[9])
for index in np.arange(num_kernels):
    ax.step(GM_time, posterior_estimate_vals[9, :, index], alpha=0.25)
ax = fig.add_subplot(235)
ax.plot(time_vals, truth_vals[10])
for index in np.arange(num_kernels):
    ax.step(GM_time, posterior_estimate_vals[10, :, index], alpha=0.25)
ax = fig.add_subplot(236)
ax.plot(time_vals, truth_vals[11])
for index in np.arange(num_kernels):
    ax.step(GM_time, posterior_estimate_vals[11, :, index], alpha=0.25)

truth_control = get_min_fuel_control(truth_vals[6:12, :], umax, truth_rho)
estimated_controls = []
for index in np.arange(num_kernels):
    estimated_control = get_min_fuel_control(posterior_estimate_vals[6:12, :, index], umax, filter_rho)
    estimated_controls.append(estimated_control)
fig = plt.figure()
ax = fig.add_subplot(311)
ax.step(time_vals, truth_control[0])
for index in np.arange(num_kernels):
    ax.step(GM_time, estimated_controls[index][0], alpha=0.25)
ax = fig.add_subplot(312)
ax.step(time_vals, truth_control[1])
for index in np.arange(num_kernels):
    ax.step(GM_time, estimated_controls[index][1], alpha=0.25)
ax = fig.add_subplot(313)
ax.step(time_vals, truth_control[2])
for index in np.arange(num_kernels):
    ax.step(GM_time, estimated_controls[index][2], alpha=0.25)

plt.show()