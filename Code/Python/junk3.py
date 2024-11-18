

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

check_results[:, 50:] = 0
check_results[:, 125:] = 1
check_results[:, 140:] = 0

measurements = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, measurement_noise_covariance, sensor_position_vals, check_results, seed)

def coasting_dynamics_equation(t, X, mu, umax, rho):

    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    jacobian = minimum_fuel_jacobian(state, costate, mu, umax, rho)

    ddt_state = CR3BP_DEs(t, state, mu)
    ddt_costate = np.zeros(6)
    ddt_STM = jacobian @ STM

    return np.concatenate((ddt_state, ddt_costate, ddt_STM.flatten()))

def min_energy_dynamics_equation(t, X, mu, umax, rho):
    
    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    jacobian = minimum_energy_jacobian(state, costate, mu, umax)

    ddt_state = minimum_fuel_ODE(0, X[0:12], mu, umax, rho)
    ddt_STM = jacobian @ STM

    return np.concatenate((ddt_state, ddt_STM.flatten()))

def thrusting_dynamics_equation(t, X, mu, umax, rho):

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

    for sensor_index in np.arange(num_sensors):
        sensor_position = sensor_position_vals[sensor_index*3:(sensor_index+1)*3, time_index]
        
        measurement[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor(X, sensor_position)
        measurement_jacobian[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor_jacobian_costate(X, sensor_position)

    return measurement, measurement_jacobian

def constant_thrust_ODE(t, X, mu, umax, direction):

    state = X[0:6]
    STM = np.reshape(X[6:36+6], (6, 6))

    B = np.vstack((np.zeros((3, 3)), np.eye(3)))

    control = direction * umax

    ddt_state_kepler = CR3BP_DEs(t, state, mu)
    ddt_state = ddt_state_kepler + B @ control

    ddt_STM = -CR3BP_jacobian(state, mu).T @ STM
    ddt_STM = ddt_STM.flatten()

    return np.concatenate((ddt_state, ddt_STM))


dynamics_args = (mu, umax, filter_rho)
measurement_args = (mu, sensor_position_vals, individual_measurement_size)
dynamics_equations = [coasting_dynamics_equation, min_energy_dynamics_equation]
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

direction = -IMM_posterior_estimate_vals[9:12, start_index, 1]/np.linalg.norm(IMM_posterior_estimate_vals[9:12, start_index, 1])
ICs = np.concatenate((IMM_posterior_estimate_vals[0:6, start_index, 1], np.eye(6).flatten()))

single_propagation = scipy.integrate.solve_ivp(constant_thrust_ODE, np.array([0, duration]), ICs, args=(mu, umax, direction), atol=1e-12, rtol=1e-12).y

multi_propagations = []

for val_index in np.arange(start_index, end_index+1):

    ICs = np.concatenate((IMM_posterior_estimate_vals[0:6, val_index, 1], np.eye(6).flatten()))
    multi_propagations.append(scipy.integrate.solve_ivp(constant_thrust_ODE, np.array([0, 0.01]), ICs, args=(mu, umax, direction), atol=1e-12, rtol=1e-12).y)

STM_vals = np.empty((6, 6, len(multi_propagations)))
for val_index in np.arange(len(multi_propagations)):
    STM_vals[:, :, val_index] = np.reshape(multi_propagations[val_index][6:42, -1], (6, 6))
total_STM = np.eye(6)
for val_index in np.arange(len(multi_propagations)):
    total_STM = STM_vals[:, :, val_index] @ total_STM

print(np.reshape(single_propagation[6:42, -1], (6, 6)))
print(total_STM)

ax = plt.figure().add_subplot(projection="3d")
ax.scatter(truth_vals[0], truth_vals[1], truth_vals[2], alpha=0.5)
ax.scatter(IMM_posterior_estimate_vals[0, :, 1], IMM_posterior_estimate_vals[1, :, 1], IMM_posterior_estimate_vals[2, :, 1], alpha=0.5)
ax.scatter(single_propagation[0], single_propagation[1], single_propagation[2], alpha=0.5)
for prop_index in np.arange(len(multi_propagations)):
    ax.scatter(multi_propagations[prop_index][0], multi_propagations[prop_index][1], multi_propagations[prop_index][2], c="y", alpha=0.5)

plt.show()