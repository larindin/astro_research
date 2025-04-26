

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

time_vals = np.arange(0, final_time, dt)
truth_vals = forward_propagation

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

# sensor_initial_conditions = sensor_initial_conditions[0:2, :]
sensor_position_vals = generate_sensor_positions(sensor_dynamics_equation, sensor_initial_conditions, (mu,), time_vals)
# sensor_position_vals = sensor_position_vals[0:3, :]
# sensor_position_vals = np.zeros((6, len(time_vals)))
# sensor_position_vals[0] = L2
# sensor_position_vals[3] = L1

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

if gap == True:
    check_results[:, 15*24:20*24] = 0

# check_results[:, 215:] = 0
# check_results[:, 300:] = 1
# check_results[:, 350:] = 0
# check_results[:, 450:] = 1

def coasting_costate_dynamics_equation(t, X, mu, umax):

    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    ddt_state = CR3BP_DEs(t, state, mu)
    # ddt_costate = CR3BP_costate_DEs(0, state, costate, mu)
    # jacobian = minimum_energy_jacobian(state, costate, mu, umax)
    K = np.diag(np.full(6, 1e2))
    jacobian = coasting_costate_jacobian(state, mu, K)
    ddt_costate = -K @ costate
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

STM_ICs = np.concatenate((initial_truth, np.eye(12).flatten()))
STM_propagation = scipy.integrate.solve_ivp(min_time_dynamics_equation, forprop_tspan, STM_ICs, args=(mu, umax), t_eval=forprop_time_vals, atol=1e-12, rtol=1e-12).y

num_measurements = len(time_vals)
num_measurements = 4
noises = []
STMs = np.empty((12, 12, num_measurements))
products = []
for time_index in range(num_measurements):
    measurement, measurement_jacobian, measurement_noise_covariance, rs = PV_measurement_equation(time_index,
                                                                                                  STM_propagation[0:12, time_index],
                                                                                                  measurement_variances,
                                                                                                  sensor_position_vals,
                                                                                                  check_results)
    
    print(np.shape(measurement_jacobian))
    
    noises.append(np.linalg.inv(measurement_noise_covariance))
    STMs[:, :, time_index] = STM_propagation[12:156, time_index].reshape((12, 12))
    products.append(measurement_jacobian @ STMs[:, :, time_index])
    
products = tuple(products)
noises = tuple(noises)
weight_matrix = scipy.linalg.block_diag(*noises)
observability_matrix = np.vstack(products)

information_matrix = observability_matrix.T @ weight_matrix @ observability_matrix

print(f"CN = {np.max(np.diag(information_matrix))/np.min(np.diag(information_matrix)):e}")
print(f"OI = {np.min(np.diag(information_matrix)):e}")