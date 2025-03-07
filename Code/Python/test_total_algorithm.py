

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from joblib import Parallel, delayed
from configuration_total_algorithm import *
from CR3BP import *
from CR3BP_pontryagin import *
from CR3BP_pontryagin_reformulated import *
from IMM import *
from minimization import *
from particle_filter import *
from helper_functions import *
from measurement_functions import *
from plotting import *

backprop_time_vals = -np.arange(0, backprop_time, dt)
forprop_time_vals = np.arange(0, final_time, dt)
backprop_tspan = np.array([backprop_time_vals[0], backprop_time_vals[-1]])
forprop_tspan = np.array([forprop_time_vals[0], forprop_time_vals[-1]])
back_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, backprop_tspan, initial_truth[0:6], args=(mu,), t_eval=backprop_time_vals, atol=1e-12, rtol=1e-12).y
back_propagation = np.vstack((back_propagation, np.full(np.shape(back_propagation), np.nan)))
back_propagation = np.flip(back_propagation, axis=1)
forward_propagation = scipy.integrate.solve_ivp(dynamics_equation, forprop_tspan, initial_truth, args=truth_dynamics_args, t_eval=forprop_time_vals, atol=1e-12, rtol=1e-12).y
truth_vals = np.concatenate((back_propagation[:, :-1], forward_propagation), axis=1)
time_vals = np.concatenate((np.flip(backprop_time_vals[1:]), forprop_time_vals)) + abs(backprop_time_vals[-1])

converted_truth_vals = np.empty((np.shape(truth_vals)))
for time_index in range(len(time_vals)):
    converted_truth_vals[:, time_index] = np.concatenate((truth_vals[0:6, time_index], standard2reformulated(truth_vals[6:12, time_index])))

initial_estimate = np.concatenate((generator.multivariate_normal(truth_vals[0:6, 0], initial_state_covariance), np.ones(6)*1e-6))

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
# check_results[:, 125:] = 1
# check_results[:, 150:] = 0

IMM_measurements = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, 
                                                measurement_noise_covariance, sensor_position_vals, check_results, generator)

def coasting_dynamics_equation(t, X, mu, umax):

    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    jacobian = minimum_energy_jacobian(state, costate, mu, umax)

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

    for sensor_index in range(num_sensors):
        sensor_position = sensor_position_vals[sensor_index*3:(sensor_index+1)*3, time_index]
        
        measurement[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor(X, sensor_position)
        measurement_jacobian[sensor_index*individual_measurement_size:(sensor_index+1)*individual_measurement_size] = az_el_sensor_jacobian_costate(X, sensor_position)

    return measurement, measurement_jacobian


IMM_dynamics_args = (mu, umax)
IMM_measurement_args = (mu, sensor_position_vals, individual_measurement_size)
IMM_dynamics_equations = [coasting_dynamics_equation, maneuvering_dynamics_equation]
num_modes = len(IMM_dynamics_equations)


IMM_output = run_IMM(initial_estimate, initial_covariance, initial_mode_probabilities,
                    IMM_dynamics_equations, filter_measurement_equation, IMM_measurements,
                    process_noise_covariances, IMM_measurement_covariance,
                    IMM_dynamics_args, IMM_measurement_args, mode_transition_matrix)

IMM_time_vals = IMM_output.t
IMM_estimate_vals = IMM_output.posterior_estimate_vals
IMM_covariance_vals = IMM_output.posterior_covariance_vals
IMM_mode_probability_vals = IMM_output.weight_vals

IMM_output_estimate_vals, IMM_output_covariance_vals = compute_IMM_output(IMM_estimate_vals, IMM_covariance_vals, IMM_mode_probability_vals)

def get_costate_STM(time_vals, state_vals, mu):

    x_spline = scipy.interpolate.CubicSpline(time_vals, state_vals[0])
    y_spline = scipy.interpolate.CubicSpline(time_vals, state_vals[1])
    z_spline = scipy.interpolate.CubicSpline(time_vals, state_vals[2])

    def STM_integrand(t, X, x_spline, y_spline, z_spline, mu):

        STM = X.reshape((6, 6))

        x, y, z = x_spline(t), y_spline(t), z_spline(t)

        state = np.array([x, y, z, 0, 0, 0])

        ddt_STM = -CR3BP_jacobian(state, mu).T @ STM

        return ddt_STM.flatten()

    STM_IC = np.eye(6).flatten()
    tspan = np.array([time_vals[-1], time_vals[0]])
    args = (x_spline, y_spline, z_spline, mu)

    STM = scipy.integrate.solve_ivp(STM_integrand, tspan, STM_IC, args=args, rtol=1e-12, atol=1e-12).y[:, -1].reshape((6, 6))

    return STM

thrusting_indices = get_thrusting_indices(IMM_time_vals, IMM_mode_probability_vals, thrusting_duration_cutoff)
thrusting_indices[0] = thrusting_indices[1] - thrusting_cutoff_offset
start_index = thrusting_indices[0]
end_index = thrusting_indices[1]
observation_end_index = end_index + additional_measurements

print(start_index)
print(end_index)
print(observation_end_index)

observation_times = IMM_time_vals[start_index:observation_end_index].copy()
observation_times -= observation_times[0]
observer_positions = sensor_position_vals[:, start_index:observation_end_index].copy()
observation_estimate_vals = IMM_output_estimate_vals[:, start_index:observation_end_index].copy()
observation_covariance_vals = IMM_output_covariance_vals[:, :, start_index:observation_end_index].copy()
observation_measurements = IMM_measurements.measurements[:, start_index:observation_end_index].copy()
observation_check_results = check_results[:, start_index:observation_end_index].copy()
observation_truth_vals = truth_vals[:, start_index:observation_end_index].copy()

observation_initial_truth_converted = np.concatenate((observation_truth_vals[0:6, 0], standard2reformulated(observation_truth_vals[6:12, 0])))
print(observation_initial_truth_converted[-1])

truth_residuals = measurement_lstsqr_reformulated_mag(observation_initial_truth_converted, observation_times, observation_measurements, 
                                                      measurement_noise_covariance, observer_positions, observation_check_results, truth_dynamics_args)
print(np.sum(truth_residuals**2))

initial_state_guess = observation_estimate_vals[0:6, 0].copy()
initial_lambdav_hat_guess = observation_estimate_vals[9:12, 0].copy()
initial_lambdav_hat_guess /= np.linalg.norm(initial_lambdav_hat_guess)
initial_theta, initial_psi = standard2reformulated(np.concatenate((np.zeros(3), initial_lambdav_hat_guess)))[3:5]
# initial_theta, initial_psi = truth_converted[9:11]
final_lambdav_guess = observation_estimate_vals[9:12, -1].copy()
final_lambdav_guess /= np.linalg.norm(final_lambdav_guess)

initial_STM = get_costate_STM(observation_times[0:thrusting_cutoff_offset], observation_estimate_vals[:, 0:thrusting_cutoff_offset], mu)

STM_vr = initial_STM[3:6, 0:3]
STM_vv = initial_STM[3:6, 3:6]

num_guesses = len(initial_guess_magnitudes)
solutions = np.empty((12, num_guesses))
solution_jacobians = []
solution_costs = np.empty(num_guesses)
cost_func_args = (observation_times, observation_measurements, measurement_noise_covariance, observer_positions, observation_check_results, truth_dynamics_args)
# for guess_index in range(num_guesses):

#     print(guess_index)

#     initial_lambdav_guess = initial_lambdav_hat_guess * initial_guess_magnitudes[guess_index]

#     

#     initial_lambdar_guess = np.linalg.inv(STM_vr) @ (final_lambdav_guess - STM_vv @ initial_lambdav_guess)
#     initial_guess = np.concatenate((initial_state_guess, initial_lambdar_guess, np.array([initial_theta, initial_psi, initial_guess_magnitudes[guess_index]])))

#     solution = scipy.optimize.least_squares(measurement_lstsqr_reformulated_mag, initial_guess, args=cost_func_args, method="lm", verbose=2)

#     solutions[:, guess_index] = solution.x
#     solution_jacobians.append(solution.jac)
#     solution_costs[guess_index] = 2*solution.cost
#     print(solution.x)
#     print(solution.success)
#     print(solution.message)
#     print(np.sum(measurement_lstsqr_reformulated_mag(solution.x, *cost_func_args)**2))

inputs = np.load("sol_jac.npz")
solutions[:, 0] = inputs["solution"]
solution_jacobians.append(inputs["jac"])
solution_costs[0] = 10

def min_fuel_STM_ode(t, X, mu, umax, rho):

    state = X[0:6]
    costate = X[6:12]
    STM = np.reshape(X[12:36+12], (6, 6))

    ddt_state = reformulated_min_fuel_ODE(0, X[0:12], mu, umax, rho)

    ddt_STM = -CR3BP_jacobian(state, mu).T @ STM
    ddt_STM = ddt_STM.flatten()

    return np.concatenate((ddt_state, ddt_STM))

def magnitude_event(t, X, mu, umax, rho):
    return X[11] - 1 

magnitude_event.terminal = True

best_solution_index = solution_costs.argmin()
best_solution = solutions[:, best_solution_index]
best_jacobian = solution_jacobians[best_solution_index]

solution_tspan = np.array([0, observation_times[-1]])
solution_STM_ICs = np.concatenate((best_solution, np.eye(6).flatten()))
# solution_STM_ICs = np.concatenate((converted_truth_vals[:, start_index], np.eye(6).flatten()))
solution_STM_propagation = scipy.integrate.solve_ivp(min_fuel_STM_ode, solution_tspan, solution_STM_ICs, events=magnitude_event, args=truth_dynamics_args, atol=1e-12, rtol=1e-12)

solution_STM_vals = solution_STM_propagation.y

solution_STM = solution_STM_vals[12:36+12, -1].reshape((6, 6))
inv_solution_STM_vr = np.linalg.inv(solution_STM[3:6, 0:3])
solution_STM_vv = solution_STM[3:6, 3:6]
initial_angles_solution = solution_STM_vals[9:11, 0]
final_angles_solution = solution_STM_vals[9:11, -1]

sampling_covariance = np.linalg.inv(best_jacobian.T @ best_jacobian)

print(observation_initial_truth_converted)
print(best_solution)
print(np.sqrt(np.diag(sampling_covariance[9:11, 9:11])))

def get_costs(value):
    return np.sum(measurement_lstsqr_reformulated_mag(value, *cost_func_args)**2)

particle_ICs = np.empty((12, num_particles))
state_perturbations = generator.multivariate_normal(np.zeros(6), sampling_covariance[0:6, 0:6], num_particles).T
first_angle_perturbations = generator.multivariate_normal(np.zeros(2), sampling_covariance[9:11, 9:11], num_particles).T
second_angle_perturbations = generator.multivariate_normal(np.zeros(2), sampling_covariance[9:11, 9:11], num_particles).T
first_magnitudes = generator.uniform(*first_magnitude_sampling_range, num_particles)
second_magnitudes = generator.uniform(*second_magnitude_sampling_range, num_particles)
for particle_index in range(num_particles):
    particle_state = best_solution[0:6] + state_perturbations[:, particle_index]
    initial_particle_angles = initial_angles_solution + first_angle_perturbations[:, particle_index]
    final_particle_angles = final_angles_solution + second_angle_perturbations[:, particle_index]

    particle_initial_lambdav = reformulated2standard(np.concatenate((np.zeros(3), initial_particle_angles, [first_magnitudes[particle_index]])))[3:6]
    particle_final_lambdav = reformulated2standard(np.concatenate((np.zeros(3), final_particle_angles, [second_magnitudes[particle_index]])))[3:6]
    particle_lambdar = inv_solution_STM_vr @ (particle_final_lambdav - solution_STM_vv @ particle_initial_lambdav)

    particle_ICs[:, particle_index] = np.concatenate((particle_state, particle_lambdar, initial_particle_angles, [first_magnitudes[particle_index]]))

# for particle_index in range(num_particles):
#     particle_ICs[:, particle_index] = converted_truth_vals[:, start_index]

# chi2_cutoff = get_chi2_cutoff(6*(thrusting_cutoff_offset+additional_measurements)-12, 0.003)
# chi2_cutoff = 150
# # print(chi2_cutoff)
# remaining_particles = num_particles
# particle_ICs = np.empty((12, num_particles))
# while remaining_particles > 0:
    
#     print(remaining_particles)

#     ICs = np.empty((12, remaining_particles))

#     state_perturbations = generator.multivariate_normal(np.zeros(6), sampling_covariance[0:6, 0:6], remaining_particles).T
#     first_angle_perturbations = generator.multivariate_normal(np.zeros(2), sampling_covariance[9:11, 9:11], remaining_particles).T
#     second_angle_perturbations = generator.multivariate_normal(np.zeros(2), sampling_covariance[9:11, 9:11], remaining_particles).T
#     first_magnitudes = generator.uniform(*first_magnitude_sampling_range, remaining_particles)
#     second_magnitudes = generator.uniform(*second_magnitude_sampling_range, remaining_particles)
    
#     for particle_index in range(remaining_particles):
#         particle_state = best_solution[0:6] + state_perturbations[:, particle_index]
#         initial_particle_angles = initial_angles_solution + first_angle_perturbations[:, particle_index]
#         final_particle_angles = final_angles_solution + second_angle_perturbations[:, particle_index]

#         particle_initial_lambdav = reformulated2standard(np.concatenate((np.zeros(3), initial_particle_angles, [first_magnitudes[particle_index]])))[3:6]
#         particle_final_lambdav = reformulated2standard(np.concatenate((np.zeros(3), final_particle_angles, [second_magnitudes[particle_index]])))[3:6]
#         particle_lambdar = inv_solution_STM_vr @ (particle_final_lambdav - solution_STM_vv @ particle_initial_lambdav)

#         ICs[:, particle_index] = np.concatenate((particle_state, particle_lambdar, initial_particle_angles, [first_magnitudes[particle_index]]))

#     particle_costs = Parallel(n_jobs=8)(delayed(get_costs)(ICs[:, particle_index]) for particle_index in range(remaining_particles))

#     for particle_index in range(remaining_particles):
#         if particle_costs[particle_index] < chi2_cutoff:
#             particle_ICs[:, remaining_particles-1] = ICs[:, particle_index]
#             remaining_particles -= 1


PF_measurements = Measurements(IMM_measurements.t[start_index:], IMM_measurements.measurements[:, start_index:], IMM_measurements.individual_measurement_size)
PF_dynamics_args = (mu, umax, truth_rho)
PF_measurement_args = (mu, sensor_position_vals[:, start_index:], individual_measurement_size)

PF_results = run_particle_filter(particle_ICs, initial_weights, reformulated_min_fuel_ODE, 
                                 filter_measurement_equation, PF_measurements, PF_measurement_covariance, 
                                 PF_dynamics_args, PF_measurement_args, roughening_cov, seed)

PF_time_vals = PF_results.t
PF_estimate_vals = PF_results.estimate_vals
PF_anterior_weight_vals = PF_results.anterior_weight_vals
PF_weight_vals = PF_results.weight_vals

PF_converted_truth_vals = converted_truth_vals[:, start_index:]

plot_particles_3d(PF_estimate_vals[0:3], PF_converted_truth_vals[0:3], 0.25, ["X[LU]", "Y[LU]", "Z[LU]"], True)
plot_particles_3d(PF_estimate_vals[3:6], PF_converted_truth_vals[3:6], 0.25, ["Vx[LU]", "Vy[LU]", "Vz[LU]"], False)
plot_particles_3d(PF_estimate_vals[6:9], PF_converted_truth_vals[6:9], 0.25, ["l1[LU]", "l2[LU]", "l3[LU]"], False)
plot_particles_2d(PF_estimate_vals[9:11], PF_converted_truth_vals[9:11], 0.25, ["theta", "psi"])
plot_particles_1d(PF_time_vals, PF_estimate_vals[11, :], PF_converted_truth_vals[11], 0.25, "magnitude")
plot_weights(PF_time_vals, PF_weight_vals)

plt.show()