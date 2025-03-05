

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
back_propagation = np.vstack((back_propagation, np.ones(np.shape(back_propagation))*1e-9))
back_propagation = np.flip(back_propagation, axis=1)
forward_propagation = scipy.integrate.solve_ivp(dynamics_equation, forprop_tspan, initial_truth, args=truth_dynamics_args, t_eval=forprop_time_vals, atol=1e-12, rtol=1e-12).y
truth_vals = np.concatenate((back_propagation, forward_propagation), axis=1)
time_vals = np.arange(0, final_time+backprop_time, dt)

initial_estimate = np.concatenate((generator.multivariate_normal(truth_vals[0:6, 0], initial_state_covariance), np.ones(6)*1e-6))


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

measurements = generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, measurement_noise_covariance, sensor_position_vals, check_results, seed)

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


dynamics_args = (mu, umax)
measurement_args = (mu, sensor_position_vals, individual_measurement_size)
dynamics_equations = [coasting_dynamics_equation, maneuvering_dynamics_equation]
num_modes = len(dynamics_equations)


filter_output = run_IMM(initial_estimate, initial_covariance, initial_mode_probabilities,
                    dynamics_equations, filter_measurement_equation, measurements,
                    process_noise_covariances, filter_measurement_covariance,
                    dynamics_args, measurement_args, mode_transition_matrix)

filter_time = filter_output.t
posterior_estimate_vals = filter_output.posterior_estimate_vals
posterior_covariance_vals = filter_output.posterior_covariance_vals
anterior_estimate_vals = filter_output.anterior_estimate_vals
mode_probabilities = filter_output.weight_vals

output_estimate_vals, output_covariance_vals = compute_IMM_output(posterior_estimate_vals, posterior_covariance_vals, mode_probabilities)

truth_control = get_min_fuel_control(truth_vals[6:12, :], umax, truth_rho)
first_order = (posterior_estimate_vals[3:6, :] - anterior_estimate_vals[3:6, :])/dt
estimated_control = (posterior_estimate_vals[3:6, :-1] - anterior_estimate_vals[3:6, :-1])/2/dt + (posterior_estimate_vals[3:6, 1:] - anterior_estimate_vals[3:6, 1:])/2/dt
posterior_control = get_min_energy_control(output_estimate_vals[6:12, :], umax)

truth_primer_vectors = compute_primer_vectors(truth_vals[9:12])
estimated_primer_vectors = compute_primer_vectors(output_estimate_vals[9:12])

estimation_errors = compute_estimation_errors(truth_vals, [output_estimate_vals], 12)
three_sigmas = compute_3sigmas([output_covariance_vals], 12)

thrusting_bool = mode_probabilities[1] > 0.5

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
ax.plot(output_estimate_vals[0], output_estimate_vals[1], output_estimate_vals[2])
plot_moon(ax, mu)
ax.set_aspect("equal")

plot_3sigma(time_vals, estimation_errors, three_sigmas, 6)
plot_3sigma_costate(time_vals, estimation_errors, three_sigmas, 6)

ax = plt.figure().add_subplot()
ax.plot(time_vals, mode_probabilities[0])
ax.plot(time_vals, mode_probabilities[1])
ax.plot(time_vals, np.linalg.norm(truth_control, axis=0), alpha=0.5)
# ax.plot(time_vals, thrusting_bool*umax, alpha=0.5)
ax.set_xlabel("Time [TU]")
ax.set_ylabel("Mode Probability")
ax.legend(["Coasting", "Thrusting"])

control_fig = plt.figure()
control_ax_labels = ["$u_1$", "$u_2$", "$u_3$"]
for ax_index in range(3):
    thing = int("31" + str(ax_index + 1))
    ax = control_fig.add_subplot(thing)
    ax.plot(time_vals, truth_control[ax_index], alpha=0.5)
    # ax.plot(time_vals[:-1], estimated_control[ax_index, :, 0], alpha=0.5)
    # ax.plot(time_vals, first_order[ax_index, :, 0], alpha=0.5)
    ax.plot(time_vals, posterior_control[ax_index], alpha=0.5)
    ax.set_ylabel(control_ax_labels[ax_index])
ax.set_xlabel("Time [TU]")
control_fig.legend(["Truth", "Estimated"])

primer_vector_fig = plt.figure()
for ax_index in range(3):
    thing = int("31" + str(ax_index + 1))
    ax = primer_vector_fig.add_subplot(thing)
    ax.plot(time_vals, truth_primer_vectors[ax_index], alpha=0.5)
    ax.plot(time_vals, estimated_primer_vectors[ax_index], alpha=0.5)

costate_fig = plt.figure()
for ax_index in range(6):
    thing = int("61" + str(ax_index+1))
    ax = costate_fig.add_subplot(thing)
    ax.plot(time_vals, output_estimate_vals[ax_index+6])


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

thrusting_indices = get_thrusting_indices(filter_output, thrusting_duration_cutoff)
thrusting_indices[0] = thrusting_indices[1] - thrusting_cutoff_offset
start_index = thrusting_indices[0]
end_index = thrusting_indices[1]
observation_end_index = end_index + additional_measurements

thrusting_times = time_vals[start_index:end_index].copy()
thrusting_times -= thrusting_times[0]
thrusting_states = output_estimate_vals[:, start_index:end_index].copy()
thrusting_covariances = output_covariance_vals[:, :, start_index:end_index].copy()
thrusting_truth_vals = truth_vals[:, start_index:end_index].copy()

observation_times = time_vals[start_index:observation_end_index].copy()
observation_times -= observation_times[0]
obs_positions = sensor_position_vals[:, start_index:observation_end_index].copy()
observation_states = output_estimate_vals[:, start_index:observation_end_index].copy()
observation_covariances = output_covariance_vals[:, :, start_index:observation_end_index].copy()
observation_measurements = measurements.measurements[:, start_index:observation_end_index].copy()
observation_check_results = check_results[:, start_index:observation_end_index].copy()
observation_truth_vals = truth_vals[:, start_index:observation_end_index].copy()

truth_converted = np.concatenate((observation_truth_vals[0:6, 0], standard2reformulated(observation_truth_vals[6:12, 0])))
print(truth_converted[-1])

# truth_cost = state_cost_function(truth_converted[0:11], truth_converted[11], observation_times, observation_states, observation_covariances, truth_dynamics_args)
# truth_residuals = measurement_lstsqr_reformulated(truth_converted[0:11], truth_converted[11], observation_times, observation_measurements, measurement_noise_covariance,
#                                        obs_positions, observation_check_results, truth_dynamics_args)
truth_residuals = measurement_lstsqr_standard(observation_truth_vals[:, 0], observation_times, observation_measurements, measurement_noise_covariance,
                                       obs_positions, observation_check_results, truth_dynamics_args )
print(np.sum(truth_residuals**2))

# magnitudes = np.linalg.norm(observation_truth_vals[9:12, 0]) * np.array([1.0, 1.1, 1.2])
magnitudes = np.linalg.norm(observation_truth_vals[9:12, 0]) * np.array([1.2])
# magnitudes = np.array([1.1, 1.2, 1.3])
# magnitudes = np.array([1.1])

initial_state_guess = thrusting_states[0:6, 0].copy()
initial_lambdav_hat_guess = thrusting_states[9:12, 0].copy()
initial_lambdav_hat_guess /= np.linalg.norm(initial_lambdav_hat_guess)
initial_theta, initial_psi = standard2reformulated(np.concatenate((np.zeros(3), initial_lambdav_hat_guess)))[3:5]
# initial_theta, initial_psi = truth_converted[9:11]
final_lambdav_guess = thrusting_states[9:12, -1].copy()
final_lambdav_guess /= np.linalg.norm(final_lambdav_guess)

STM = get_costate_STM(thrusting_times, thrusting_states, mu)

STM_vr = STM[3:6, 0:3]
STM_vv = STM[3:6, 3:6]

num_guesses = len(magnitudes)
# solutions = np.empty((11, num_guesses))
solutions = np.empty((12, num_guesses))
jacobians = []
for guess_index in range(num_guesses):

    print(guess_index)

    initial_lambdav_guess = initial_lambdav_hat_guess * magnitudes[guess_index]

    # cost_func_args = (magnitudes[guess_index], observation_times, observation_states, observation_covariances, truth_dynamics_args)
    cost_func_args = (observation_times, observation_measurements, measurement_noise_covariance, obs_positions, observation_check_results, truth_dynamics_args)
    # cost_func_args = (magnitudes[guess_index], observation_times, observation_measurements, measurement_noise_covariance, obs_positions, observation_check_results, truth_dynamics_args)
    # cost_func_args = (observation_states[0:6, 0], magnitudes[guess_index], observation_times, observation_measurements, measurement_noise_covariance, obs_positions, observation_check_results, truth_dynamics_args)

    initial_lambdar_guess = np.linalg.inv(STM_vr) @ (final_lambdav_guess - STM_vv @ initial_lambdav_guess)
    # initial_guess = np.concatenate((initial_state_guess, initial_lambdar_guess, initial_lambdav_guess))
    # initial_guess = np.concatenate((initial_state_guess, initial_lambdar_guess, np.array([initial_theta, initial_psi])))
    initial_guess = np.concatenate((initial_state_guess, initial_lambdar_guess, np.array([initial_theta, initial_psi, magnitudes[guess_index]])))
    # initial_guess = np.concatenate((initial_lambdar_guess, np.array([initial_theta, initial_psi])))
    # initial_guess = observation_truth_vals[:, 0]
    # initial_guess = truth_converted[0:11].copy()

    # # solution = scipy.optimize.minimize(state_cost_function, initial_guess, cost_func_args, method="BFGS")
    # # solution = scipy.optimize.minimize(measurement_cost_costate, initial_guess, cost_func_args, method="BFGS")
    # # solution = scipy.optimize.least_squares(measurement_lstsqr_standard, initial_guess, args=cost_func_args, method="lm", verbose=2)
    # # solution = scipy.optimize.least_squares(measurement_lstsqr_reformulated, initial_guess, args=cost_func_args, method="lm", verbose=2)
    # solution = scipy.optimize.least_squares(measurement_lstsqr_reformulated_mag, initial_guess, args=cost_func_args, method="lm", verbose=2)
    # # solution = scipy.optimize.least_squares(measurement_lstsqr_costate, initial_guess, args=cost_func_args, method="lm", verbose=2)

    # solutions[:, guess_index] = solution.x
    # jacobians.append(solution.jac)
    # print(solution.x)
    # print(solution.success)
    # print(solution.message)
    # # print(np.sum(measurement_lstsqr_standard(solution.x, *cost_func_args)**2))
    # # print(np.sum(measurement_lstsqr_reformulated(solution.x, *cost_func_args)**2))
    # print(np.sum(measurement_lstsqr_reformulated_mag(solution.x, *cost_func_args)**2))

inputs = np.load("sol_jac.npz")
solutions[:, 0] = inputs["solution"]
jacobians.append(inputs["jac"])

proptime = 2.0
# proptime = observation_times[-1] - observation_times[0]
teval = np.arange(observation_times[0], observation_times[-1]+proptime, dt/5)
tspan = np.array([teval[0], teval[-1]])

truth_propagation = scipy.integrate.solve_ivp(dynamics_equation, tspan, thrusting_truth_vals[:, 0], args=truth_dynamics_args, t_eval=teval, atol=1e-12, rtol=1e-12).y

# ICs = np.concatenate((initial_guess, np.array([magnitudes[0]])))
# initial_guess_propagation = scipy.integrate.solve_ivp(minimum_fuel_ODE, tspan, initial_guess, args=truth_dynamics_args, t_eval=teval, atol=1e-12, rtol=1e-12).y
# initial_guess_propagation = scipy.integrate.solve_ivp(reformulated_min_fuel_ODE, tspan, ICs, args=truth_dynamics_args, t_eval=teval, atol=1e-12, rtol=1e-12).y
initial_guess_propagation = scipy.integrate.solve_ivp(reformulated_min_fuel_ODE, tspan, solutions[:, 0], args=truth_dynamics_args, t_eval=teval, atol=1e-12, rtol=1e-12).y

truth_propagation_control = get_min_fuel_control(truth_propagation[6:12, :], umax, truth_rho)
initial_guess_control = get_reformulated_min_fuel_control(initial_guess_propagation[6:12, :], umax, truth_rho)
# initial_guess_control = get_min_fuel_control(initial_guess_propagation[6:12, :], umax, truth_rho)

test_propagations = []
test_controls = []
test_errors = []

for guess_index in range(num_guesses):
    # ICs = np.concatenate((observation_states[0:6, 0], solutions[:, guess_index], np.array([magnitudes[guess_index]])))
    # ICs = np.concatenate((solutions[:, guess_index], np.array([magnitudes[guess_index]])))
    ICs = solutions[:, guess_index]
    # new_propagation = scipy.integrate.solve_ivp(minimum_fuel_ODE, tspan, ICs, args=truth_dynamics_args, t_eval=teval, atol=1e-12, rtol=1e-12).y
    new_propagation = scipy.integrate.solve_ivp(reformulated_min_fuel_ODE, tspan, ICs, args=truth_dynamics_args, t_eval=teval, atol=1e-12, rtol=1e-12).y
    test_propagations.append(new_propagation)
    # test_controls.append(get_min_fuel_control(new_propagation[6:12, :], umax, truth_rho))
    test_controls.append(get_reformulated_min_fuel_control(new_propagation[6:12, :], umax, truth_rho))
    test_errors.append(new_propagation - truth_propagation)

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

solution_tspan = np.array([0, (thrusting_cutoff_offset+additional_measurements)*dt])
solution_STM_ICs = np.concatenate((solutions[:, 0], np.eye(6).flatten()))
solution_STM_propagation = scipy.integrate.solve_ivp(min_fuel_STM_ode, solution_tspan, solution_STM_ICs, events=magnitude_event, args=truth_dynamics_args, atol=1e-12, rtol=1e-12)

solution_STM_vals = solution_STM_propagation.y

solution_STM = solution_STM_vals[12:36+12, -1].reshape((6, 6))
inv_solution_STM_vr = np.linalg.inv(solution_STM[3:6, 0:3])
solution_STM_vv = solution_STM[3:6, 3:6]
# initial_lambdav_hat_solution = solution_STM_vals[9:12, 0]
# initial_lambdav_hat_solution /= np.linalg.norm(initial_lambdav_hat_solution)
initial_angles_solution = solution_STM_vals[9:11, 0]
# final_lambdav_hat_solution = solution_STM_vals[9:12, -1]
final_angles_solution = solution_STM_vals[9:11, -1]

solution_mean = solutions[:, 0]
state_mean = solution_mean[0:6]
jacobian = jacobians[0]
sampling_covariance = np.linalg.inv(jacobians[guess_index].T @ jacobians[guess_index])

print(truth_converted)
print(solution_mean)
print(np.sqrt(sampling_covariance[9:11, 9:11]))
quit()

def get_costs(value):
    return np.sum(measurement_lstsqr_reformulated_mag(value, *cost_func_args)**2)

def get_propagations(value): 
    return scipy.integrate.solve_ivp(reformulated_min_fuel_ODE, tspan, value, args=truth_dynamics_args, t_eval=teval, atol=1e-12, rtol=1e-12).y

def get_controls(propagation):
    return get_reformulated_min_fuel_control(propagation[6:12, :], umax, truth_rho)

# particle_ICs = np.empty((12, num_particles))
# sampled_perturbations = generator.multivariate_normal(np.zeros(12), sampling_covariance, num_particles).T
# second_angle_perturbations = generator.multivariate_normal(np.zeros(2), sampling_covariance[9:11, 9:11], num_particles).T
# state_perturbations = sampled_perturbations[0:6]
# first_angle_perturbations = sampled_perturbations[9:11]
# first_magnitudes = generator.uniform(*first_magnitude_sampling_range, num_particles)
# second_magnitudes = generator.uniform(*second_magnitude_sampling_range, num_particles)
# for particle_index in range(num_particles):
#     particle_state = state_mean + state_perturbations[:, particle_index]
#     initial_particle_angles = initial_angles_solution + first_angle_perturbations[:, particle_index]
#     final_particle_angles = final_angles_solution + second_angle_perturbations[:, particle_index]

#     particle_initial_lambdav = reformulated2standard(np.concatenate((np.zeros(3), initial_particle_angles, [first_magnitudes[particle_index]])))[3:6]
#     particle_final_lambdav = reformulated2standard(np.concatenate((np.zeros(3), final_particle_angles, [second_magnitudes[particle_index]])))[3:6]
#     particle_lambdar = inv_solution_STM_vr @ (particle_final_lambdav - solution_STM_vv @ particle_initial_lambdav)

#     particle_ICs[:, particle_index] = np.concatenate((particle_state, particle_lambdar, initial_particle_angles, [first_magnitudes[particle_index]]))

chi2_cutoff = get_chi2_cutoff(6*(thrusting_cutoff_offset+additional_measurements)-12, 0.003)
# chi2_cutoff = 150
print(chi2_cutoff)
remaining_particles = num_particles
particle_ICs = np.empty((12, num_particles))
while remaining_particles > 0:
    
    print(remaining_particles)

    ICs = np.empty((12, remaining_particles))

    sampled_perturbations = generator.multivariate_normal(np.zeros(12), sampling_covariance, remaining_particles).T
    second_angle_perturbations = generator.multivariate_normal(np.zeros(2), sampling_covariance[9:11, 9:11], remaining_particles).T
    state_perturbations = sampled_perturbations[0:6]
    first_angle_perturbations = sampled_perturbations[9:11]
    first_magnitudes = generator.uniform(*first_magnitude_sampling_range, remaining_particles)
    second_magnitudes = generator.uniform(*second_magnitude_sampling_range, remaining_particles)
    
    for particle_index in range(remaining_particles):
        particle_state = state_mean + state_perturbations[:, particle_index]
        initial_particle_angles = initial_angles_solution + first_angle_perturbations[:, particle_index]
        final_particle_angles = final_angles_solution + second_angle_perturbations[:, particle_index]

        particle_initial_lambdav = reformulated2standard(np.concatenate((np.zeros(3), initial_particle_angles, [first_magnitudes[particle_index]])))[3:6]
        particle_final_lambdav = reformulated2standard(np.concatenate((np.zeros(3), final_particle_angles, [second_magnitudes[particle_index]])))[3:6]
        particle_lambdar = inv_solution_STM_vr @ (particle_final_lambdav - solution_STM_vv @ particle_initial_lambdav)

        ICs[:, particle_index] = np.concatenate((particle_state, particle_lambdar, initial_particle_angles, [first_magnitudes[particle_index]]))

    particle_costs = Parallel(n_jobs=8)(delayed(get_costs)(ICs[:, particle_index]) for particle_index in range(remaining_particles))

    for particle_index in range(remaining_particles):
        if particle_costs[particle_index] < chi2_cutoff:
            particle_ICs[:, remaining_particles-1] = ICs[:, particle_index]
            remaining_particles -= 1


# print(truth_converted[9], truth_converted[10], truth_converted[11])
# print(solution_mean[9], solution_mean[10], solution_mean[11])
# print(angle_perturbations)
# print(particle_ICs[9])
# print(particle_ICs[10])
# print(particle_ICs[11])

particle_propagations = []
particle_controls = []
particle_costs = []

# particle_ICs = generator.multivariate_normal(solution_mean, sampling_covariance, num_particles).T
# chi2_cutoff = get_chi2_cutoff(6*(thrusting_cutoff_offset+additional_measurements)-12, 0.003)
# print(chi2_cutoff)
# remaining_particles = num_particles
# particle_ICs = np.empty((12, num_particles))
# while remaining_particles > 0:
    
#     print(remaining_particles)

#     ICs = generator.multivariate_normal(solution_mean, sampling_covariance, remaining_particles)
#     particle_costs = Parallel(n_jobs=8)(delayed(big_function)(ICs[particle_index, :]) for particle_index in range(remaining_particles))

#     for particle_index in range(remaining_particles):
#         if particle_costs[particle_index] < chi2_cutoff:
#             particle_ICs[:, remaining_particles-1] = ICs[particle_index, :]
#             remaining_particles -= 1


print("getting costs")
particle_costs = Parallel(n_jobs=8)(delayed(get_costs)(particle_ICs[:, particle_index]) for particle_index in range(num_particles))
print("getting propagations")
particle_propagations = Parallel(n_jobs=8)(delayed(get_propagations)(particle_ICs[:, particle_index]) for particle_index in range(num_particles))
print("getting controls")
particle_controls = Parallel(n_jobs=8)(delayed(get_controls)(particle_propagations[particle_index]) for particle_index in range(num_particles))

ax = plt.figure().add_subplot()
# ax.hist(np.array(particle_costs), bins=(60, 70, 80, 90, 100, 110, 120))
ax.hist(np.array(particle_costs), bins=(50, 100, 150, 200, 250, 1000, 10000, 100000))
ax.set_xscale("log")

np.set_printoptions(suppress=True, precision=5, linewidth=500)

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_propagation[0], truth_propagation[1], truth_propagation[2], alpha=0.5)
ax.plot(initial_guess_propagation[0], initial_guess_propagation[1], initial_guess_propagation[2], alpha=0.5)
# # print(observation_truth_vals[:, 0])
# print(np.concatenate((observation_truth_vals[0:6, 0], standard2reformulated(observation_truth_vals[6:12, 0]))))
# for guess_index in range(num_guesses):
#     ax.plot(test_propagations[guess_index][0], test_propagations[guess_index][1], test_propagations[guess_index][2], alpha=0.5)
#     print(solutions[:, guess_index])
#     # print(np.concatenate((solutions[:, guess_index], np.array([magnitudes[guess_index]]))))
#     print(np.sqrt(np.diag(np.linalg.inv(jacobians[guess_index].T @ jacobians[guess_index]))))
for particle_index in range(num_particles):
    ax.plot(particle_propagations[particle_index][0], particle_propagations[particle_index][1], particle_propagations[particle_index][2], alpha=0.15)
ax.set_aspect("equal")
plot_moon(ax, mu)

ax = plt.figure().add_subplot()
ax.hist(particle_ICs[11], bins=np.linspace(1.02, 1.2, 19))

ax = plt.figure().add_subplot(projection="3d")
ax.scatter(truth_converted[6], truth_converted[7], truth_converted[8], alpha=0.75)
ax.scatter(solution_mean[6], solution_mean[7], solution_mean[8], alpha=0.75)
ax.scatter(particle_ICs[6], particle_ICs[7], particle_ICs[8], alpha=0.2)
ax.set_aspect("equal")

ax = plt.figure().add_subplot()
ax.scatter(truth_converted[9], truth_converted[10], alpha=0.75)
ax.scatter(initial_angles_solution[0], initial_angles_solution[1], alpha=0.75)
ax.scatter(particle_ICs[9], particle_ICs[10], alpha=0.2)
ax.set_aspect("equal")

test_errors_fig = plt.figure()
test_errors_ax_nums = [231, 232, 233, 234, 235, 236]
test_errors_ax_labels = ["X", "Y", "Z", "Vx", "Vy", "Vz"]
for ax_index in range(6):
    thing = test_errors_ax_nums[ax_index]
    ax = test_errors_fig.add_subplot(thing)
    ax.plot(teval, truth_propagation[ax_index])
    for guess_index in range(num_guesses):
        ax.plot(teval, test_propagations[guess_index][ax_index])
    ax.set_ylabel(test_errors_ax_labels[ax_index])
    ax.grid(True)

test_control_fig = plt.figure()
for ax_index in range(3):
    thing = int("31" + str(ax_index + 1))
    ax = test_control_fig.add_subplot(thing)
    ax.plot(teval, truth_propagation_control[ax_index], alpha=0.5)
    ax.plot(teval, initial_guess_control[ax_index], alpha=0.5)
    # for guess_index in range(num_guesses):
    #     ax.plot(teval, test_controls[guess_index][ax_index], alpha=0.5)
    for particle_index in range(num_particles):
        ax.plot(teval, particle_controls[particle_index][ax_index], alpha=0.15)

plt.show()