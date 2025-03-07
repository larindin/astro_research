

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
back_propagation = np.vstack((back_propagation, np.full(np.shape(back_propagation), np.nan)))
back_propagation = np.flip(back_propagation, axis=1)
forward_propagation = scipy.integrate.solve_ivp(dynamics_equation, forprop_tspan, initial_truth, args=truth_dynamics_args, t_eval=forprop_time_vals, atol=1e-12, rtol=1e-12).y
truth_vals = np.concatenate((back_propagation[:, :-1], forward_propagation), axis=1)
time_vals = np.concatenate((np.flip(backprop_time_vals[1:]), forprop_time_vals)) + abs(backprop_time_vals[-1])

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

thrusting_indices = get_thrusting_indices(filter_time, mode_probabilities, thrusting_duration_cutoff)
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

solutions = np.load("lstsqr_solutions/solutions.npy")

num_solutions = np.size(solutions, 1)

jacobians = []
jacobian_stds = np.empty(np.shape(solutions))
for solution_index in range(num_solutions):
    jacobian = np.load("lstsqr_solutions/jac_" + str(solution_index) + ".npy")
    jacobian_stds[:, solution_index] = np.sqrt(np.diag(np.linalg.inv(jacobian.T @ jacobian)))

position_fig = plt.figure().add_subplot(projection="3d")
velocity_fig = plt.figure().add_subplot(projection="3d") 
lambdar_fig = plt.figure().add_subplot(projection="3d")
angles_fig = plt.figure().add_subplot()

position_fig.scatter(truth_converted[0], truth_converted[1], truth_converted[2])
velocity_fig.scatter(truth_converted[3], truth_converted[4], truth_converted[5])
lambdar_fig.scatter(truth_converted[6], truth_converted[7], truth_converted[8])
angles_fig.scatter(truth_converted[9], truth_converted[10])

for solution_index in range(num_solutions):
    position_fig.scatter(solutions[0, solution_index], solutions[1, solution_index], solutions[2, solution_index], s=4, alpha=0.25)
    velocity_fig.scatter(solutions[3, solution_index], solutions[4, solution_index], solutions[5, solution_index], s=4, alpha=0.25)
    lambdar_fig.scatter(solutions[6, solution_index], solutions[7, solution_index], solutions[8, solution_index], s=4, alpha=0.25)
    angles_fig.scatter(solutions[9, solution_index], solutions[10, solution_index], s=4, alpha=0.25)

position_fig.set_aspect("equal")
velocity_fig.set_aspect("equal")
lambdar_fig.set_aspect("equal")
angles_fig.set_aspect("equal")

pos_std_fig = plt.figure().add_subplot(projection="3d")
vel_std_fig = plt.figure().add_subplot(projection="3d") 
lamr_std_fig = plt.figure().add_subplot(projection="3d")
ang_std_fig = plt.figure().add_subplot()

for solution_index in range(num_solutions):
    pos_std_fig.scatter(jacobian_stds[0, solution_index], jacobian_stds[1, solution_index], jacobian_stds[2, solution_index], s=4, alpha=0.25)
    vel_std_fig.scatter(jacobian_stds[3, solution_index], jacobian_stds[4, solution_index], jacobian_stds[5, solution_index], s=4, alpha=0.25)
    lamr_std_fig.scatter(jacobian_stds[6, solution_index], jacobian_stds[7, solution_index], jacobian_stds[8, solution_index], s=4, alpha=0.25)
    ang_std_fig.scatter(jacobian_stds[9, solution_index], jacobian_stds[10, solution_index], s=4, alpha=0.25)

pos_std_fig.set_aspect("equal")
vel_std_fig.set_aspect("equal")
lamr_std_fig.set_aspect("equal")
ang_std_fig.set_aspect("equal")

inputs = np.load("sol_jac.npz")
best_solution = inputs["solution"]
jacobians = []
jacobians.append(inputs["jac"])

magnitudes = -np.arange(-best_solution[-1], -1.02, 0.01)
num_magnitudes = len(magnitudes)
# solutions = np.empty((11, num_magnitudes))

# def min_fuel_STM_ode(t, X, mu, umax, rho):

#     state = X[0:6]
#     costate = X[6:12]
#     STM = np.reshape(X[12:36+12], (6, 6))

#     ddt_state = reformulated_min_fuel_ODE(0, X[0:12], mu, umax, rho)

#     ddt_STM = -CR3BP_jacobian(state, mu).T @ STM
#     ddt_STM = ddt_STM.flatten()

#     return np.concatenate((ddt_state, ddt_STM))

# def magnitude_event(t, X, mu, umax, rho):
#     return X[11] - 1 

# magnitude_event.terminal = True

# solution_tspan = np.array([0, (thrusting_cutoff_offset+additional_measurements)*dt])
# solution_STM_ICs = np.concatenate((best_solution, np.eye(6).flatten()))
# solution_STM_propagation = scipy.integrate.solve_ivp(min_fuel_STM_ode, solution_tspan, solution_STM_ICs, events=magnitude_event, args=truth_dynamics_args, atol=1e-12, rtol=1e-12)

# solution_STM_vals = solution_STM_propagation.y

# solution_STM = solution_STM_vals[12:36+12, -1].reshape((6, 6))
# inv_solution_STM_vr = np.linalg.inv(solution_STM[3:6, 0:3])
# solution_STM_vv = solution_STM[3:6, 3:6]
# # initial_lambdav_hat_solution = solution_STM_vals[9:12, 0]
# # initial_lambdav_hat_solution /= np.linalg.norm(initial_lambdav_hat_solution)
# initial_angles_solution = solution_STM_vals[9:11, 0]
# # final_lambdav_hat_solution = solution_STM_vals[9:12, -1]
# final_angles_solution = solution_STM_vals[9:11, -1]
# initial_lambdavhat_solution = reformulated2standard(solution_STM_vals[6:12, 0])[3:6]/solution_STM_vals[11, 0]
# final_lambdav_solution = reformulated2standard(solution_STM_vals[6:12, -1])[3:6]

# for magnitude_index in range(num_magnitudes):

#     print(magnitude_index)

#     cost_func_args = (magnitudes[magnitude_index], observation_times, observation_measurements, measurement_noise_covariance, obs_positions, observation_check_results, truth_dynamics_args)

#     initial_lambdar_guess = inv_solution_STM_vr @ (final_lambdav_solution - solution_STM_vv @ initial_lambdavhat_solution*magnitudes[magnitude_index])
#     initial_guess = np.concatenate((solution_STM_vals[0:6, 0], initial_lambdar_guess, initial_angles_solution))
#     # initial_guess = np.concatenate((initial_lambdar_guess, np.array([initial_theta, initial_psi])))
#     # initial_guess = observation_truth_vals[:, 0]
#     # initial_guess = truth_converted[0:11].copy()

#     # solution = scipy.optimize.least_squares(measurement_lstsqr_standard, initial_guess, args=cost_func_args, method="lm", verbose=2)
#     solution = scipy.optimize.least_squares(measurement_lstsqr_reformulated, initial_guess, args=cost_func_args, method="lm", verbose=2)

#     solutions[:, magnitude_index] = solution.x
#     jacobians.append(solution.jac)

#     np.save("lstsqr_solutions/sol_" + str(magnitude_index) + ".npy", solution.x)
#     np.save("lstsqr_solutions/jac_" + str(magnitude_index) + ".npy", solution.jac)

#     print(solution.x)
#     print(solution.success)
#     print(solution.message)
#     # print(np.sum(measurement_lstsqr_standard(solution.x, *cost_func_args)**2))
#     print(np.sum(measurement_lstsqr_reformulated(solution.x, *cost_func_args)**2))

# np.save("lstsqr_solutions/solutions.npy", solutions)

my_solution = np.concatenate((solutions[:, -7], [magnitudes[-7]]))
angle_variations = np.arange(-0.25+solutions[-1, -7], 0.05+solutions[-1, -7], 0.01)
num_solutions = len(angle_variations)

proptime = 2
# proptime = observation_times[-1] - observation_times[0]
teval = np.arange(observation_times[0], observation_times[-1]+proptime, dt/5)
tspan = np.array([teval[0], teval[-1]])

truth_propagation = scipy.integrate.solve_ivp(dynamics_equation, tspan, thrusting_truth_vals[:, 0], args=truth_dynamics_args, t_eval=teval, atol=1e-12, rtol=1e-12).y

truth_propagation_control = get_min_fuel_control(truth_propagation[6:12, :], umax, truth_rho)

test_propagations = []
test_controls = []
test_costs = []

print("getting props")
for solution_index in range(num_solutions):
    # ICs = np.concatenate((solutions[:, solution_index], np.array([magnitudes[solution_index]])))
    ICs = np.concatenate((my_solution[0:10], [angle_variations[solution_index]], [my_solution[11]]))
    print(ICs)
    new_propagation = scipy.integrate.solve_ivp(reformulated_min_fuel_ODE, tspan, ICs, args=truth_dynamics_args, t_eval=teval, atol=1e-12, rtol=1e-12).y
    test_propagations.append(new_propagation)
    test_controls.append(get_reformulated_min_fuel_control(new_propagation[6:12, :], umax, truth_rho))
    new_residuals = measurement_lstsqr_standard(ICs, observation_times, observation_measurements, measurement_noise_covariance, 
                                           obs_positions, observation_check_results, truth_dynamics_args )
    test_costs.append(np.sum(new_residuals**2))
    
print(test_costs)

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_propagation[0], truth_propagation[1], truth_propagation[2], alpha=0.5)
for solution_index in range(num_solutions):
    ax.plot(test_propagations[solution_index][0], test_propagations[solution_index][1], test_propagations[solution_index][2], alpha=0.15)
ax.set_aspect("equal")
plot_moon(ax, mu)

control_fig = plt.figure()
control_ax_labels = ["$u_1$", "$u_2$", "$u_3$"]
for ax_index in range(3):
    thing = int("31" + str(ax_index + 1))
    ax = control_fig.add_subplot(thing)
    ax.plot(teval, truth_propagation_control[ax_index], alpha=0.75)
    for solution_index in range(num_solutions):
        ax.plot(teval, test_controls[solution_index][ax_index], alpha=0.25)
    ax.set_ylabel(control_ax_labels[ax_index])
ax.set_xlabel("Time [TU]")

plt.show()
quit()

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