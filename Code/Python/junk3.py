import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from configuration_dual_GMIMM import *
from CR3BP import *
from CR3BP_pontryagin_reformulated import *
from IMM import *
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

time_vals = np.arange(0, final_time, dt)
tspan = np.array([time_vals[0], time_vals[-1]])
initial_truth = np.concatenate((initial_truth, np.eye(6).flatten()))
truth_propagation = scipy.integrate.solve_ivp(min_fuel_costateSTM, tspan, initial_truth, args=truth_dynamics_args, t_eval=time_vals, atol=1e-12, rtol=1e-12)
truth_vals = truth_propagation.y

final_thrust_timestep = 50

total_STM = truth_vals[12:, final_thrust_timestep].reshape((6, 6))

initial_costate_original = truth_vals[6:12, 0].copy()
initial_costate = standard2reformulated(initial_costate_original)
final_costate_original = truth_vals[6:12, final_thrust_timestep].copy()
final_costate = standard2reformulated(final_costate_original)

num_particles = 1000
costate_angle_covariance = np.eye(2)*np.deg2rad(5)**2
magnitude_error_std = 1e-1
generator = np.random.default_rng(seed)
initial_costate_angle_errors = generator.multivariate_normal(np.zeros(2), costate_angle_covariance, num_particles)
final_costate_angle_errors = generator.multivariate_normal(np.zeros(2), costate_angle_covariance, num_particles)
initial_magnitude_errors = generator.normal(0, magnitude_error_std, num_particles)
final_magnitude_errors = generator.normal(0, magnitude_error_std, num_particles)

# initial_costate_angle_errors *= 0
# final_costate_angle_errors *= 0
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

initial_costate = initial_truth[6:12]

ax = plt.figure().add_subplot(projection="3d")
ax.scatter(costate_estimates[3], costate_estimates[4], costate_estimates[5], alpha=0.05)
ax.scatter(initial_costate[3], initial_costate[4], initial_costate[5])
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
ax.set_aspect("equal")

ax = plt.figure().add_subplot(projection="3d")
ax.scatter(costate_estimates[0], costate_estimates[1], costate_estimates[2], alpha=0.05)
ax.scatter(initial_costate[0], initial_costate[1], initial_costate[2])
ax.set_aspect("equal")

# propagations = []
# for run_index in np.arange(num_particles*num_kernels):
#     print(run_index)
#     propagation_initial_conditions = np.concatenate((initial_truth[0:6], costate_estimates[:, run_index]))
#     new_propagation = scipy.integrate.solve_ivp(dynamics_equation, tspan, propagation_initial_conditions, args=(mu, umax, truth_rho), t_eval=time_vals, atol=1e-12, rtol=1e-12)
#     propagations.append(new_propagation.y)

# ax = plt.figure().add_subplot(projection="3d")
# ax.plot(truth_vals[0], truth_vals[1], truth_vals[2], alpha=0.75)
# for run_index in np.arange(num_particles*num_kernels):
#     propagation = propagations[run_index]
#     ax.plot(propagation[0], propagation[1], propagation[2], alpha=0.35)
# ax.set_xlabel("X [LU]")
# ax.set_ylabel("Y [LU]")
# ax.set_zlabel("Z [LU]")
# ax.set_aspect("equal")
# plot_moon(ax, mu)

# truth_control = get_min_fuel_control(truth_vals[6:12, :], umax, truth_rho)
# estimated_controls = []
# for index in np.arange(num_particles*num_kernels):
#     estimated_control = get_min_fuel_control(propagations[index][6:12, :], umax, filter_rho)
#     estimated_controls.append(estimated_control)
# fig = plt.figure()
# for ax_index in np.arange(3):
#     thing = int("31" + str(ax_index+1))
#     ax = fig.add_subplot(thing)
#     ax.plot(time_vals, truth_control[ax_index], alpha=0.75)
#     for index in np.arange(num_particles*num_kernels):
#         ax.plot(time_vals, estimated_controls[index][ax_index], alpha=0.35)

# # fig = plt.figure()
# # for ax_index in np.arange(6):
# #     thing = int("61" + str(ax_index+1))
# #     ax = fig.add_subplot(thing)
# #     ax.plot(time_vals, truth_vals[ax_index+6])

plt.show()