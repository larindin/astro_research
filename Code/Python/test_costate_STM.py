

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from joblib import Parallel, delayed
from configuration_initial_filter_algorithm import *
from CR3BP import *
from CR3BP_pontryagin import *
from EKF import *
from dual_filter import *
from helper_functions import *
from measurement_functions import *
from plotting import *

def min_fuel_STM_ode(t, X, mu, umax, rho):

    state = X[0:6]
    costate = X[6:12]
    STM = np.reshape(X[12:36+12], (6, 6))

    ddt_state = minimum_fuel_ODE(0, X[0:12], mu, umax, rho)

    ddt_STM = -CR3BP_jacobian(state, mu).T @ STM
    ddt_STM = ddt_STM.flatten()

    return np.concatenate((ddt_state, ddt_STM))

time_vals = np.arange(0, final_time, dt)
tspan = np.array([time_vals[0], time_vals[-1]])
truth_ICs = np.concatenate((initial_truth, np.eye(6).flatten()))
truth_propagation = scipy.integrate.solve_ivp(min_fuel_STM_ode, tspan, truth_ICs, args=(mu, umax, truth_rho), t_eval=time_vals, atol=1e-12, rtol=1e-12)
truth_vals = truth_propagation.y

# final_time_index = 15 # 1 to 3
final_time_index = 28
STM = truth_vals[12:36+12, final_time_index].reshape((6, 6))
initial_lambdav = truth_vals[9:12, 0]
final_lambdav = truth_vals[9:12, final_time_index]

initial_lambdav_hat = initial_lambdav.copy() / np.linalg.norm(initial_lambdav)
final_lambdav_hat = final_lambdav.copy() / np.linalg.norm(final_lambdav)

print(final_lambdav)
print(np.linalg.norm(final_lambdav))

STM_vr = STM[3:6, 0:3]
STM_vv = STM[3:6, 3:6]
num_magnitudes = 100
magnitudes = np.linspace(1.001, 2.0, num_magnitudes)

initial_costate_guesses = np.empty((6, num_magnitudes))

final_lambdav_guess = final_lambdav_hat
for magnitude_index in np.arange(num_magnitudes):
    
    initial_lambdav_guess = magnitudes[magnitude_index] * initial_lambdav_hat
    initial_lambdar_guess = np.linalg.inv(STM_vr) @ (final_lambdav_guess - STM_vv @ initial_lambdav_guess)
    initial_costate_guesses[:, magnitude_index] = np.concatenate((initial_lambdar_guess, initial_lambdav_guess))


teval = np.arange(0, 2.5, dt)

truth_propagation = scipy.integrate.solve_ivp(dynamics_equation, np.array([teval[0], teval[-1]]), initial_truth, args=truth_dynamics_args, t_eval=teval, atol=1e-12, rtol=1e-12).y
def get_propagations(initial_state, initial_costate):
    value = np.concatenate((initial_state, initial_costate)) 
    return scipy.integrate.solve_ivp(dynamics_equation, np.array([teval[0], teval[-1]]), value, args=truth_dynamics_args, t_eval=teval, atol=1e-12, rtol=1e-12).y

print("getting propagations")
propagations = Parallel(n_jobs=8)(delayed(get_propagations)(truth_vals[0:6, 0], initial_costate_guesses[:, magnitude_index]) for magnitude_index in range(num_magnitudes))

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_propagation[0], truth_propagation[1], truth_propagation[2], alpha=0.65)
for propagation in propagations:
    ax.plot(propagation[0], propagation[1], propagation[2], alpha=0.15)
ax.set_aspect("equal")

truth_control = get_min_fuel_control(truth_propagation[6:12, :], umax, truth_rho)
propagation_controls = []
for index in np.arange(num_magnitudes):
    propagation_control = get_min_fuel_control(propagations[index][6:12, :], umax, truth_rho)
    propagation_controls.append(propagation_control)

test_control_fig = plt.figure()
for ax_index in np.arange(3):
    thing = int("31" + str(ax_index + 1))
    ax = test_control_fig.add_subplot(thing)
    ax.plot(teval, truth_control[ax_index], alpha=0.5)
    # ax.plot(time_vals, initial_guess_control[ax_index], alpha=0.5)
    for magnitude_index in np.arange(num_magnitudes):
        ax.plot(teval, propagation_controls[magnitude_index][ax_index], alpha=0.15)

ax = plt.figure().add_subplot(projection="3d")
for magnitude_index in np.arange(num_magnitudes):
    ax.scatter(initial_costate_guesses[0, magnitude_index], initial_costate_guesses[1, magnitude_index], initial_costate_guesses[2, magnitude_index])

ax = plt.figure().add_subplot()
ax.plot(teval, np.linalg.norm(truth_propagation[9:12], axis=0))
for propagation in propagations:
    ax.plot(teval, np.linalg.norm(propagation[9:12], axis=0), alpha=0.15)
ax.hlines(1, teval[0], teval[-1], "black", "dashed")

plt.show()
quit()

fig = plt.figure()
ax = fig.add_subplot(231)
ax.plot(time_vals, truth_vals[6])
for index in np.arange(num_kernels):
    ax.plot(time_vals, propagations[index].y[6], alpha=0.25)
ax = fig.add_subplot(232)
ax.plot(time_vals, truth_vals[7])
for index in np.arange(num_kernels):
    ax.plot(time_vals, propagations[index].y[7], alpha=0.25)
ax = fig.add_subplot(233)
ax.plot(time_vals, truth_vals[8])
for index in np.arange(num_kernels):
    ax.plot(time_vals, propagations[index].y[8], alpha=0.25)
ax = fig.add_subplot(234)
ax.plot(time_vals, truth_vals[9])
for index in np.arange(num_kernels):
    ax.plot(time_vals, propagations[index].y[9], alpha=0.25)
ax = fig.add_subplot(235)
ax.plot(time_vals, truth_vals[10])
for index in np.arange(num_kernels):
    ax.plot(time_vals, propagations[index].y[10], alpha=0.25)
ax = fig.add_subplot(236)
ax.plot(time_vals, truth_vals[11])
for index in np.arange(num_kernels):
    ax.plot(time_vals, propagations[index].y[11], alpha=0.25)

plt.show()