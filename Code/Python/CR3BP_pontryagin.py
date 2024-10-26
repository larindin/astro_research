

import numpy as np
import scipy.integrate
from CR3BP import *


def minimum_energy_ODE(t, X, mu, umax):

    state = X[0:6]
    costate = X[6:12]

    B = np.vstack((np.zeros((3, 3)), np.eye(3)))

    p = -B.T @ costate
    p_mag = np.linalg.norm(p)

    if p_mag > 2*umax:
        control = umax * p/p_mag
    else:
        control = p/2

    ddt_state_kepler = CR3BP_DEs(t, state, mu)
    ddt_state = ddt_state_kepler + B @ control

    ddt_costate = CR3BP_costate_DEs(t, state, costate, mu)

    return np.concatenate((ddt_state, ddt_costate), 0)

def minimum_fuel_ODE(t, X, mu, umax, rho):

    state = X[0:6]
    costate = X[6:12]

    B = np.vstack((np.zeros((3, 3)), np.eye(3)))

    p = -B.T @ costate
    control = umax/2 * (1 + np.tanh((np.linalg.norm(p) - 1)/rho)) * p/np.linalg.norm(p)

    ddt_state_kepler = CR3BP_DEs(t, state, mu)
    ddt_state = ddt_state_kepler + B @ control

    ddt_costate = CR3BP_costate_DEs(t, state, costate, mu)

    return np.concatenate((ddt_state, ddt_costate), 0)

def min_energy_shooting_function(guess, initial_state, target_state, mu, umax):

    initial_costate = guess[0:6]
    tf = guess[6]

    ICs = np.concatenate((initial_state, initial_costate))
    timespan = np.array([0, tf])

    result = scipy.integrate.solve_ivp(minimum_energy_ODE, timespan, ICs, args=(mu, umax), atol=1e-12, rtol=1e-12)

    final = result.y[:, -1]
    final_state = final[0:6]
    final_costate = final[6:12]

    arrival_constraint = final_state - target_state

    B = np.vstack((np.zeros((3, 3)), np.eye(3)))
    p = -B.T @ final_costate
    p_mag = np.linalg.norm(p)
    if p_mag > 2*umax:
        final_control = umax * p/p_mag
    else:
        final_control = p/2
    final_hamiltonian = np.linalg.norm(final_control)**2 + final_costate.T@(CR3BP_DEs(final_state, mu) + B@final_control)

    constraint = np.concatenate((arrival_constraint, np.array([final_hamiltonian])))

    print(constraint)
    return constraint

def min_energy_shooting_function_noT(guess, initial_state, target_state, tf, mu, umax):

    initial_costate = guess

    ICs = np.concatenate((initial_state, initial_costate))
    timespan = np.array([0, tf])

    result = scipy.integrate.solve_ivp(minimum_energy_ODE, timespan, ICs, args=(mu, umax), atol=1e-12, rtol=1e-12)

    final_state = result.y[0:6, -1]

    constraint = final_state - target_state

    print(constraint)
    return constraint

def min_fuel_shooting_function(guess, initial_state, target_state, mu, umax, rho):

    initial_costate = guess[0:6]
    tf = guess[6]

    ICs = np.concatenate((initial_state, initial_costate))
    timespan = np.array([0, tf])

    result = scipy.integrate.solve_ivp(minimum_fuel_ODE, timespan, ICs, args=(mu, umax, rho), atol=1e-12, rtol=1e-12)

    final = result.y[:, -1]
    final_state = final[0:6]
    final_costate = final[6:12]

    arrival_constraint = final_state - target_state

    B = np.vstack((np.zeros((3, 3)), np.eye(3)))
    p = -B.T @ final_costate
    final_control = umax/2 * (1 + np.sign(np.linalg.norm(p) - 1)) * p/np.linalg.norm(p)
    final_hamiltonian = np.linalg.norm(final_control)**2 + final_costate.T@(CR3BP_DEs(final_state, mu) + B@final_control)

    constraint = np.concatenate((arrival_constraint, np.array([final_hamiltonian])))

    print(constraint)
    return constraint

def get_min_energy_control(costate_output, umax):

    B = np.vstack((np.zeros((3, 3)), np.eye(3)))

    control = costate_output[0:3]*0
    for time_index in np.arange(len(costate_output[0])):
        
        costate = costate_output[:, time_index]

        p = -B.T @ costate
        p_mag = np.linalg.norm(p)

        if p_mag > 2*umax:
            control[:, time_index] = umax * p/p_mag
        else:
            control[:, time_index] = p/2
    
    return control

def get_min_fuel_control(costate_output, umax, rho):

    B = np.vstack((np.zeros((3, 3)), np.eye(3)))

    control = costate_output[0:3]*0
    for time_index in np.arange(len(costate_output[0])):

        costate = costate_output[:, time_index]

        p = -B.T @ costate
        control[:, time_index] = umax/2 * (1 + np.tanh((np.linalg.norm(p) - 1)/rho)) * p/np.linalg.norm(p)
    
    return control

def get_min_fuel_initial_costates(initial_state, initial_lv, mu, umax, magnitudes, durations):

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

    tf = durations[-1]
    tspan = np.array([0, tf])
    num_magnitudes = len(magnitudes)
    num_durations = len(durations)
    num_costates = num_magnitudes * num_durations
    costate_estimates = np.empty((6, num_costates))

    direction = -initial_lv/np.linalg.norm(initial_lv)
    final_lv = -direction

    constant_thrust_ICs = np.concatenate((initial_state, np.eye(6).flatten()))
    constant_thrust_propagation = scipy.integrate.solve_ivp(constant_thrust_ODE, tspan, constant_thrust_ICs, args=(mu, umax, direction), t_eval=durations, atol=1e-12, rtol=1e-12)
    constant_thrust_vals = constant_thrust_propagation.y

    costate_index = 0
    for magnitude_index in np.arange(num_magnitudes):

        estimated_initial_lv = -magnitudes[magnitude_index] * direction

        for duration_index in np.arange(num_durations):

            STM = np.reshape(constant_thrust_vals[6:36+6, duration_index], (6, 6))
            STM_vr = STM[3:6, 0:3]
            STM_vv = STM[3:6, 3:6]
            initial_lr = np.linalg.inv(STM_vr) @ (final_lv - STM_vv @ initial_lv)

            costate_estimates[:, costate_index] = np.concatenate((initial_lr, estimated_initial_lv))

            costate_index += 1

    return costate_estimates