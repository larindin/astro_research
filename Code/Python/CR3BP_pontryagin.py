

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

    ddt_state_kepler = CR3BP_DEs(state, mu)
    ddt_state = ddt_state_kepler + B @ control

    ddt_costate = CR3BP_costate_DEs(state, costate, mu)

    return np.concatenate((ddt_state, ddt_costate), 0)

def minimum_thrust_ODE(t, X, mu, umax, rho):

    state = X[0:6]
    costate = X[6:12]

    B = np.vstack((np.zeros((3, 3)), np.eye(3)))

    p = -B.T @ costate
    control = umax/2 * (1 + np.tanh((np.linalg.norm(p) - 1)/rho)) * p/np.linalg.norm(p)

    ddt_state_kepler = CR3BP_DEs(state, mu)
    ddt_state = ddt_state_kepler + B @ control

    ddt_costate = CR3BP_costate_DEs(state, costate, mu)

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

def min_thrust_shooting_function(guess, initial_state, target_state, mu, umax):

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
    final_control = umax/2 * (1 + np.sign(np.linalg.norm(p) - 1)) * p/np.linalg.norm(p)
    final_hamiltonian = np.linalg.norm(final_control)**2 + final_costate.T@(CR3BP_DEs(final_state, mu) + B@final_control)

    constraint = np.concatenate((arrival_constraint, np.array([final_hamiltonian])))

    print(constraint)
    return constraint