

import numpy as np
import scipy
from configuration_rho_homotopy import *
from CR3BP import *
from CR3BP_pontryagin import *
from helper_functions import *
from plotting import *

def min_fuel_ODE_STM(t, X, mu, umax, rho):

    state = X[0:6]
    costate = X[6:12]
    STM = X[12:].reshape((12, 12))

    jacobian = minimum_fuel_jacobian(state, costate, mu, umax, rho)

    ddt_state = minimum_fuel_ODE(t, X[0:12], mu, umax, rho)
    ddt_STM = jacobian @ STM

    return np.concatenate((ddt_state, ddt_STM.flatten()))


def forback_shooting_function(guess, boundary_conditions, tf, patching_time_factor, mu, umax, rho, atol, rtol):

    initial_state = boundary_conditions[0:6]
    final_state = boundary_conditions[6:12]

    initial_costate = guess[0:6]
    final_costate = guess[6:12]

    initial_conditions = np.concatenate((initial_state, initial_costate, np.eye(12).flatten()))
    final_conditions = np.concatenate((final_state, final_costate, np.eye(12).flatten()))

    patching_time = tf * patching_time_factor

    forward_timespan = np.array([0, patching_time])
    forward_propagation = scipy.integrate.solve_ivp(min_fuel_ODE_STM, forward_timespan, initial_conditions, args=(mu, umax, rho), atol=atol, rtol=rtol).y

    backward_timespan = np.array([tf, patching_time])
    backward_propagation = scipy.integrate.solve_ivp(min_fuel_ODE_STM, backward_timespan, final_conditions, args=(mu, umax, rho), atol=atol, rtol=rtol).y

    residual = backward_propagation[0:12, -1] - forward_propagation[0:12, -1]
    
    forward_STM = forward_propagation[12:, -1].reshape((12, 12))
    backward_STM = backward_propagation[12:, -1].reshape((12, 12))
    jacobian = np.concatenate((-forward_STM[:, 6:12], backward_STM[:, 6:12]), axis=1)

    print(residual)

    return residual, jacobian

solutions = []

boundary_conditions = np.concatenate((initial_state, final_state))

rho = truth_rho
scaling_factor = gamma

while rho >= 1e-4:

    print(rho)
    
    solver_args = (boundary_conditions, tf, patching_time_factor, mu, umax, rho, 1e-9, 1e-9)
    solution = scipy.optimize.root(forback_shooting_function, initial_costate_guess, solver_args, jac=True, tol=1e-6)

    sol = solution.x
    fev = solution.fun
    success = solution.success
    status = solution.status

    print(success)
    print(status)
    if success and np.linalg.norm(fev) < 1:
        to_be_printed = np.concatenate((np.array([[rho]]), np.array([sol])), 1)
        solutions.append(to_be_printed)

        if rho == 1e-4:
            break
        initial_costate_guess = sol
        scaling_factor = gamma
        rho = scaling_factor * rho
        if rho < 1e-4:
            rho = 1e-4
    else:
        rho = rho / scaling_factor
        scaling_factor = np.sqrt(scaling_factor)
        rho = rho * scaling_factor

        num_solutions = len(solutions)

solution_array = solutions[-1]
np.savetxt(filename, solution_array, delimiter=",")