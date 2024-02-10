
import numpy as np
import scipy.integrate
from CR3BP import *


def guess2integration(guess):

    initial_x = guess[0]
    initial_z = guess[1]
    initial_yvel = guess[2]
    integration_time = guess[3]

    initial_state = np.array([initial_x, 0, initial_z, 0, initial_yvel, 0])
    initial_STM = np.eye(6)
    y0 = np.concatenate((initial_state, initial_STM.flatten()))
    t_span = np.array([0, integration_time])
    atol = 1e-12
    rtol = 1e-12
    inputs = {"y0":y0, "t_span":t_span, "atol":atol, "rtol":rtol}

    return inputs

    
def single_shooting(initial_guess, dynamics, constraint, constraint_gradient, tolerance: float):

    residual = np.array([10, 10, 10])
    guess = initial_guess
    while np.linalg.norm(residual) > tolerance:
        integration_inputs = guess2integration(guess)
        propagation = scipy.integrate.solve_ivp(dynamics, **integration_inputs)
        
        residual = constraint(propagation)
        gradient = constraint_gradient(propagation)

        guess = guess - 0.5*gradient.T @ np.linalg.inv(gradient @ gradient.T) @ residual
        print(guess)

    return guess