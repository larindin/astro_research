
import numpy as np
import scipy.integrate
from CR3BP import *


    
def single_shooting(initial_guess, dynamics, constraint, constraint_gradient, guess2integration, tolerance: float, attenuation: float):

    residual = np.array([10, 10, 10])
    guess = initial_guess
    while np.linalg.norm(residual) > tolerance:
        print(guess)
        integration_inputs = guess2integration(guess)
        propagation = scipy.integrate.solve_ivp(dynamics, **integration_inputs)
        
        residual = constraint(propagation)
        gradient = constraint_gradient(propagation)

        guess = guess - attenuation*gradient.T @ np.linalg.inv(gradient @ gradient.T) @ residual
        

    return guess