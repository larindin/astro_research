
import numpy as np
import scipy


initial_guess
constraint_jacobian

guess = initial_guess

def constraint_function(propagation):

    residual = propagation[[1, 3, 5]][-1]

    return residual

def constraint_jacobian()

def guess2integration(guess):

    initial_x = guess[0]
    initial_yvel = guess[1]
    integration_time = guess[2]

    y0 = np.array([initial_x, 0, 0, 0, initial_yvel, 0])
    t_span = np.array([0, integration_time])
    atol = 1e-12
    rtol = 1e-12

    inputs = {"y0":y0, "t_span":t_span, "atol":atol, "rtol":rtol}

    return inputs
    


