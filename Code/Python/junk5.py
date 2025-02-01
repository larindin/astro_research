import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import time
from configuration_initial_filter_algorithm import *
from CR3BP import *
from CR3BP_pontryagin_reformulated import *
from particle_filter import *
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

initial_timestep = 4
final_timestep = 32

initial_STM = truth_vals[12:, initial_timestep].reshape((6, 6))
final_STM = truth_vals[12:, final_timestep].reshape((6, 6))

initial_costate = truth_vals[6:12, initial_timestep]
final_costate = truth_vals[6:12, final_timestep]

base_costate = initial_truth[6:12]

print(initial_STM @ base_costate)
print(initial_costate)
print(final_STM @ base_costate)
print(final_costate)
print(np.linalg.inv(final_STM) @ final_costate)
print(base_costate)