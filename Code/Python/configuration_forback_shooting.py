

import numpy as np
from helper_functions import *

L2_halo_S = np.array([1.149946551054244370e+00, 0, -1.465767685210560556e-01, 0, -2.181937221565175000e-01, 0, 3.193209656999975277e+00])
L1_lyapunov = np.array([6.139604230137676311e-01, 0, 0, 0, 8.397208166184089162e-01, 0, 6.82])
DRO = np.array([9.255293901714612970e-01, 0, 0, 0, 5.121704760209937479e-01, 0, 0.8])
L1_halo_N = np.array([0.82338697, 0, -0.02282543, 0, 0.13455489, 0, 2.74650039])

boundary_states = [L2_halo_S, L1_lyapunov, DRO, L1_halo_N]

initial_state = boundary_states[0][0:6]
final_state = boundary_states[2][0:6]

patching_time_factor = 0.5
tf = 25 * 24/NONDIM_TIME_HR

costate_guess_distribution = (-1e2, 1e2)

truth_rho = 1

mass = 1500 # kg
thrust = 5 # N
umax = thrust/mass / 1000 / NONDIM_LENGTH * NONDIM_TIME**2

num_guesses = 100
seed = 1