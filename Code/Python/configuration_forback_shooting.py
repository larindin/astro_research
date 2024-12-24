

import numpy as np
from helper_functions import *

np.set_printoptions(suppress=True, precision=4, linewidth=500)

L2_halo_S = np.array([1.149946551054244370e+00, 0, -1.465767685210560556e-01, 0, -2.181937221565175000e-01, 0, 3.193209656999975277e+00])
L1_lyapunov = np.array([6.139604230137676311e-01, 0, 0, 0, 8.397208166184089162e-01, 0, 6.82])
DRO = np.array([9.255293901714612970e-01, 0, 0, 0, 5.121704760209937479e-01, 0, 0.8])
L1_halo_N = np.array([0.82338697, 0, -0.02282543, 0, 0.13455489, 0, 2.74650039])

boundary_states = [L2_halo_S, L1_lyapunov, DRO, L1_halo_N]

orbit1 = 3
orbit2 = 2

initial_state = boundary_states[orbit1][0:6]
final_state = boundary_states[orbit2][0:6]

patching_time_factor = 0.5
tf = 25 * 24/NONDIM_TIME_HR

costate_guess_distribution = (-1e1, 1e1)

truth_rho = 1e-4

mass = 1500 # kg
thrust = 0.6 # N
umax = thrust/mass / 1000 / NONDIM_LENGTH * NONDIM_TIME**2

num_guesses = 1000
seed = 0

filename = "LT_transfers/solution_" + str(orbit1) + str(orbit2) + "06_1_25.csv"