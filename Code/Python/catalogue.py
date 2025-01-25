

import numpy as np

# Only 10, 12, 13, 20, 21, 31, 32 are valid

L2_halo_S = np.array([1.149946551054244370e+00, 0, -1.465767685210560556e-01, 0, -2.181937221565175000e-01, 0, 3.193209656999975277e+00])
L1_lyapunov = np.array([6.139604230137676311e-01, 0, 0, 0, 8.397208166184089162e-01, 0, 6.82])
DRO = np.array([9.255293901714612970e-01, 0, 0, 0, 5.121704760209937479e-01, 0, 0.8])
L1_halo_N = np.array([0.82338697, 0, -0.02282543, 0, 0.13455489, 0, 2.74650039])

boundary_states = [L2_halo_S, L1_lyapunov, DRO, L1_halo_N]

costate_1_0 = np.loadtxt("LT_transfers/solution_10_06_e4_25.csv", delimiter=",")[0:6]
costate_1_2 = np.loadtxt("LT_transfers/solution_12_06_e4_25.csv", delimiter=",")[0:6]
costate_1_3 = np.loadtxt("LT_transfers/solution_13_06_e4_25.csv", delimiter=",")[0:6]
costate_2_0 = np.loadtxt("LT_transfers/solution_20_06_e4_25.csv", delimiter=",")[0:6]
costate_2_1 = np.loadtxt("LT_transfers/solution_21_06_e4_25.csv", delimiter=",")[0:6]
costate_2_3 = np.loadtxt("LT_transfers/solution_23_06_e4_25.csv", delimiter=",")[0:6]
costate_3_0 = np.loadtxt("LT_transfers/solution_30_06_e4_27.csv", delimiter=",")[0:6]
costate_3_1 = np.loadtxt("LT_transfers/solution_31_06_e4_25.csv", delimiter=",")[0:6]
costate_3_2 = np.loadtxt("LT_transfers/solution_32_06_e4_25.csv", delimiter=",")[0:6]

costates = [[0, 0, 0, 0],
            [costate_1_0, 0, costate_1_2, costate_1_3],
            [costate_2_0, costate_2_1, 0, costate_2_3],
            [costate_3_0, costate_3_1, costate_3_2, 0]]