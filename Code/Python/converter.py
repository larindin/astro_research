

import numpy as np
from CR3BP import *

def calculate_JC(IC):
    mu = 1.215059e-2
    x, y, z, vx, vy, vz = IC
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)
    return x**2 + y**2 + 2*((1 - mu)/r1 + (mu)/r2) - (vx**2 + vy**2 + vz**2)

def toICs(solution):

    x, z, vy, T = solution

    IC = np.array([x, 0, z, 0, vy, 0])
    JC = calculate_JC(IC)

    to_be_saved = np.array([x, 0, z, 0, vy, 0, JC, T])
    return to_be_saved
    
def NtoStoN(IC):

    x, y, z, vx, vy, vz, JC, T = IC
    IC = np.array([x, y, -z, vx, vy, vz, JC, T])
    return IC

solutions = np.loadtxt("L2_lyapunov.csv", delimiter=",")

num_sol = np.size(solutions, 0)
total = np.zeros((num_sol, 8))

for sol_index in range(num_sol):
    solution = solutions[sol_index]
    to_be_saved = toICs(solution)
    total[sol_index, :] = to_be_saved

np.savetxt("L2_lyapunov_ICs.csv", total, delimiter=",")

# ICs = np.loadtxt("./L1_halo/L1_halo_southern_ICs.csv", delimiter=",")

# num_IC = np.size(ICs, 0)
# total = ICs*0

# for IC_index in range(num_IC):
#     IC = ICs[IC_index]
#     to_be_saved = NtoStoN(IC)
#     total[IC_index, :] = to_be_saved

# np.savetxt("./L1_halo/L1_halo_northern_ICs.csv", total, delimiter=",")