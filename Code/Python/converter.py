

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
    

solutions = np.loadtxt("./L2_halo/L2_halo_southern_ICs.csv", delimiter=",")

num_sol = np.size(solutions, 0)
total = np.zeros((num_sol, 8))

for sol_index in np.arange(num_sol):
    solution = solutions[sol_index]
    to_be_saved = toICs(solution)
    total[sol_index, :] = to_be_saved

np.savetxt("L2_halo_southern_ICs.csv", total, delimiter=",")