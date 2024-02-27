
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from CR3BP import *
from dynamics_functions import *
from shooting import *


def CR3BP_periodic_constraint(propagation):

    output = propagation.y
    residual = output[[1, 3, 5], -1]

    return residual

def CR3BP_periodic_constraint_jacobian(propagation):

    output = propagation.y
    final_state = output[0:6, -1]
    final_STM = output[6:42, -1].reshape((6, 6))

    final_ddtState = CR3BP_DEs(final_state, 1.215059e-2)

    jacobian = np.array([[final_STM[1, 0], final_STM[1, 2], final_STM[1, 4], final_ddtState[1]],
                         [final_STM[3, 0], final_STM[3, 2], final_STM[3, 4], final_ddtState[3]],
                         [final_STM[5, 0], final_STM[5, 2], final_STM[5, 4], final_ddtState[4]]])

    return jacobian

def CR3BP_periodic_constraint_jacobian_nox(propagation):

    output = propagation.y
    final_state = output[0:6, -1]
    final_STM = output[6:42, -1].reshape((6, 6))

    final_ddtState = CR3BP_DEs(final_state, 1.215059e-2)

    jacobian = np.array([[0, final_STM[1, 2], final_STM[1, 4], final_ddtState[1]],
                         [0, final_STM[3, 2], final_STM[3, 4], final_ddtState[3]],
                         [0, final_STM[5, 2], final_STM[5, 4], final_ddtState[4]]])

    return jacobian

def CR3BP_periodic_constraint_jacobian_noz(propagation):

    output = propagation.y
    final_state = output[0:6, -1]
    final_STM = output[6:42, -1].reshape((6, 6))

    final_ddtState = CR3BP_DEs(final_state, 1.215059e-2)

    jacobian = np.array([[final_STM[1, 0], 0, final_STM[1, 4], final_ddtState[1]],
                         [final_STM[3, 0], 0, final_STM[3, 4], final_ddtState[3]],
                         [final_STM[5, 0], 0, final_STM[5, 4], final_ddtState[4]]])

    return jacobian

def CR3BP_periodic_constraint_jacobian_not(propagation):

    output = propagation.y
    final_STM = output[6:42, -1].reshape((6, 6))

    jacobian = np.array([[final_STM[1, 0], final_STM[1, 2], final_STM[1, 4], 0],
                         [final_STM[3, 0], final_STM[3, 2], final_STM[3, 4], 0],
                         [final_STM[5, 0], final_STM[5, 2], final_STM[5, 4], 0]])

    return jacobian

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


def CR3BP_STM(t, X):
    mu = 1.215059e-2
    state = X[0:6]
    STM = X[6:42].reshape((6, 6))

    jacobian = CR3BP_jacobian(state, mu)

    ddtState = CR3BP_DEs(state, mu)
    ddtSTM = jacobian @ STM

    ddtX = np.concatenate((ddtState, ddtSTM.flatten()))
    return ddtX

# initial_state = np.array([1.1503 0 −0.1459 0 −0.2180 0)
initial_guess = np.array([1.1277174668299654, 0, 1.4106927720325751e-1, 3.40])

original_solution = single_shooting(initial_guess, CR3BP_STM, CR3BP_periodic_constraint, CR3BP_periodic_constraint_jacobian_not, guess2integration, 1e-9, 0.5)

solutions = [original_solution]

index = 0
try: 
    while solutions[-1][3] < 8.2:

        guess = solutions[-1] + np.array([0, 0, 0, 1e-2])
        new_solution = single_shooting(guess, CR3BP_STM, CR3BP_periodic_constraint, CR3BP_periodic_constraint_jacobian_not, guess2integration, 1e-9, 0.5)
        solutions.append(new_solution)
        index += 1
except:
    solutions_tobesaved = np.zeros((4, len(solutions)))
    for solution_index, solution in enumerate(solutions):
        solutions_tobesaved[:, solution_index] = solution
    np.savetxt("L2_lyapunov.csv", solutions_tobesaved.T, delimiter=",")

solutions_tobesaved = np.zeros((4, len(solutions)))
for solution_index, solution in enumerate(solutions):
    solutions_tobesaved[:, solution_index] = solution
np.savetxt("L2_lyapunov.csv", solutions_tobesaved.T, delimiter=",")