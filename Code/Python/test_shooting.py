
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
initial_guess = np.array([1.1503, -0.1459, -0.218, 3])

original_solution = single_shooting(initial_guess, CR3BP_STM, CR3BP_periodic_constraint, CR3BP_periodic_constraint_jacobian, 1e-12)

initial_guess_integration = guess2integration(initial_guess)
guess_propagation = scipy.integrate.solve_ivp(CR3BP_STM, **initial_guess_integration)

solution_integration = guess2integration(original_solution)
solution_propagation = scipy.integrate.solve_ivp(CR3BP_STM, **solution_integration)

solution_propagations = []

for b in np.linspace(-4e-3, 4e-3, 5):

    guess = original_solution + np.array([b, 0, 0, 0])
    solution = single_shooting(guess, CR3BP_STM, CR3BP_periodic_constraint, CR3BP_periodic_constraint_jacobian_nox, 1e-12)

    solution_integration = guess2integration(solution)
    solution_propagation = scipy.integrate.solve_ivp(CR3BP_STM, **solution_integration)

    solution_propagations.append(solution_propagation)

ax = plt.figure().add_subplot(projection="3d")
ax.set_aspect("equal")
ax.plot(guess_propagation.y[0], guess_propagation.y[1], guess_propagation.y[2])
for solution_propagation in solution_propagations:
    ax.plot(solution_propagation.y[0], solution_propagation.y[1], solution_propagation.y[2])

plt.show()