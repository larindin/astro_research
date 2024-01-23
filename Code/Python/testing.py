

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from CR3BP import *
# from shooting import *


def ode_eq(t, X, mu):

    x, y, z, vx, vy, vz = X

    dXdt = CR3BP_DEs(x, y, z, vx, vy, vz, mu)

    return dXdt

def ode_eq_STM(t, X, mu):

    x, y, z, vx, vy, vz = X[:6]
    STM = X[6:].reshape((6, 6))

    state_derivative = CR3BP_DEs(x, y, z, vx, vy, vz, mu)
    jacobian = CR3BP_jacobian(x, y, z, vx, vy, vz, mu)
    ddtSTM = np.matmul(jacobian, STM)

    ddtX = np.concatenate((state_derivative, ddtSTM.flatten()))

    return ddtX


mu = 1.215059e-2

simulation_options = {"atol":1e-12, "rtol":1e-12, "args":(mu,)}

if 0==0:

    residual = np.ones(3)
    # while np.linalg.norm(residual) >= 1e-12:

    #     inputs = guess2integration(guess)
    #     propagation = scipy.integrate.solve_ivp(**inputs, args=(mu,))
    #     residual = constraint_function(propagation.y)

    #     new_guess = guess - np.linalg.lstsq(constraint_jacobian(guess), residual)
    #     guess = new_guess


initial_state = np.array([0.7783690577, 0, 0, 0, 0.5556648548, 0])
initial_STM = np.eye(6)
ICs = np.concatenate((initial_state, initial_STM.flatten()))
tspan = np.array([0, 3.7214359005])

results = scipy.integrate.solve_ivp(ode_eq_STM, tspan, ICs, **simulation_options)

time = results.t
output = results.y

final_STM = np.reshape(output[6:, -1], (6, 6))

print(final_STM)
print(final_STM @ initial_state)

num_points = len(time)
STM_results = np.zeros((6, num_points))
for time_index in np.arange(num_points):
    STM = np.reshape(output[6:, time_index], (6,6))
    STM_results[:, time_index] = STM @ initial_state

ax = plt.figure().add_subplot(projection="3d")
ax.plot(STM_results[0, :100], STM_results[1, :100], STM_results[2, :100])
ax.plot(output[0, :100], output[1, :100], output[2, :100])

plt.show()
    

quit()

ax = plt.figure().add_subplot()
ax.set_aspect("equal")
ax.set_xlabel("X [dimensionless]")
ax.set_ylabel("Y [dimensionless]")
ax.set_title("CR3BP Orbits in XY Plane")
ax.grid(True)
for run_index in (0, 1, 2):
    ax.plot(results[run_index].y[0], results[run_index].y[1])
ax.legend(["JC = 3.3949", "JC = 2.9123", "JC = 2.6655"])
ax.plot(1 - mu, 0, "ko")

plt.show()