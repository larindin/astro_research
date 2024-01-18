

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
    STM = np.reshape(X[6:], (6, 6))

    state_derivative = CR3BP_DEs(x, y, z, vx, vy, vz, mu)
    jacobian = CR3BP_jacobian(x, y, z, vx, vy, vz, mu)
    ddtSTM = jacobian @ STM

    ddtX = np.concatenate((state_derivative, ddtSTM.flatten()))

    return ddtX

if __name__ == "__main__":

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
    
    integration_times = np.array([0.2230147974, 3.7214359005, 6.0500729893])
    initial_xs = np.array([0.9624690577, 0.7783690577, 0.4625690577])
    initial_vys = np.array([0.7184165432, 0.5556648548, 1.3253326531])

    results = []

    for run_index in (0, 1, 2):

        t_final = integration_times[run_index]
        initial_x = initial_xs[run_index]
        initial_vy = initial_vys[run_index]

        initial_conditions = np.array([initial_x, 0, 0, 0, initial_vy, 0])

        result = scipy.integrate.solve_ivp(ode_eq, np.array([0, t_final]), initial_conditions, **simulation_options)
        results.append(result)

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