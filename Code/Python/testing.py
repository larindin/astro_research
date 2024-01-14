

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from CR3BP import *


def ode_eq(t, X, mu):

    x, y, z, vx, vy, vz = X

    dXdt = CR3BP_DEs_rotating(x, y, z, vx, vy, vz, mu)

    return dXdt


if __name__ == "__main__":

    mu = 1.215059e-2
    initial_x = 0.4625690577
    initial_yvel = 1.3253326531
    initial_conditions = np.array([initial_x, 0, 0, 0, initial_yvel, 0])
    tspan = np.array([0, 7])

    simulation_options = {"atol":1e-12, "rtol":1e-12, "args":(mu,)}

    results = scipy.integrate.solve_ivp(ode_eq, tspan, initial_conditions, **simulation_options)

    ax = plt.figure().add_subplot()
    ax.plot(results.y[0], results.y[1])
    ax.set_aspect("equal")
    plt.show()