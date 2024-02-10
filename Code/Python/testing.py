

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

    state = X[0:6]
    STM = X[6:].reshape((6, 6))

    state_derivative = CR3BP_DEs(state, mu)
    jacobian = CR3BP_jacobian(state, mu)
    ddtSTM = jacobian @ STM

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




initial_state = np.array([0.85, 0, 0.17546505, 0, 0.2628980369, 0])
perturbation = np.array([1e-4, 0, 0, 0, 0, 0])
perturbed_state = initial_state + perturbation
initial_STM = np.eye(6)
ICs = np.concatenate((initial_state, initial_STM.flatten()))
perturbed_ICs = np.concatenate((perturbed_state, initial_STM.flatten()))
tspan = np.array([0, 2.5543991])
t_eval = np.arange(0, 2*2.5543991, 0.01)

results = scipy.integrate.solve_ivp(ode_eq_STM, np.array([0, t_eval[-1]]), ICs, t_eval=t_eval, **simulation_options)
perturbed_results = scipy.integrate.solve_ivp(ode_eq_STM, np.array([0, t_eval[-1]]), perturbed_ICs, t_eval=t_eval, **simulation_options)

time = results.t
output = results.y
perturbed_time = perturbed_results.t
perturbed_output = perturbed_results.y

differential = perturbed_output - output


final_STM = np.reshape(output[6:, -1], (6, 6))

num_points = len(time)
STM_differential = np.zeros((6, num_points))
for time_index in np.arange(num_points):
    STM = np.reshape(output[6:, time_index], (6,6))
    STM_differential[:, time_index] = STM @ perturbation
STM_output = output[0:6, :] + STM_differential

ax = plt.figure().add_subplot()
ax.plot(time, differential[0])
ax.plot(time, STM_differential[0])

ax = plt.figure().add_subplot()
ax.plot(time, differential[1])
ax.plot(time, STM_differential[1])

ax = plt.figure().add_subplot()
ax.plot(time, differential[2])
ax.plot(time, STM_differential[2])

ax = plt.figure().add_subplot()
ax.plot(time, differential[3])
ax.plot(time, STM_differential[3])

ax = plt.figure().add_subplot()
ax.plot(time, differential[4])
ax.plot(time, STM_differential[4])

ax = plt.figure().add_subplot()
ax.plot(time, differential[5])
ax.plot(time, STM_differential[5])

ax = plt.figure().add_subplot(projection="3d")
ax.plot(perturbed_output[0], perturbed_output[1], perturbed_output[2])
ax.plot(STM_output[0], STM_output[1], STM_output[2])
ax.set_aspect("equal")

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