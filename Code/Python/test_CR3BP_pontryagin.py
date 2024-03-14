

import numpy as np
import scipy
import matplotlib.pyplot as plt
from CR3BP import *
from CR3BP_pontryagin import *

mu = 1.215059e-2

initial_state = np.array([8.690931345289143461e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,4.711295234849984803e-01,0.000000000000000000e+00])
target_state = np.array([9.003532227272393884e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,4.772732705106432216e-01,0.000000000000000000e+00])
target_state = np.array([9.251388163276373922e-01,0.000000000000000000e+00,2.188093146262887201e-01,0.000000000000000000e+00,1.215781574069972060e-01,0.000000000000000000e+00])

generator = np.random.default_rng(8)
initial_costate_guess = generator.uniform(-1, 1, 6)
initial_tf_guess = np.array([1.900000000000086064e+00])
initial_guess = np.concatenate((initial_costate_guess, initial_tf_guess))
initial_guess = np.array([2.95572981, -1.42927365, -1.31411278,  0.71418243,  0.46577776,  0.76251288, 2.26612172])
initial_guess = np.array([3.326658, -1.68744601, -1.44844744, 0.83396345, 0.47519227, 0.79534702, 2.27737826])
initial_guess = np.array([4.03614954, -2.19485316, -1.71557662,  1.05814049,  0.48757237, 0.85669964, 2.29264179])
initial_guess = np.array([0.53764911, -0.38209534, -0.23687759 , 0.18051313 ,-0.08999829,  0.50075501, 2.77222073])
initial_guess = np.array([1.57530098 ,-0.98334662, -0.59654574 , 0.48099917 ,-0.06922038 ,0.87602963, 3])

umax = 0.29

solution = scipy.optimize.root(min_energy_shooting_function, initial_guess, args=(initial_state, target_state, mu, umax), tol=1e-8)

print(solution.x)

initial_costate = solution.x[0:6]
tf = solution.x[6]

ICs = np.concatenate((initial_state, initial_costate))

sol_integration = scipy.integrate.solve_ivp(minimum_energy_ODE, np.array([0, tf]), ICs, args=(mu, umax), atol=1e-12, rtol=1e-12)
results = sol_integration.y

def ODE(t, X, mu):
    return CR3BP_DEs(X, mu)

departure_integration = scipy.integrate.solve_ivp(ODE, np.array([0, 1.9]), initial_state, args=(mu,), atol=1e-12, rtol=1e-12)
arrival_integration = scipy.integrate.solve_ivp(ODE, np.array([0, 1.806]), target_state, args=(mu,), atol=1e-12, rtol=1e-12)
departure_results = departure_integration.y
arrival_results = arrival_integration.y

ax = plt.figure().add_subplot(projection="3d")

ax.plot(departure_results[0], departure_results[1], departure_results[2])
ax.plot(arrival_results[0], arrival_results[1], arrival_results[2])
ax.plot(results[0], results[1], results[2])
ax.scatter(initial_state[0], initial_state[1], initial_state[2])
ax.scatter(target_state[0], target_state[1], target_state[2])
ax.legend(["Departure Orbit", "Arrival Orbit", "Transfer", "Departure Point", "Target Point"])
ax.set_aspect("equal")

plt.show()