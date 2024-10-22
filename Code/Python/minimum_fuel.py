

import numpy as np
import scipy
import matplotlib.pyplot as plt
from CR3BP import *
from CR3BP_pontryagin import *

def CR3BP_ODE(t, state, mu):
    return CR3BP_DEs(state, mu)

mu = 1.215059e-2


initial_orbit_ICs = np.array([8.690931345289143461e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,4.711295234849984803e-01,0.000000000000000000e+00])
target_orbit_ICs = np.array([8.807921420645279387e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,4.701225389091986395e-01,0.000000000000000000e+00])

initial_orbit_period = 1.9
target_orbit_period = 1.66

tau_initial = 0
tau_target = 0

initial_state = scipy.integrate.solve_ivp(CR3BP_ODE, np.array([0, tau_initial*initial_orbit_period]), initial_orbit_ICs, args=(mu,), atol=1e-12, rtol=1e-12).y[:, -1]
target_state = scipy.integrate.solve_ivp(CR3BP_ODE, np.array([0, tau_target*target_orbit_period]), target_orbit_ICs, args=(mu,), atol=1e-12, rtol=1e-12).y[:, -1]

initial_costate_guess = np.array([-3.10213973e-01, -5.23214620e-01,  0,  2.22433613e-01, 9.46091171e-01,  0])
initial_tf_guess = np.array([1.792034912415646])
initial_guess = np.concatenate((initial_costate_guess, initial_tf_guess))

umax = 0.07
rho = 0.001953125

while rho >= 1e-5:

    print("running with rho = ", rho)

    attempt = scipy.optimize.root(min_fuel_shooting_function, initial_guess, args=(initial_state, target_state, mu, umax, rho), tol=1e-8)

    if attempt.success == True:
        solution = attempt
    
    if attempt.success == False:
        break

    initial_guess = attempt.x

    rho /= 2

    
print(repr(solution.x[0:6]))
print(repr(solution.x[6]))

initial_costate = solution.x[0:6]
tf = solution.x[6]

ICs = np.concatenate((initial_state, initial_costate))

sol_integration = scipy.integrate.solve_ivp(minimum_fuel_ODE, np.array([0, tf]), ICs, args=(mu, umax, rho), atol=1e-12, rtol=1e-12)
results = sol_integration.y
time = sol_integration.t
control = get_min_fuel_control(results[6:12], umax, rho)

departure_integration = scipy.integrate.solve_ivp(CR3BP_ODE, np.array([0, initial_orbit_period]), initial_state, args=(mu,), atol=1e-12, rtol=1e-12)
arrival_integration = scipy.integrate.solve_ivp(CR3BP_ODE, np.array([0, target_orbit_period]), target_state, args=(mu,), atol=1e-12, rtol=1e-12)
departure_results = departure_integration.y
arrival_results = arrival_integration.y


# draw sphere
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
sphere_x = np.cos(u)*np.sin(v) * 0.00452653486 + (1 - mu)
sphere_y = np.sin(u)*np.sin(v) * 0.00452653486
sphere_z = np.cos(v) * 0.00452653486


ax = plt.figure().add_subplot(projection="3d")

ax.plot(departure_results[0], departure_results[1], departure_results[2])
ax.plot(arrival_results[0], arrival_results[1], arrival_results[2])
ax.plot(results[0], results[1], results[2])
ax.scatter(initial_state[0], initial_state[1], initial_state[2])
ax.scatter(target_state[0], target_state[1], target_state[2])
try: 
    ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="r")
except:
    pass
ax.legend(["Departure Orbit", "Arrival Orbit", "Transfer", "Departure Point", "Target Point"])
ax.set_aspect("equal")

ax = plt.figure().add_subplot()
ax.plot(time, np.linalg.norm(control, axis=0))

plt.show()