

import numpy as np
import scipy
import matplotlib.pyplot as plt
from CR3BP import *
from CR3BP_pontryagin import *

def CR3BP_ODE(t, state, mu):
    return CR3BP_DEs(state, mu)

mu = 1.215059e-2
rho = 1

initial_orbit_ICs = np.array([8.690931345289143461e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,4.711295234849984803e-01,0.000000000000000000e+00])
target_orbit_ICs = np.array([1.016894014695402193e+00,0.000000000000000000e+00,-1.783034885833979788e-01,0.000000000000000000e+00,-9.180989348649777615e-02,0.000000000000000000e+00])

initial_orbit_period = 1.9
target_orbit_period = 1.443209657
# target_orbit_period = 1.806

tau_initial = 0
tau_target = 0

initial_state = scipy.integrate.solve_ivp(CR3BP_ODE, np.array([0, tau_initial*initial_orbit_period]), initial_orbit_ICs, args=(mu,), atol=1e-12, rtol=1e-12).y[:, -1]
target_state = scipy.integrate.solve_ivp(CR3BP_ODE, np.array([0, tau_target*target_orbit_period]), target_orbit_ICs, args=(mu,), atol=1e-12, rtol=1e-12).y[:, -1]

generator = np.random.default_rng(5)

umax = 1

count = 1

initial_costate_guess = np.array([ 4.52629885, -3.00736285, -0.31079454,  1.51423959, -0.24420033,
       -0.27148703])

multiplier = 1.75

while multiplier <= 2:

    print("Trying with multiplier ", multiplier)

    # initial_costate_guess = generator.uniform(-1, 1, 6)
    
    # initial_guess = initial_costate_guess
    # initial_tf_guess = generator.uniform(initial_orbit_period, 2*initial_orbit_period, 1)
    # initial_tf_guess = np.array([1.6505217931867644])
    # initial_guess = np.concatenate((initial_costate_guess))
    initial_guess = initial_costate_guess
    final_time = 1.6505217931195273*multiplier

    solution = scipy.optimize.root(min_energy_shooting_function_noT, initial_guess, args=(initial_state, target_state, final_time, mu, umax), tol=1e-8)

    initial_costate_guess = solution.x[0:6]
    multiplier += 0.01
    print(repr(solution.x[0:6]))
    

# print(repr(solution.x[6]))

initial_costate = solution.x[0:6]
# tf = solution.x[6]
tf = final_time

ICs = np.concatenate((initial_state, initial_costate))

sol_integration = scipy.integrate.solve_ivp(minimum_energy_ODE, np.array([0, tf]), ICs, args=(mu, umax), atol=1e-12, rtol=1e-12)
results = sol_integration.y
time = sol_integration.t
control = get_min_energy_control(results[6:12], umax)

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
ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="r")
ax.legend(["Departure Orbit", "Arrival Orbit", "Transfer", "Departure Point", "Target Point", "Moon"])
ax.set_aspect("equal")

ax = plt.figure().add_subplot()
ax.plot(time, np.linalg.norm(control, axis=0))

plt.show()