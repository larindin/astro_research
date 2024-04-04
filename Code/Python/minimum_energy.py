

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
target_orbit_ICs = np.array([8.807921420645279387e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,4.701225389091986395e-01,0.000000000000000000e+00])
# target_orbit_ICs = np.array([9.251388163276373922e-01,0.000000000000000000e+00,2.188093146262887201e-01,0.000000000000000000e+00,1.215781574069972060e-01,0.000000000000000000e+00])

initial_orbit_period = 1.9
target_orbit_period = 1.66
# target_orbit_period = 1.806

tau_initial = 0
tau_target = 0

initial_state = scipy.integrate.solve_ivp(CR3BP_ODE, np.array([0, tau_initial*initial_orbit_period]), initial_orbit_ICs, args=(mu,), atol=1e-12, rtol=1e-12).y[:, -1]
target_state = scipy.integrate.solve_ivp(CR3BP_ODE, np.array([0, tau_target*target_orbit_period]), target_orbit_ICs, args=(mu,), atol=1e-12, rtol=1e-12).y[:, -1]

generator = np.random.default_rng(1)
initial_costate_guess = generator.uniform(-1, 1, 6)
initial_tf_guess = np.array([initial_orbit_period])
initial_guess = np.concatenate((initial_costate_guess, initial_tf_guess))

umax = 1

count = 1

while 1 == 1:

    print("Trying with attempt ", count)

    initial_costate_guess = generator.uniform(-1, 1, 6)
    initial_costate_guess = np.array([-2.14405024e-02,  7.31532504e-03,  6.27407077e-11, -2.32202799e-03, 2.77304847e-02, -3.12533967e-11])
    initial_tf_guess = np.array([initial_orbit_period])*2
    initial_tf_guess = np.array([1.7803980197539624])
    initial_guess = np.concatenate((initial_costate_guess, initial_tf_guess))

    solution = scipy.optimize.root(min_energy_shooting_function, initial_guess, args=(initial_state, target_state, mu, umax), tol=1e-8)

    if solution.success == True:
        break

    count += 1
    
print(repr(solution.x[0:6]))
print(repr(solution.x[6]))

initial_costate = solution.x[0:6]
tf = solution.x[6]

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