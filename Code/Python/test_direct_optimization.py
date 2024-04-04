

import numpy as np
import scipy
import matplotlib.pyplot as plt
from direct_optimization import *

mu = 1.215059e-2

initial_orbit = [6.600080203317177929e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,7.788798338729785442e-01,0.000000000000000000e+00,2.842415841982109281e+00,5.240000000000017089e+00]
final_orbit = [6.865799958354950050e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,7.211593261631034091e-01,0.000000000000000000e+00,2.859538447132872641e+00,5.000000000000022204e+00]

initial_orbit_ICs = initial_orbit[0:6]
final_orbit_ICs = final_orbit[0:6]
initial_orbit_period = initial_orbit[-1]
final_orbit_period = final_orbit[-1]

num_nodes = 100
Isp = 3000
Tmax = 0.04
m0 = 500
epsilon = 1e-4
final_time_guess = initial_orbit_period

position_continuity_constraints = []
for node_num in np.arange(1, num_nodes):
    new_constraint = {"type":"eq", "fun":CR3BP_position_continuity_constraint, "args":(node_num, num_nodes, mu)}
    position_continuity_constraints.append(new_constraint)

mass_continuity_constraints = []
for node_num in np.arange(1, num_nodes):
    new_constraint = {"type":"eq", "fun":mass_continuity_constraint, "args":(node_num, num_nodes, Isp, epsilon, mu)}
    mass_continuity_constraints.append(new_constraint)
mass_continuity_constraints.append({"type":"eq", "fun":initial_mass_continuity_constraint, "args":(initial_orbit_ICs, m0, Isp, epsilon, mu)})

thrust_constraints = []
for node_num in np.arange(1, num_nodes):
    new_constraint = {"type":"ineq", "fun":thrust_constraint, "args":(node_num, num_nodes, Tmax, epsilon, mu)}
    thrust_constraints.append(new_constraint)
thrust_constraints.append({"type":"ineq", "fun":initial_thrust_constraint, "args":(num_nodes, initial_orbit_ICs, Tmax, epsilon, mu)})

misc_constraints = []

new_constraint = {"type":"eq", "fun":initial_orbit_constraint, "args":(initial_orbit_ICs, mu)}
misc_constraints.append(new_constraint)
new_constraint = {"type":"eq", "fun":initial_tau_constraint, "args":(0,)}
misc_constraints.append(new_constraint)
new_constraint = {"type":"eq", "fun":final_orbit_constraint, "args":(final_orbit_ICs, mu)}
misc_constraints.append(new_constraint)
new_constraint = {"type":"eq", "fun":final_tau_constraint, "args":(0,)}
misc_constraints.append(new_constraint)
new_constraint = {"type":"eq", "fun":initial_mass_constraint, "args":(m0,)}
misc_constraints.append(new_constraint)

constraints = position_continuity_constraints + mass_continuity_constraints + thrust_constraints + misc_constraints

# Construct initial guess
initial_guess = np.array([])
initial_guess_masses = np.zeros(num_nodes) + m0
initial_guess_states = np.zeros((num_nodes, 6))

initial_orbit_propagation = scipy.integrate.solve_ivp(CR3BP_ODE, np.array([0, initial_orbit_period]), initial_orbit_ICs, t_eval=np.linspace(0, initial_orbit_period, int(num_nodes)), args=(mu,), rtol=1e-12, atol=1e-12).y
final_orbit_propagation = scipy.integrate.solve_ivp(CR3BP_ODE, np.array([0, final_orbit_period]), final_orbit_ICs, t_eval=np.linspace(0, final_orbit_period, int(num_nodes)), args=(mu,), rtol=1e-12, atol=1e-12).y
for node_num in np.arange(int(num_nodes)):
    initial_guess_states[node_num, :] = initial_orbit_propagation[:, node_num]
# for node_num in np.arange(int(num_nodes/2), num_nodes):
#     initial_guess_states[node_num, :] = final_orbit_propagation[:, node_num - int(num_nodes/2)]
initial_guess_states.flatten()

for node_num in np.arange(num_nodes):
    initial_guess = np.concatenate((initial_guess, initial_guess_states[node_num]), 0)
    initial_guess = np.concatenate((initial_guess, initial_guess_masses[node_num, None]), 0)
initial_guess = np.concatenate((initial_guess, np.array([initial_orbit_period, 0, 0])), 0)

solution = scipy.optimize.minimize(minimum_mass_objective, initial_guess, method="SLSQP", constraints=constraints, tol=1e-6, options={"disp":True})
output = solution.x

x = output[np.linspace(0, 7*num_nodes, num_nodes)]
y = output[np.linspace(1, 7*num_nodes+1, num_nodes)]

ax = plt.figure().add_subplot()
ax.plot(x, y)

plt.show()