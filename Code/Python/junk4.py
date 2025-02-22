
import numpy as np
import scipy
import matplotlib.pyplot as plt
from configuration_forback_shooting import *
from CR3BP_pontryagin import *
from plotting import *

# initial_costate = np.loadtxt("LT_transfers/solution_" + str(orbit1) + str(orbit2) + "_06_e4_25.csv", delimiter=",")
initial_costate10 = np.loadtxt("LT_transfers/solution_" + str(1) + str(0) + "_06_e4_25.csv", delimiter=",")
initial_costate12 = np.loadtxt("LT_transfers/solution_" + str(1) + str(2) + "_06_e4_25.csv", delimiter=",")
initial_costate13 = np.loadtxt("LT_transfers/solution_" + str(1) + str(3) + "_06_e4_25.csv", delimiter=",")
initial_costate20 = np.loadtxt("LT_transfers/solution_" + str(2) + str(0) + "_06_e4_25.csv", delimiter=",")
initial_costate21 = np.loadtxt("LT_transfers/solution_" + str(2) + str(1) + "_06_e4_25.csv", delimiter=",")
initial_costate31 = np.loadtxt("LT_transfers/solution_" + str(3) + str(1) + "_06_e4_25.csv", delimiter=",")
initial_costate32 = np.loadtxt("LT_transfers/solution_" + str(3) + str(2) + "_06_e4_25.csv", delimiter=",")

initial_costates = [initial_costate10, initial_costate12, initial_costate13, initial_costate20, initial_costate21, initial_costate31, initial_costate32]

state0 = boundary_states[0][0:6]
state1 = boundary_states[1][0:6]
state2 = boundary_states[2][0:6]
state3 = boundary_states[3][0:6]

initial_indices = [1, 1, 1, 2, 2, 3, 3]
final_indices = [0, 2, 3, 0, 1, 1, 2]

num_transfers = len(initial_costates)


# for transfer_index in np.arange(num_transfers):

#     initial_costate = initial_costates[transfer_index]
#     initial_index = initial_indices[transfer_index]
#     final_index = final_indices[transfer_index]

#     initial_state = boundary_states[initial_index][0:6]
#     final_state = boundary_states[final_index][0:6]



#     initial_conditions = np.concatenate((initial_state, initial_costate[0:6]))
#     final_conditions = np.concatenate((final_state, initial_costate[6:12]))

#     forward_tspan = np.array([0, tf*patching_time_factor])
#     backward_tspan = np.array([tf, tf*patching_time_factor])
#     forward_result = scipy.integrate.solve_ivp(minimum_fuel_ODE, forward_tspan, initial_conditions, args=(mu, umax, truth_rho), atol=1e-12, rtol=1e-12)
#     forward_propagation = forward_result.y
#     forward_time = forward_result.t
#     backward_result = scipy.integrate.solve_ivp(minimum_fuel_ODE, backward_tspan, final_conditions, args=(mu, umax, truth_rho), atol=1e-12, rtol=1e-12)
#     backward_propagation = backward_result.y
#     backward_time = backward_result.t

#     total_time = np.concatenate((forward_time, np.flip(backward_time)))
#     total_propagation = np.concatenate((forward_propagation, np.flip(backward_propagation, 1)), axis=1)

#     control = get_min_fuel_control(total_propagation[6:12, :], umax, truth_rho)

#     moon_vectors = np.vstack((total_propagation[0] - 1 + mu, total_propagation[1], total_propagation[2]))
#     moon_distances = np.linalg.norm(moon_vectors, axis=0)

#     initial_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, np.array([0, 6]), initial_state, args=(mu,), atol=1e-12, rtol=1e-12).y
#     final_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, np.array([0, 6]), final_state, args=(mu,), atol=1e-12, rtol=1e-12).y

    
#     ax.plot(initial_propagation[0], initial_propagation[1], initial_propagation[2], c="b", alpha=0.3)
#     ax.plot(final_propagation[0], final_propagation[1], final_propagation[2], c="b", alpha=0.3)
#     ax.plot(forward_propagation[0], forward_propagation[1], forward_propagation[2], c="r", alpha=0.3)
#     ax.plot(backward_propagation[0], backward_propagation[1], backward_propagation[2], c="r", alpha=0.3)
#     plot_moon(ax, mu)
#     ax.set_aspect("equal")

initial_state = boundary_states[1][0:6]
final_state = boundary_states[0][0:6]

initial_perturbation = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]) * -1e-2

ICs = np.concatenate((initial_state, initial_costate10[0:6]))
timespan = np.array([0, tf])

initial_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, np.array([0, 8]), initial_state, args=(mu,), atol=1e-12, rtol=1e-12).y
final_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, np.array([0, 8]), final_state, args=(mu,), atol=1e-12, rtol=1e-12).y
total_propagation = scipy.integrate.solve_ivp(minimum_fuel_ODE, timespan, ICs, args=(mu, umax, truth_rho), atol=1e-12, rtol=1e-12)
perturbed = scipy.integrate.solve_ivp(minimum_fuel_ODE, timespan, ICs + initial_perturbation, args=(mu, umax, truth_rho), atol=1e-12, rtol=1e-12)

propagation = total_propagation.y
time = total_propagation.t

perturbed_propagation = perturbed.y
perturbed_time = perturbed.t

control = get_min_fuel_control(propagation[6:12, :], umax, truth_rho)
perturbed_control = get_min_fuel_control(perturbed_propagation[6:12, :], umax, truth_rho)

ax = plt.figure().add_subplot(projection="3d")
# ax.plot(initial_propagation[0], initial_propagation[1], initial_propagation[2])
# ax.plot(final_propagation[0], final_propagation[1], final_propagation[2])
ax.plot(propagation[0], propagation[1], propagation[2])
ax.plot(perturbed_propagation[0], perturbed_propagation[1], perturbed_propagation[2])
ax.scatter(initial_state[0], initial_state[1], initial_state[2], "^b")
ax.scatter(final_state[0], final_state[1], final_state[2], "vo")
plot_moon(ax, mu)
ax.set_aspect("equal")
ax.set_xlabel("X [LU]")
ax.set_ylabel("Y [LU]")
ax.set_zlabel("Z [LU]")

labels = ["$u_1$", "$u_2$", "$u_3$"]
fig = plt.figure()
for ax_index in np.arange(3):
    thing = int("41" + str(ax_index + 1))
    ax = fig.add_subplot(thing)
    ax.plot(time, control[ax_index])
    ax.plot(perturbed_time, perturbed_control[ax_index])
    ax.set_ylabel(labels[ax_index])
ax = fig.add_subplot(414)
ax.plot(time, np.linalg.norm(control, axis=0))
ax.plot(perturbed_time, np.linalg.norm(perturbed_control, axis=0))
ax.set_ylabel(r"|u|")
ax.set_xlabel("Time [TU]")
# ax = plt.figure().add_subplot()
# ax.plot(total_time, moon_distances*NONDIM_LENGTH)
# ax.hlines((1740, 1840), 0, total_time[-1])

plt.show()