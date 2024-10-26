

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.integrate
from configuration_GM_fuel import *
from CR3BP import *
from CR3BP_pontryagin import *
from EKF import *
from dual_filter import *
from helper_functions import *
from measurement_functions import *
from plotting import *

time_vals = np.arange(0, final_time, dt)
tspan = np.array([time_vals[0], time_vals[-1]])
truth_propagation = scipy.integrate.solve_ivp(dynamics_equation, tspan, initial_truth, args=(mu, umax, truth_rho), t_eval=time_vals, atol=1e-12, rtol=1e-12)
truth_vals = truth_propagation.y

initial_costate_estimates = get_min_fuel_initial_costates(truth_vals[0:6, 0], truth_vals[9:12, 0], mu, umax, magnitudes, durations)

propagations = []
for kernel_index in np.arange(num_kernels):
    print(kernel_index)
    propagation_initial_conditions = np.concatenate((truth_vals[0:6, 0], initial_costate_estimates[:, kernel_index]))
    new_propagation = scipy.integrate.solve_ivp(dynamics_equation, tspan, propagation_initial_conditions, args=(mu, umax, truth_rho), t_eval=time_vals, atol=1e-12, rtol=1e-12)
    propagations.append(new_propagation)

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
for propagation in propagations:
    vals = propagation.y
    ax.plot(vals[0], vals[1], vals[2], alpha=0.25)
ax.set_aspect("equal")

truth_control = get_min_fuel_control(truth_vals[6:12, :], umax, truth_rho)
propagation_controls = []
for index in np.arange(num_kernels):
    propagation_control = get_min_fuel_control(propagations[index].y[6:12, :], umax, truth_rho)
    propagation_controls.append(propagation_control)
fig = plt.figure()
ax = fig.add_subplot(311)
ax.step(time_vals, truth_control[0])
for index in np.arange(num_kernels):
    ax.step(time_vals, propagation_controls[index][0], alpha=0.25)
ax = fig.add_subplot(312)
ax.step(time_vals, truth_control[1])
for index in np.arange(num_kernels):
    ax.step(time_vals, propagation_controls[index][1], alpha=0.25)
ax = fig.add_subplot(313)
ax.step(time_vals, truth_control[2])
for index in np.arange(num_kernels):
    ax.step(time_vals, propagation_controls[index][2], alpha=0.25)

fig = plt.figure()
ax = fig.add_subplot(231)
ax.plot(time_vals, truth_vals[6])
for index in np.arange(num_kernels):
    ax.plot(time_vals, propagations[index].y[6], alpha=0.25)
ax = fig.add_subplot(232)
ax.plot(time_vals, truth_vals[7])
for index in np.arange(num_kernels):
    ax.plot(time_vals, propagations[index].y[7], alpha=0.25)
ax = fig.add_subplot(233)
ax.plot(time_vals, truth_vals[8])
for index in np.arange(num_kernels):
    ax.plot(time_vals, propagations[index].y[8], alpha=0.25)
ax = fig.add_subplot(234)
ax.plot(time_vals, truth_vals[9])
for index in np.arange(num_kernels):
    ax.plot(time_vals, propagations[index].y[9], alpha=0.25)
ax = fig.add_subplot(235)
ax.plot(time_vals, truth_vals[10])
for index in np.arange(num_kernels):
    ax.plot(time_vals, propagations[index].y[10], alpha=0.25)
ax = fig.add_subplot(236)
ax.plot(time_vals, truth_vals[11])
for index in np.arange(num_kernels):
    ax.plot(time_vals, propagations[index].y[11], alpha=0.25)

plt.show()