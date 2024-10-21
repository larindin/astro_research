

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.integrate
from configuration_costate_fuel import *
from CR3BP import *
from CR3BP_pontryagin import *
from EKF import *
from dual_filter import *
from dynamics_functions import *
from helper_functions import *
from measurement_functions import *
from plotting import *

time_vals = np.arange(0, final_time, dt)
tspan = np.array([time_vals[0], time_vals[-1]])
truth_propagation = scipy.integrate.solve_ivp(dynamics_equation, tspan, initial_truth, args=(mu, umax, truth_rho), t_eval=time_vals, atol=1e-12, rtol=1e-12)
truth_vals = truth_propagation.y

def CR3BP_DEs(t, state, mu):

    x, y, z, vx, vy, vz = state

    d = np.array([x+mu, y, z])
    r = np.array([x+mu-1, y, z])
    dmag = np.linalg.norm(d)
    rmag = np.linalg.norm(r)

    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = -(1 - mu)*(x + mu)/dmag**3 - mu*(x - 1 + mu)/rmag**3 + 2*vy + x
    dvydt = -(1 - mu)*y/dmag**3 - mu*y/rmag**3 - 2*vx + y
    dvzdt = -(1 - mu)*z/dmag**3 - mu*z/rmag**3

    dXdt = np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt])

    return dXdt

def real_quick(t, X, mu, umax, direction):

    state = X[0:6]
    STM = np.reshape(X[6:36+6], (6, 6))

    B = np.vstack((np.zeros((3, 3)), np.eye(3)))

    control = direction * umax

    ddt_state_kepler = CR3BP_DEs(t, state, mu)
    ddt_state = ddt_state_kepler + B @ control

    ddt_STM = -CR3BP_jacobian(state, mu).T @ STM
    ddt_STM = ddt_STM.flatten()

    return np.concatenate((ddt_state, ddt_STM))


direction = -initial_truth[9:12]/np.linalg.norm(initial_truth[9:12])
estimated_propagation_ICs = np.concatenate((initial_truth[0:6], np.eye(6).flatten()))
estimated_propagation = scipy.integrate.solve_ivp(real_quick, tspan, estimated_propagation_ICs, args=(mu, umax, direction), t_eval=time_vals, atol=1e-12, rtol=1e-12)
estimated_vals = estimated_propagation.y

origin_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, tspan, initial_truth[0:6], args=(mu,), atol=1e-12, rtol=1e-12)
origin_vals = origin_propagation.y

print(np.linalg.norm(initial_truth[9:12]))
# initial_lv = 1.1 * initial_truth[9:12]/np.linalg.norm(initial_truth[9:12])
initial_lv = 1.1 * initial_truth[9:12]
final_lv = initial_truth[9:12]/np.linalg.norm(initial_truth[9:12])


num_propagations = len(timestamps)
initial_lr = np.empty((3, num_propagations))
for index in np.arange(num_propagations):
    timestamp = timestamps[index]
    
    STM = np.reshape(estimated_vals[6:36+6, timestamp], (6, 6))
    STM_vr = STM[3:6, 0:3]
    STM_vv = STM[3:6, 3:6]
    initial_lr[:, index] = np.linalg.inv(STM_vr) @ (final_lv - STM_vv @ initial_lv)

propagations = []
for index in np.arange(num_propagations):
    
    propagation_initial_conditions = np.concatenate((initial_truth[0:6], initial_lr[:, index], initial_lv))
    new_propagation = scipy.integrate.solve_ivp(dynamics_equation, tspan, propagation_initial_conditions, args=(mu, umax, truth_rho), t_eval=time_vals, atol=1e-12, rtol=1e-12)
    propagations.append(new_propagation)


ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
ax.plot(estimated_vals[0], estimated_vals[1], estimated_vals[2])
ax.plot(origin_vals[0], origin_vals[1], origin_vals[2])
ax.set_aspect("equal")

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
for propagation in propagations:
    vals = propagation.y
    ax.plot(vals[0], vals[1], vals[2])
ax.set_aspect("equal")

truth_control = get_min_thrust_control(truth_vals[6:12, :], umax, truth_rho)
fake_control = np.empty((3, len(time_vals)))
fake_control_vector = -initial_truth[9:12]/np.linalg.norm(initial_truth[9:12])
for index in np.arange(len(time_vals)):
    fake_control[:, index] = fake_control_vector
propagation_controls = []
for index in np.arange(num_propagations):
    propagation_control = get_min_thrust_control(propagations[index].y[6:12, :], umax, truth_rho)
    propagation_controls.append(propagation_control)
fig = plt.figure()
ax = fig.add_subplot(311)
ax.step(time_vals, truth_control[0])
for index in np.arange(num_propagations):
    ax.step(time_vals, propagation_controls[index][0], alpha=0.25)
ax = fig.add_subplot(312)
ax.step(time_vals, truth_control[1])
for index in np.arange(num_propagations):
    ax.step(time_vals, propagation_controls[index][1], alpha=0.25)
ax = fig.add_subplot(313)
ax.step(time_vals, truth_control[2])
for index in np.arange(num_propagations):
    ax.step(time_vals, propagation_controls[index][2], alpha=0.25)

fig = plt.figure()
ax = fig.add_subplot(231)
ax.plot(time_vals, truth_vals[6])
for index in np.arange(num_propagations):
    ax.plot(time_vals, propagations[index].y[6], alpha=0.25)
ax = fig.add_subplot(232)
ax.plot(time_vals, truth_vals[7])
for index in np.arange(num_propagations):
    ax.plot(time_vals, propagations[index].y[7], alpha=0.25)
ax = fig.add_subplot(233)
ax.plot(time_vals, truth_vals[8])
for index in np.arange(num_propagations):
    ax.plot(time_vals, propagations[index].y[8], alpha=0.25)
ax = fig.add_subplot(234)
ax.plot(time_vals, truth_vals[9])
for index in np.arange(num_propagations):
    ax.plot(time_vals, propagations[index].y[9], alpha=0.25)
ax = fig.add_subplot(235)
ax.plot(time_vals, truth_vals[10])
for index in np.arange(num_propagations):
    ax.plot(time_vals, propagations[index].y[10], alpha=0.25)
ax = fig.add_subplot(236)
ax.plot(time_vals, truth_vals[11])
for index in np.arange(num_propagations):
    ax.plot(time_vals, propagations[index].y[11], alpha=0.25)

plt.show()