

import numpy as np
import matplotlib.pyplot as plt
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

    B = np.vstack((np.zeros((3, 3)), np.eye(3)))

    control = direction * umax

    ddt_state_kepler = CR3BP_DEs(t, state, mu)
    ddt_state = ddt_state_kepler + B @ control

    return ddt_state

direction = -initial_truth[9:12]/np.linalg.norm(initial_truth[9:12])
estimated_propagation = scipy.integrate.solve_ivp(real_quick, tspan, initial_truth[0:6], args=(mu, umax, direction), atol=1e-12, rtol=1e-12)
estimated_vals = estimated_propagation.y

origin_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, tspan, initial_truth[0:6], args=(mu,), atol=1e-12, rtol=1e-12)
origin_vals = origin_propagation.y

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
ax.plot(estimated_vals[0], estimated_vals[1], estimated_vals[2])
ax.plot(origin_vals[0], origin_vals[1], origin_vals[2])
ax.set_aspect("equal")

truth_control = get_min_thrust_control(truth_vals[6:12, :], umax, truth_rho)
fake_control = np.empty((3, len(time_vals)))
fake_control_vector = -initial_truth[9:12]/np.linalg.norm(initial_truth[9:12])
for index in np.arange(len(time_vals)):
    fake_control[:, index] = fake_control_vector
fig = plt.figure()
ax = fig.add_subplot(311)
ax.step(time_vals, truth_control[0])
ax.step(time_vals, fake_control[0])
ax = fig.add_subplot(312)
ax.step(time_vals, truth_control[1])
ax.step(time_vals, fake_control[1])
ax = fig.add_subplot(313)
ax.step(time_vals, truth_control[2])
ax.step(time_vals, fake_control[2])

plt.show()