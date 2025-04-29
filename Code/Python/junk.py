

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from configuration_IMM import *
from CR3BP import *
from CR3BP_pontryagin import *
from IMM import *
from helper_functions import *
from measurement_functions import *
from plotting import *

backprop_time_vals = -np.arange(0, backprop_time, dt)
forprop_time_vals = np.arange(0, final_time, dt)
additional_time_vals = np.arange(forprop_time_vals[-1], forprop_time_vals[-1]+additional_time, dt)
backprop_tspan = np.array([backprop_time_vals[0], backprop_time_vals[-1]])
forprop_tspan = np.array([forprop_time_vals[0], forprop_time_vals[-1]])
additional_tspan = np.array([additional_time_vals[0], additional_time_vals[-1]])
back_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, backprop_tspan, initial_truth[0:6], args=(mu,), t_eval=backprop_time_vals, atol=1e-12, rtol=1e-12).y
back_propagation = np.vstack((back_propagation, np.full(np.shape(back_propagation), 1e-12)))
back_propagation = np.flip(back_propagation, axis=1)
forward_propagation = scipy.integrate.solve_ivp(dynamics_equation, forprop_tspan, initial_truth, args=truth_dynamics_args, t_eval=forprop_time_vals, atol=1e-12, rtol=1e-12).y
additional_propagation = scipy.integrate.solve_ivp(CR3BP_DEs, additional_tspan, forward_propagation[0:6, -1], args=(mu,), t_eval=additional_time_vals, atol=1e-12, rtol=1e-12).y
additional_propagation = np.vstack((additional_propagation, np.full(np.shape(additional_propagation), 1e-12)))
truth_vals = np.concatenate((back_propagation[:, :-1], forward_propagation, additional_propagation[:, 1:]), axis=1)
time_vals = np.concatenate((np.flip(backprop_time_vals[1:]), forprop_time_vals, additional_time_vals[1:])) + abs(backprop_time_vals[-1])

final_time = 25*24 / NONDIM_TIME_HR
final_time *= 10

def stop_condition(t, y, mu):
    return y[1]

stop_condition.direction = True
stop_condition.terminal = 2

for sensor_index in range(3):
    sensor_position_vals = generate_sensor_positions(sensor_dynamics_equation, sensor_initial_conditions, (mu,), time_vals, 0., sensor_period)

ax = plt.figure(layout="constrained").add_subplot()
for sensor_index in range(3):
    ax.plot(sensor_position_vals[sensor_index*3, :], sensor_position_vals[1+sensor_index*3, :], alpha=0.5)
    ax.scatter(sensor_position_vals[sensor_index*3, 0], sensor_position_vals[1+sensor_index*3, 0])
ax.set_aspect("equal")

plt.show(block=False)
plt.pause(0.001)
input("press [enter] to continue")
plt.close("all")