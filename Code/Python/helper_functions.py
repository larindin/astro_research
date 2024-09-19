

import numpy as np
import scipy

NONDIM_LENGTH = 3.844e5
NONDIM_TIME = 3.751903e5

def generate_sun_vectors(time_vals, phase):

    num_vals = len(time_vals)
    sun_vector_vals = np.zeros((3, num_vals))

    for time_index in np.arange(num_vals):
        time = time_vals[time_index]
        sun_vector_vals[:, time_index] = -np.array([np.cos(time*0.07470278234 + phase), np.sin(time*0.07470278234 + phase), 0])
    
    return sun_vector_vals

def generate_moon_vectors(time_vals, sensor_pos_vals):

    num_vals = len(time_vals)
    moon_vector_vals = np.zeros((3, num_vals))

    for time_index in np.arange(num_vals):
        sensor_pos = sensor_pos_vals[:, time_index]
        moon_vector_vals[:, time_index] = np.array([1 - 1.215059e-2, 0, 0]) - sensor_pos
        moon_vector_vals[:, time_index] /= np.linalg.norm(moon_vector_vals[:, time_index])
    
    return moon_vector_vals

def generate_earth_vectors(time_vals, sensor_pos_vals):

    num_vals = len(time_vals)
    earth_vector_vals = np.zeros((3, num_vals))

    for time_index in np.arange(num_vals):
        sensor_pos = sensor_pos_vals[:, time_index]
        earth_vector_vals[:, time_index] = np.array([-1.215059e-2, 0, 0]) - sensor_pos
        earth_vector_vals[:, time_index] /= np.linalg.norm(earth_vector_vals[:, time_index])
    
    return earth_vector_vals

def generate_sensor_positions(sensor_dynamics_equation, sensor_initial_conditions, args, t_eval):

    num_sensors = np.size(sensor_initial_conditions, 0)
    sensor_positions = np.empty((num_sensors*3, len(t_eval)))
    tspan = np.array([t_eval[0], t_eval[-1]])
    for sensor_index in np.arange(num_sensors):
        initial_conditions = sensor_initial_conditions[sensor_index, :]
        sensor_positions[sensor_index*3:(sensor_index + 1)*3] = scipy.integrate.solve_ivp(sensor_dynamics_equation, tspan, initial_conditions, args=args, t_eval=t_eval, atol=1e-12, rtol=1e-12).y[0:3, :]
    
    return sensor_positions