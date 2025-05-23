

import numpy as np
import scipy
import scipy.integrate

# np.set_printoptions(suppress=True, precision=5, linewidth=500)

NONDIM_LENGTH = 3.844e5 # kilometers
NONDIM_TIME = 3.751903e5 # seconds
NONDIM_TIME_HR = 1.042195278e2 # hours
L1 = 0.83691513	# Earth-Moon system
L2 = 1.15568217 # Earth-Moon system
mu = 1.215059e-2 # Earth-Moon system

def generate_sun_vectors(time_vals, phase):

    num_vals = len(time_vals)
    sun_vector_vals = np.zeros((3, num_vals))

    for time_index in range(num_vals):
        time = time_vals[time_index]
        sun_vector_vals[:, time_index] = -np.array([np.cos(time*0.07470278234 + phase), np.sin(time*0.07470278234 + phase), 0])
    
    return sun_vector_vals

def generate_moon_vectors(time_vals, sensor_pos_vals):

    num_vals = len(time_vals)
    moon_vector_vals = np.zeros((3, num_vals))

    for time_index in range(num_vals):
        sensor_pos = sensor_pos_vals[:, time_index]
        moon_vector_vals[:, time_index] = np.array([1 - 1.215059e-2, 0, 0]) - sensor_pos
        moon_vector_vals[:, time_index] /= np.linalg.norm(moon_vector_vals[:, time_index])
    
    return moon_vector_vals

def generate_earth_vectors(time_vals, sensor_pos_vals):

    num_vals = len(time_vals)
    earth_vector_vals = np.zeros((3, num_vals))

    for time_index in range(num_vals):
        sensor_pos = sensor_pos_vals[:, time_index]
        earth_vector_vals[:, time_index] = np.array([-1.215059e-2, 0, 0]) - sensor_pos
        earth_vector_vals[:, time_index] /= np.linalg.norm(earth_vector_vals[:, time_index])
    
    return earth_vector_vals

def generate_sensor_positions(sensor_dynamics_equation, sensor_initial_conditions, args, t_eval, phasing=0, period=0):

    num_sensors = np.size(sensor_initial_conditions, 0)
    sensor_positions = np.empty((num_sensors*3, len(t_eval)))
    tspan = np.array([t_eval[0], t_eval[-1]])

    modified_ICs = np.empty(np.shape(sensor_initial_conditions))
    if period:
        offset = period*phasing
        for sensor_index in range(num_sensors):
            initial_conditions = sensor_initial_conditions[sensor_index, :]
            propagation = scipy.integrate.solve_ivp(sensor_dynamics_equation, [0, offset], initial_conditions, args=args, atol=1e-12, rtol=1e-12)
            modified_ICs[sensor_index] = propagation.y[:, -1]
    else:
        modified_ICs = sensor_initial_conditions

    for sensor_index in range(num_sensors):
        initial_conditions = modified_ICs[sensor_index, :]
        sensor_positions[sensor_index*3:(sensor_index + 1)*3] = scipy.integrate.solve_ivp(sensor_dynamics_equation, tspan, initial_conditions, args=args, t_eval=t_eval, atol=1e-12, rtol=1e-12).y[0:3, :]
    
    return sensor_positions

def enforce_symmetry(covariance_matrix: np.ndarray):
    fixed_matrix = (covariance_matrix + covariance_matrix.T) / 2
    return fixed_matrix
    
def check_innovations(innovations):
    for index, innovation in enumerate(innovations):
        if abs(innovation) > np.pi:
            innovations[index] = -np.sign(innovation)*(2*np.pi - abs(innovation))
    return innovations

def assess_measurement_likelihood(innovations, innovations_covariance):
    return np.sqrt(np.linalg.det(2*np.pi*innovations_covariance)), -0.5 * innovations.T @ np.linalg.inv(innovations_covariance) @ innovations

def trim_zero_weights(posterior_estimate_vals, posterior_covariance_vals, weight_vals):

    num_kernels = np.size(weight_vals, 0)

    for kernel_index in range(num_kernels):
        posterior_estimate_vals[:, weight_vals[kernel_index, :]==0, kernel_index] = np.nan
        posterior_covariance_vals[:, :, weight_vals[kernel_index, :]==0, kernel_index] = np.nan
    
    return posterior_estimate_vals, posterior_covariance_vals

def compute_primer_vectors(lambda_v_vals):
    norms = np.linalg.norm(lambda_v_vals, axis=0)
    p1 = (-lambda_v_vals[0]/norms)[None, :]
    p2 = (-lambda_v_vals[1]/norms)[None, :]
    p3 = (-lambda_v_vals[2]/norms)[None, :]
    return np.vstack((p1, p2, p3))

def get_chi2_cutoff(k, p):
    return scipy.optimize.root_scalar(lambda chi: 1 - scipy.stats.chi2.cdf(chi, k) - p, x0=k).root

def get_thrusting_arc_indices(control_history):

    control_norms = np.linalg.norm(control_history, axis=0)
    control_bool = control_norms > 0.01

    num_timesteps = len(control_bool)
    thrusting_arc_indices = []
    started_flag = 0
    for timestep_index in range(num_timesteps):
        if control_bool[timestep_index] and started_flag == 0:
            start_index = timestep_index
            started_flag = 1
        elif control_bool[timestep_index] == 0 and started_flag == 1:
            end_index = timestep_index
            started_flag = 0
            thrusting_arc_indices.append((start_index, end_index))
        elif timestep_index == num_timesteps - 1 and started_flag == 1:
            end_index = timestep_index
            started_flag = 0
            thrusting_arc_indices.append((start_index, end_index))
    
    return thrusting_arc_indices

def assess_measurement(measurement, individual_measurement_size):

    measurement_size = np.size(measurement, 0)
    num_measurements = int(measurement_size/individual_measurement_size)
    new_measurement = np.array([])
    valid_indices = []

    for checking_index in range(num_measurements):
        
        to_be_checked = measurement[checking_index*individual_measurement_size:(checking_index + 1)*individual_measurement_size]
        
        if not np.array_equal(to_be_checked, np.empty(individual_measurement_size)*np.nan, equal_nan=True):
            new_measurement = np.concatenate((new_measurement, to_be_checked))
            valid_indices.append(checking_index)
    
    return new_measurement, valid_indices