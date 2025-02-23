
import numpy as np
import scipy
from typing import *

class Measurements:
    def __init__(self, time_vals, measurement_vals, individual_measurement_size):     
        assert np.size(time_vals, 0) == np.size(measurement_vals, 1)

        self.t = time_vals
        self.measurements = measurement_vals
        self.individual_measurement_size = individual_measurement_size

def azimuth_elevation(X, mu):

    x, y, z = X[0:3]

    azimuth = np.arctan2(y, (x + mu))
    elevation = np.arctan2(z, np.sqrt((x + mu)**2 + y**2))

    return azimuth, elevation

def azimuth_elevation_jacobian(X, mu):

    x, y, z = X[0:3]

    jacobian = np.zeros((2, 6))

    jacobian[0, 0] = 1 / (1 + (y / (x + mu))**2) * (-y / (x + mu)**2)
    jacobian[0, 1] = 1 / (1 + (y / (x + mu))**2) * (1 / (x + mu))
    jacobian[0, 2] = 0
    jacobian[1, 0] = 1 / (1 + (z**2 / ((x + mu)**2 + y**2))) * -(x + mu)*z / ((x + mu)**2 + y**2)**(3/2)
    jacobian[1, 1] = 1 / (1 + (z**2 / ((x + mu)**2 + y**2))) * -y*z / ((x + mu)**2 + y**2)**(3/2)
    jacobian[1, 2] = 1 / (1 + (z**2 / ((x + mu)**2 + y**2))) * 1 / np.sqrt((x + mu)**2 + y**2)

    return jacobian

def azimuth_elevation_jacobian_costate(X, mu):

    x, y, z = X[0:3]

    jacobian = np.zeros((2, 12))

    jacobian[0, 0] = 1 / (1 + (y / (x + mu))**2) * (-y / (x + mu)**2)
    jacobian[0, 1] = 1 / (1 + (y / (x + mu))**2) * (1 / (x + mu))
    jacobian[0, 2] = 0
    jacobian[1, 0] = 1 / (1 + (z**2 / ((x + mu)**2 + y**2))) * -(x + mu)*z / ((x + mu)**2 + y**2)**(3/2)
    jacobian[1, 1] = 1 / (1 + (z**2 / ((x + mu)**2 + y**2))) * -y*z / ((x + mu)**2 + y**2)**(3/2)
    jacobian[1, 2] = 1 / (1 + (z**2 / ((x + mu)**2 + y**2))) * 1 / np.sqrt((x + mu)**2 + y**2)

    return jacobian

def az_el_moon(X, mu):

    x, y, z = X[0:3]

    azimuth = np.arctan2(y, (x + mu - 1))
    elevation = np.arctan2(z, np.sqrt((x + mu - 1)**2 + y**2))

    return azimuth, elevation

def az_el_moon_jacobian(X, mu):

    x, y, z = X[0:3]

    jacobian = np.zeros((2, 6))

    jacobian[0, 0] = 1 / (1 + (y / (x + mu - 1))**2) * (-y / (x + mu - 1)**2)
    jacobian[0, 1] = 1 / (1 + (y / (x + mu - 1))**2) * (1 / (x + mu - 1))
    jacobian[0, 2] = 0
    jacobian[1, 0] = 1 / (1 + (z**2 / ((x + mu - 1)**2 + y**2))) * -(x + mu - 1)*z / ((x + mu - 1)**2 + y**2)**(3/2)
    jacobian[1, 1] = 1 / (1 + (z**2 / ((x + mu - 1)**2 + y**2))) * -y*z / ((x + mu - 1)**2 + y**2)**(3/2)
    jacobian[1, 2] = 1 / (1 + (z**2 / ((x + mu - 1)**2 + y**2))) * 1 / np.sqrt((x + mu - 1)**2 + y**2)

    return jacobian

def az_el_moon_jacobian_costate(X, mu):

    x, y, z = X[0:3]

    jacobian = np.zeros((2, 12))

    jacobian[0, 0] = 1 / (1 + (y / (x + mu - 1))**2) * (-y / (x + mu - 1)**2)
    jacobian[0, 1] = 1 / (1 + (y / (x + mu - 1))**2) * (1 / (x + mu - 1))
    jacobian[0, 2] = 0
    jacobian[1, 0] = 1 / (1 + (z**2 / ((x + mu - 1)**2 + y**2))) * -(x + mu - 1)*z / ((x + mu - 1)**2 + y**2)**(3/2)
    jacobian[1, 1] = 1 / (1 + (z**2 / ((x + mu - 1)**2 + y**2))) * -y*z / ((x + mu - 1)**2 + y**2)**(3/2)
    jacobian[1, 2] = 1 / (1 + (z**2 / ((x + mu - 1)**2 + y**2))) * 1 / np.sqrt((x + mu - 1)**2 + y**2)

    return jacobian

def az_el_sensor(X, sensor_pos):

    x, y, z = X[0:3]
    sensor_x, sensor_y, sensor_z = sensor_pos

    azimuth = np.arctan2(y - sensor_y, (x - sensor_x))
    elevation = np.arctan2(z - sensor_z, np.sqrt((x - sensor_x)**2 + (y - sensor_y)**2))

    return azimuth, elevation

def az_el_sensor_jacobian(X, sensor_pos):

    x, y, z = X[0:3]
    sensor_x, sensor_y, sensor_z = sensor_pos
    x_diff = x - sensor_x
    y_diff = y - sensor_y
    z_diff = z - sensor_z

    jacobian = np.zeros((2, 6))
    jacobian[0, 0] = 1 / (1 + ((y_diff) / (x_diff))**2) * (-(y_diff) / (x_diff)**2)
    jacobian[0, 1] = 1 / (1 + ((y_diff) / (x_diff))**2) * (1 / (x_diff))
    jacobian[1, 0] = 1 / (1 + ((z_diff)**2 / ((x_diff)**2 + (y_diff)**2))) * -(x_diff)*(z_diff) / ((x_diff)**2 + (y_diff)**2)**(3/2)
    jacobian[1, 1] = 1 / (1 + ((z_diff)**2 / ((x_diff)**2 + (y_diff)**2))) * -(y_diff)*(z_diff) / ((x_diff)**2 + (y_diff)**2)**(3/2)
    jacobian[1, 2] = 1 / (1 + ((z_diff)**2 / ((x_diff)**2 + (y_diff)**2))) * 1 / np.sqrt((x_diff)**2 + (y_diff)**2)

    return jacobian

def az_el_sensor_jacobian_costate(X, sensor_pos):

    x, y, z = X[0:3]
    sensor_x, sensor_y, sensor_z = sensor_pos
    x_diff = x - sensor_x
    y_diff = y - sensor_y
    z_diff = z - sensor_z

    jacobian = np.zeros((2, 12))
    jacobian[0, 0] = 1 / (1 + ((y_diff) / (x_diff))**2) * (-(y_diff) / (x_diff)**2)
    jacobian[0, 1] = 1 / (1 + ((y_diff) / (x_diff))**2) * (1 / (x_diff))
    jacobian[1, 0] = 1 / (1 + ((z_diff)**2 / ((x_diff)**2 + (y_diff)**2))) * -(x_diff)*(z_diff) / ((x_diff)**2 + (y_diff)**2)**(3/2)
    jacobian[1, 1] = 1 / (1 + ((z_diff)**2 / ((x_diff)**2 + (y_diff)**2))) * -(y_diff)*(z_diff) / ((x_diff)**2 + (y_diff)**2)**(3/2)
    jacobian[1, 2] = 1 / (1 + ((z_diff)**2 / ((x_diff)**2 + (y_diff)**2))) * 1 / np.sqrt((x_diff)**2 + (y_diff)**2)

    return jacobian

def cartesian_costate(X):

    B = np.hstack((np.eye(3), np.zeros((3, 9))))

    x, y, z = B @ X

    return x, y, z

def cartesian_jacobian_costate(X):

    jacobian = np.hstack((np.eye(3), np.zeros((3, 9))))

    return jacobian

def cartesian(X):

    B = np.hstack((np.eye(3), np.zeros((3, 3))))

    x, y, z = B @ X
    
    return x, y, z

def cartesian_jacobian(X):

    B = np.hstack((np.eye(3), np.zeros((3, 3))))

    return B

def check_exclusion(time, truth, sensor_pos, exclusion_vector, exclusion_angle):

    measurement_vector = truth[0:3] - sensor_pos

    measurement_unit_vector = measurement_vector / np.linalg.norm(measurement_vector)
    exclusion_unit_vector = exclusion_vector / np.linalg.norm(exclusion_vector)

    conjunction_angle = np.arccos(np.dot(measurement_unit_vector, exclusion_unit_vector))

    return (conjunction_angle > exclusion_angle)

def check_exclusion_dynamic(time, target_pos, sensor_pos, exclusion_vector, diameter, additional_exclusion):

    measurement_vector = target_pos - sensor_pos
    distance = np.linalg.norm(measurement_vector)

    measurement_unit_vector = measurement_vector / distance
    exclusion_unit_vector = exclusion_vector / np.linalg.norm(exclusion_vector)

    conjunction_angle = np.arccos(np.dot(measurement_unit_vector, exclusion_unit_vector))
    exclusion_angle = 2*np.arcsin(diameter/2/distance) + additional_exclusion

    return (conjunction_angle > exclusion_angle)

def check_brightness(time, target_pos, sensor_pos, sun_vector, reflectivity, object_radius, ):

    object_flux = 1.79161566e9

    measurement_vector = target_pos - sensor_pos
    measurement_distance = np.linalg.norm(measurement_vector)

    object_radius /= 3.844e5

    sensor_flux = (object_radius / measurement_distance)**2 * object_flux

    sun_vector = 385.17 * sun_vector  - 0
    moon_vector = np.array([1 - 1.215059e-2, 0, 0]) - sensor_pos


def check_validity(time_vals: np.ndarray, target_pos_vals: np.ndarray, sensor_pos_vals: np.ndarray, exclusion_vector_vals, check_function: Callable, check_parameters: tuple):
    
    num_checks = len(time_vals)
    check_results = np.zeros(num_checks)

    for time_index in range(num_checks):

        time = time_vals[time_index]
        target_pos = target_pos_vals[:, time_index]
        sensor_pos = sensor_pos_vals[:, time_index]
        exclusion_vector = exclusion_vector_vals[:, time_index]
        check_results[time_index] = check_function(time, target_pos, sensor_pos, exclusion_vector, *check_parameters)

    return check_results

def generate_measurements(time_vals: np.ndarray, truth_vals: np.ndarray, measurement_equation, measurement_size: int, noise_covariance, measurement_args, seed):

    num_measurements = len(time_vals)
    measurement_vals = np.zeros((measurement_size, num_measurements))

    generator = np.random.default_rng(seed)
    noise_mean = np.zeros(measurement_size)
    noise_vals = generator.multivariate_normal(noise_mean, noise_covariance, num_measurements)

    for time_index in range(num_measurements):
        args = (time_index, truth_vals[:, time_index],) + measurement_args
        measurement_vals[:, time_index] = measurement_equation(*args) + noise_vals[time_index, :]
    
    return Measurements(time_vals, measurement_vals)

def generate_sensor_measurements(time_vals, truth_vals, measurement_equation, individual_measurement_size, noise_covariance, sensor_position_vals, check_results, seed):

    num_measurements = len(time_vals)
    num_sensors = int(np.size(sensor_position_vals, 0)/3)
    measurement_vals = np.zeros((individual_measurement_size*num_sensors, num_measurements))

    for sensor_index in range(num_sensors):
        
        sensor_positions = sensor_position_vals[sensor_index*3:(sensor_index + 1)*3]
        sensor_check_results = check_results[sensor_index, :]
        sensor_measurements = np.empty((individual_measurement_size, num_measurements))
        
        for measurement_index in range(0, num_measurements):
            
            if sensor_check_results[measurement_index] == 0:
                sensor_measurements[:, measurement_index] = np.nan
            else:
                truth = truth_vals[:, measurement_index]
                sensor_position = sensor_positions[:, measurement_index]
                sensor_measurements[:, measurement_index] = measurement_equation(truth, sensor_position)
        
        measurement_vals[sensor_index*individual_measurement_size:(sensor_index + 1)*individual_measurement_size, :] = sensor_measurements

    generator = np.random.default_rng(seed)
    noise_mean = np.zeros(num_sensors*individual_measurement_size)
    noise_covariance = scipy.linalg.block_diag(*(noise_covariance,)*num_sensors)
    
    noise_vals = generator.multivariate_normal(noise_mean, noise_covariance, num_measurements)
    measurement_vals += noise_vals.T

    return Measurements(time_vals, measurement_vals, individual_measurement_size)