
import numpy as np
import scipy.integrate

class Measurements:
    def __init__(self, time_vals, measurement_vals):     
        assert np.size(time_vals, 0) == np.size(measurement_vals, 1)

        self.t = time_vals
        self.measurements = measurement_vals

class Truth:
    def __init__(self, time_vals, truth_vals):
        assert np.size(time_vals, 0) == np.size(truth_vals, 1)

        self.t = time_vals
        self.truth_vals = truth_vals

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

def generate_measurements(time_vals: np.ndarray, truth_vals: np.ndarray, measurement_equation, measurement_size: int, noise_covariance, measurement_args, seed):

    num_measurements = len(time_vals[1:])
    measurement_vals = np.zeros((measurement_size, num_measurements))

    generator = np.random.default_rng(seed)
    noise_mean = np.zeros(measurement_size)
    noise_vals = generator.multivariate_normal(noise_mean, noise_covariance, num_measurements)

    for time_index in np.arange(num_measurements):
        args = (truth_vals[:, time_index],) + measurement_args
        measurement_vals[:, time_index] = measurement_equation(*args) + noise_vals[time_index, :]
    


    return Measurements(time_vals[1:], measurement_vals)