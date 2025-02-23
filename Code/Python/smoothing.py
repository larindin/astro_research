

import numpy as np
import scipy.integrate
from EKF import *
from helper_functions import *

def run_EKF_smoothing(results):

    t = results.t
    anterior_estimate_vals = results.anterior_estimate_vals
    posterior_estimate_vals = results.posterior_estimate_vals
    anterior_covariance_vals = results.anterior_covariance_vals
    posterior_covariance_vals = results.posterior_covariance_vals
    STM_vals = results.STM_vals
    innovations_vals = results.innovations_vals

    num_vals = len(t)
    state_size = np.size(posterior_estimate_vals, 0)

    smoothed_estimate_vals = np.empty((state_size, num_vals))
    smoothed_covariance_vals = np.empty((state_size, state_size, num_vals))

    smoothed_estimate_vals[:, -1] = posterior_estimate_vals[:, -1]

    for val_index in np.arange(num_vals-2, 0, -1):

        STM = STM_vals[:, :, val_index+1]

        S = posterior_covariance_vals[:, :, val_index] @ STM.T @ np.linalg.inv(anterior_covariance_vals[:, :, val_index+1])
        smoothed_estimate_vals[:, val_index] = posterior_estimate_vals[:, val_index] + S @ (smoothed_estimate_vals[:, val_index+1] - STM @ posterior_estimate_vals[:, val_index])
        
    return smoothed_estimate_vals

def run_OCBE_smoothing(results, control_noise_covariance):

    t = results.t
    posterior_estimate_vals = results.posterior_estimate_vals
    anterior_covariance_vals = results.anterior_covariance_vals
    posterior_covariance_vals = results.posterior_covariance_vals
    STM_vals = results.STM_vals
    costate_vals = results.costate_vals
    control_vals = results.control_vals

    num_vals = len(t)
    state_size = np.size(posterior_estimate_vals, 0)
    B = np.vstack((np.zeros((3, 3)), np.eye(3)))

    smoothed_estimate_vals = np.empty((state_size, num_vals))
    smoothed_covariance_vals = np.empty((state_size, state_size, num_vals))
    smoothed_costate_vals = np.empty((state_size, num_vals))
    smoothed_control_vals = np.empty((3, num_vals))

    smoothed_estimate_vals[:, -1] = posterior_estimate_vals[:, -1]
    smoothed_covariance_vals[:, :, -1] = posterior_covariance_vals[:, :, -1]
    smoothed_costate_vals[:, -1] = costate_vals[:, -1]
    smoothed_control_vals[:, -1] = control_vals[:, -1]

    for val_index in np.arange(num_vals-2, 0, -1):

        STM_xx = STM_vals[:, :, val_index+1][0:state_size, 0:state_size]
        STM_ll = STM_vals[:, :, val_index][state_size:2*state_size, state_size:2*state_size]

        S = posterior_covariance_vals[:, :, val_index] @ STM_xx.T @ np.linalg.inv(anterior_covariance_vals[:, :, val_index+1])
        smoothed_estimate_vals[:, val_index] = posterior_estimate_vals[:, val_index] + S @ (smoothed_estimate_vals[:, val_index+1] - STM_xx @ posterior_estimate_vals[:, val_index])
        smoothed_covariance_vals[:, :, val_index] = S @ (smoothed_covariance_vals[:, :, val_index+1] - anterior_covariance_vals[:, :, val_index+1]) @ S.T
        smoothed_costate_vals[:, val_index] = -np.linalg.inv(posterior_covariance_vals[:, :, val_index]) @ S @ (smoothed_estimate_vals[:, val_index+1] - STM_xx @ posterior_estimate_vals[:, val_index])
        smoothed_control_vals[:, val_index] = -control_noise_covariance @ B.T @ STM_ll @ np.linalg.inv(posterior_covariance_vals[:, :, val_index]) @ (posterior_estimate_vals[:, val_index] - smoothed_estimate_vals[:, val_index])

    return smoothed_estimate_vals, smoothed_covariance_vals, smoothed_costate_vals, smoothed_control_vals
    