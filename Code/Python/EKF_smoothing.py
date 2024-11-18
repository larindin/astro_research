

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