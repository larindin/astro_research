
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from configuration_costate import *
from CR3BP import *
from CR3BP_pontryagin import *
from EKF import *
from measurement_functions import *
from plotting import *

time_vals = np.arange(0, final_time, 0.001)
tspan = np.array([time_vals[0], time_vals[-1]])
truth_propagation = scipy.integrate.solve_ivp(dynamics_equation, tspan, initial_truth, args=(mu, umax), t_eval=time_vals, atol=1e-12, rtol=1e-12)
truth_vals = truth_propagation.y

measurements = generate_measurements(time_vals, truth_vals, measurement_equation, measurement_dimension, measurement_noise_covariance, (mu,), seed)

dynamics_args = (mu, umax)
measurement_args = (mu,)

def EKF_dynamics_equation(t, X, mu, umax, process_noise_covariance):

    state = X[0:6]
    costate = X[6:12]
    covariance = X[12:156].reshape((12, 12))

    jacobian = minimum_energy_jacobian(state, costate, mu, umax)

    ddt_state = minimum_energy_ODE(0, X[0:12], mu, umax)
    ddt_covariance = jacobian @ covariance + covariance @ jacobian.T + process_noise_covariance

    return np.concatenate((ddt_state, ddt_covariance.flatten()))
    
def EKF_measurement_equation(X, mu):

    measurement = az_el_moon(X, mu)
    measurement_jacobian = az_el_moon_jacobian_costate(X, mu)

    return measurement, measurement_jacobian


filter_output = run_EKF(initial_estimate, initial_covariance,
                        EKF_dynamics_equation, EKF_measurement_equation,
                        measurements, process_noise_covariance,
                        measurement_noise_covariance, 
                        dynamics_args, measurement_args)

filter_time = filter_output.t
posterior_estimate_vals = filter_output.posterior_estimate_vals
posterior_covariance_vals = filter_output.posterior_covariance_vals
anterior_estimate_vals = filter_output.anterior_estimate_vals
anterior_covariance_vals = filter_output.anterior_covariance_vals
innovations = filter_output.innovations_vals

ax = plt.figure().add_subplot(projection="3d")
ax.plot(truth_vals[0], truth_vals[1], truth_vals[2])
ax.plot(posterior_estimate_vals[0], posterior_estimate_vals[1], posterior_estimate_vals[2])
ax.set_aspect("equal")

ax = plt.figure().add_subplot()
ax.plot(measurements.t, measurements.measurements[0])
ax.plot(measurements.t, measurements.measurements[1])

ax = plt.figure().add_subplot()
ax.plot(measurements.t, innovations[0])
ax.plot(measurements.t, innovations[1])

ax = plt.figure().add_subplot()
ax.plot(measurements.t, anterior_estimate_vals[0])
ax.plot(measurements.t, anterior_estimate_vals[1])
ax.plot(measurements.t, anterior_estimate_vals[2])

posterior_estimates = [posterior_estimate_vals]
posterior_covariances = [posterior_covariance_vals]

estimation_errors = compute_estimation_errors(truth_vals, posterior_estimates, 12)
three_sigmas = compute_3sigmas(posterior_covariances, 12)
plot_3sigma(time_vals, estimation_errors, three_sigmas, 1, 1, 6)
plot_3sigma_costate(time_vals, estimation_errors, three_sigmas, 1, 1, 6)

plt.show()