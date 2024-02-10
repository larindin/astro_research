
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from CR3BP import *
from EKF import *
from dynamics_functions import *
from plotting import *


mu = 1.215059e-2

measurement_noise_covariance = np.deg2rad(np.eye(2)*(0.005)**2)
process_noise_covariance = np.zeros((6, 6))
generator_seed = 0

def truth_dynamics_equation(t, X, mu):

    ddtX = CR3BP_DEs(X, mu)

    return ddtX

initial_truth = np.array([0.85, 0, 0.17546505, 0, 0.2628980369, 0])
time_vals = np.arange(0, 2.5543991, 0.005)
tspan = np.array([time_vals[0], time_vals[-1]])
truth_propagation = scipy.integrate.solve_ivp(truth_dynamics_equation, tspan, initial_truth, args=(mu, ), t_eval=time_vals, atol=1e-12, rtol=1e-12)
truth_vals = truth_propagation.y

measurements = generate_measurements(time_vals, truth_vals, azimuth_elevation, 2, measurement_noise_covariance, (mu, ), generator_seed)

initial_estimate = np.array([0.85, 0, 0.17546505, 0, 0.2628980369, 0])
initial_covariance = np.eye(6)*0.025

dynamics_args = (mu, )
measurement_args = (mu, )

def EKF_dynamics_equation(t, X, mu, process_noise_covariance):

    state = X[0:6]
    covariance = X[6:42].reshape((6, 6))

    jacobian = CR3BP_jacobian(state, mu)

    ddt_state = CR3BP_DEs(state, mu)
    ddt_covariance = jacobian @ covariance + covariance @ jacobian.T + process_noise_covariance

    return np.concatenate((ddt_state, ddt_covariance.flatten()))
    
def EKF_measurement_equation(X, mu):

    measurement = azimuth_elevation(X, mu)
    measurement_jacobian = azimuth_elevation_jacobian(X, mu)

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
ax.plot(time_vals, 3*np.sqrt(posterior_covariance_vals[0, 0]))

ax = plt.figure().add_subplot()
ax.plot(measurements.t, np.rad2deg(measurements.measurements[0]))
ax.plot(measurements.t, np.rad2deg(measurements.measurements[1]))

ax = plt.figure().add_subplot()
ax.plot(measurements.t, innovations[0])
ax.plot(measurements.t, innovations[1])

ax = plt.figure().add_subplot()
ax.plot(measurements.t, anterior_estimate_vals[0])
ax.plot(measurements.t, anterior_estimate_vals[1])
ax.plot(measurements.t, anterior_estimate_vals[2])


posterior_estimates = [posterior_estimate_vals]
posterior_covariances = [posterior_covariance_vals]

estimation_errors = compute_estimation_errors(truth_vals, posterior_estimates, 6)
three_sigmas = compute_3sigmas(posterior_covariances, 6)
plot_3sigma(time_vals, estimation_errors, three_sigmas, 1, 1, 6)

plt.show()