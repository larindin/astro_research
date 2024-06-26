
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from CR3BP import *
from CR3BP_pontryagin import *
from EKF import *
from dynamics_functions import *
from plotting import *


mu = 1.215059e-2
umax = 1

measurement_noise_covariance = np.eye(3)*(0.001)**2
process_noise_covariance = np.eye(6)*(0.01)**2
generator_seed = 0

def truth_dynamics_equation(t, X, mu, umax):

    ddtX = minimum_energy_ODE(t, X, mu, umax)

    return ddtX

initial_truth = np.array([0.869093134528914, 0, 0, 0, 0.471129523484998, 0, 2.728393465768669479e+00,-2.213062564132921128e+00, -1.368370228190690652e+00, 9.316407006623674825e-01, -2.255988376284393804e-01, 3.620024666599501173e-01])
time_vals = np.arange(0, 2.346224101773999760e+00-0.05, 0.002)
tspan = np.array([time_vals[0], time_vals[-1]])
truth_propagation = scipy.integrate.solve_ivp(truth_dynamics_equation, tspan, initial_truth, args=(mu, umax), t_eval=time_vals, atol=1e-12, rtol=1e-12)
truth_vals = truth_propagation.y

measurements = generate_measurements(time_vals, truth_vals, cartesian_costate, 3, measurement_noise_covariance, (), generator_seed)

initial_estimate = np.array([0.869093134528914, 0, 0, 0, 0.471129523484998, 0])
initial_covariance = np.eye(6)*0.001**2

dynamics_args = (mu, )
measurement_args = ()

def EKF_dynamics_equation(t, X, mu, process_noise_covariance):

    state = X[0:6]
    covariance = X[6:42].reshape((6, 6))

    jacobian = CR3BP_jacobian(state, mu)

    ddt_state = CR3BP_DEs(state, mu)
    ddt_covariance = jacobian @ covariance + covariance @ jacobian.T + process_noise_covariance

    return np.concatenate((ddt_state, ddt_covariance.flatten()))
    
def EKF_measurement_equation(X):

    measurement = cartesian(X)
    measurement_jacobian = cartesian_jacobian(X)

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
ax.plot(measurements.t, anterior_estimate_vals[0])
ax.plot(measurements.t, anterior_estimate_vals[1])
ax.plot(measurements.t, anterior_estimate_vals[2])

posterior_estimates = [posterior_estimate_vals]
posterior_covariances = [posterior_covariance_vals]

estimation_errors = compute_estimation_errors(truth_vals, posterior_estimates, 6)
three_sigmas = compute_3sigmas(posterior_covariances, 6)
plot_3sigma(time_vals, estimation_errors, three_sigmas, 1, 1, 6)

plt.show()