

import numpy as np
import scipy
import matplotlib.pyplot as plt
from CR3BP import *
from CR3BP_pontryagin import *
from configuration_dual_GMIMM import *

def thrusting_dynamics_equation(t, X, mu, umax, rho):

    state = X[0:6]
    costate = X[6:12]
    STM = X[12:156].reshape((12, 12))

    jacobian = minimum_fuel_jacobian(state, costate, mu, umax, rho)

    ddt_state = minimum_fuel_ODE(0, X[0:12], mu, umax, rho)
    ddt_STM = jacobian @ STM

    return np.concatenate((ddt_state, ddt_STM.flatten()))

time_vals = np.arange(0, final_time, dt)
tf = 1
tspan = np.array([0, tf])
t_eval = np.arange(0, tf+0.01, 0.01)
differential = np.zeros(156)
differential[0] = 0.01
differential[6] = 0.01
differential = np.ones(156)*0.01
initial_covariance = np.eye(12)
initial_covariance[0:3, 0:3] *= 1e-12
initial_covariance[3:9, 3:9] *= 1e-6
initial_covariance[6:12, 6:12] *= 1e-4

initial_covariance = np.array([[ 7.66223917e-11, -2.66201175e-12, -2.53008586e-11,  1.04454083e-08,
  -4.56916894e-10, -3.45002794e-09,  1.02336173e+00,  2.26039977e-02,
   1.02314641e+00,  3.27445884e-02,  1.02289213e+00,  4.28690429e-02],
 [-2.66201175e-12,  1.02607727e-12,  8.88729878e-13, -3.61695993e-10,
   1.47057276e-10,  1.21222508e-10,  1.02147924e+00,  8.31462494e-02,
   1.02102576e+00,  9.31460479e-02,  1.02053174e+00,  1.03112500e-01],
 [-2.53008586e-11,  8.88729878e-13,  9.89145008e-12, -3.44718570e-09,
   1.52223785e-10,  1.35451210e-09,  1.01814727e+00,  1.42592354e-01,
   1.01744850e+00,  1.52353260e-01,  1.01670855e+00,  1.62065745e-01],
 [ 1.04454083e-08, -3.61695993e-10, -3.44718570e-09,  5.89138550e-06,
  -1.00690874e-07, -9.73505161e-07,  1.01333681e+00,  2.00386884e-01,
   1.01239107e+00,  2.09824437e-01,  1.01140435e+00,  2.19200933e-01],
 [-4.56916894e-10,  1.47057276e-10,  1.52223785e-10, -1.00690874e-07,
   2.98822565e-06,  3.31818784e-08,  1.00705005e+00,  2.56060359e-01,
   1.00675807e+00,  2.58315293e-01,  8.00721357e-05, -3.15469963e-06],
 [-3.45002794e-09,  1.21222508e-10,  1.35451210e-09, -9.73505161e-07,
   3.31818784e-08,  3.33234395e-06,  1.69038658e-02, -9.99930694e-05,
   8.58415092e-06, -8.96677218e-05, -1.02457036e-05, -5.32895973e-06],
 [ 1.31123216e-04, -1.04524458e+00, -2.61341776e-02, -9.99900038e-03,
  -4.27891267e+00, -2.91598837e-02,  1.00000000e-04,  0.00000000e+00,
   0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
 [ 2.15635064e-04, -1.26583420e-02,  8.90274939e-03,  1.00006061e+00,
   4.72235674e-02,  7.25293442e-04,  0.00000000e+00,  1.00000000e-04,
   0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
 [-4.84971118e-06,  1.00006556e+00,  2.93939228e+00, -2.31137442e-01,
   3.01989702e-02,  1.49285158e-02,  0.00000000e+00,  0.00000000e+00,
   1.00000000e-04,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
 [-2.31511071e+02,  4.51495359e-02, -1.00004969e+00,  1.61262444e+00,
  -1.16960227e-08,  2.56067754e-04,  0.00000000e+00,  0.00000000e+00,
   0.00000000e+00,  1.00000000e-04,  0.00000000e+00,  0.00000000e+00],
 [ 1.00004741e+00,  9.13822745e-04,  1.04786737e-05, -7.76685272e-03,
   1.49502489e-02,  1.34326210e-04,  0.00000000e+00,  0.00000000e+00,
   0.00000000e+00,  0.00000000e+00,  1.00000000e-04,  0.00000000e+00],
 [-4.79418570e-07,  2.99240637e-05, -3.33773310e-01,  4.99231440e-07,
   4.51775628e-02,  7.41358490e-04,  0.00000000e+00,  0.00000000e+00,
   0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e-04]])

initial_conditions = np.concatenate((initial_truth, np.eye(12).flatten()))
differential_conditions = initial_conditions + differential

truth_propagation = scipy.integrate.solve_ivp(thrusting_dynamics_equation, tspan, initial_conditions, t_eval=t_eval, args=truth_dynamics_args, atol=1e-12, rtol=1e-12)
differential_propagation = scipy.integrate.solve_ivp(thrusting_dynamics_equation, tspan, differential_conditions, t_eval=t_eval, args=truth_dynamics_args, atol=1e-12, rtol=1e-12)
t = truth_propagation.t
truth_vals = truth_propagation.y
differential_vals = differential_propagation.y


num_vals = len(t)
covariance_vals = np.empty((12, 12, num_vals))
estimated_vals = np.empty((12, num_vals))
for time_index in np.arange(num_vals):
    STM = np.reshape(truth_vals[12:, time_index], (12, 12))
    estimated_vals[:, time_index] = truth_vals[0:12, time_index] + STM @ differential[0:12]
    covariance_vals[:, :, time_index] = STM @ initial_covariance @ STM.T

for time_index in np.arange(num_vals):
    print(np.diag(covariance_vals[:, :, time_index]))

ax = plt.figure().add_subplot(projection="3d")
ax.scatter(truth_vals[0], truth_vals[1], truth_vals[2], alpha=0.5)
ax.scatter(differential_vals[0], differential_vals[1], differential_vals[2], alpha=0.5)
ax.scatter(estimated_vals[0], estimated_vals[1], estimated_vals[2], alpha=0.5)
ax.set_aspect("equal")

# ax = plt.figure().add_subplot()
# ax.plot(t, trace_vals)

plt.show()