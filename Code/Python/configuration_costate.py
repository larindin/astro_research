
import numpy as np
from measurement_functions import *
from CR3BP_pontryagin import *

# Truth parameters
initial_truth = np.array([5.700765369968086027e-01, 0, 0, 0, 1.001309137115701908e+00, 0, 1.725410447851861151e-01, 1.995473951467236374e-03, 0, 2.043043419011393320e-03, 7.594073607095458422e-02, 0])
# initial_truth = np.array([1.023860, 0, -0.183349, 0, -0.107237, 0, 1.005271672718543563e+00, -4.867201096742823641e-01, 1.039996619651046172e+00, 6.123775200273015029e-01, 5.309094110569010372e-01, 5.410185901250754714e-01])
# initial_truth = np.array([0.869093134528914, 0, 0, 0, 0.471129523484998, 0, 2.728393465768669479e+00,-2.213062564132921128e+00, -1.368370228190690652e+00, 9.316407006623674825e-01, -2.255988376284393804e-01, 3.620024666599501173e-01])
final_time = 6.185934071545815982
# final_time = 2.199798443376498547
# final_time = 2.346224101773999760
dynamics_equation = minimum_energy_ODE
umax = 1
mu = 1.215059e-2

# Measurement parameters
measurement_equation = az_el_moon
measurement_jacobian = az_el_moon_jacobian_costate
measurement_noise_covariance = np.eye(2)*np.deg2rad(0.001)**2
measurement_dimension = 2
seed = 0

# Filter parameters
initial_estimate = np.array([5.700765369968086027e-01, 0, 0, 0, 1.001309137115701908e+00, 0, 0, 0, 0, 0, 0, 0])
# initial_estimate = np.array([1.023860, 0, -0.183349, 0, -0.107237, 0, 0, 0, 0, 0, 0, 0])
# initial_estimate = np.array([8.690931345289143461e-01, 0, 0, 0, 4.711295234849984803e-01, 0, 0, 0, 0, 0, 0, 0])
initial_covariance = np.vstack((np.hstack((np.eye(6)*0.001**2, np.zeros((6, 6)))), np.hstack((np.zeros((6, 6)), np.eye(6)*0.1**2))))
process_noise_covariance = np.vstack((np.hstack((np.eye(6)*0.001**2, np.zeros((6, 6)))), np.hstack((np.zeros((6, 6)), np.eye(6)*0.001**2))))
filter_type = 0