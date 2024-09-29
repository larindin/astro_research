
import numpy as np
from measurement_functions import *
from CR3BP_pontryagin import *

# Truth parameters
# initial_truth = np.array([5.700765369968086027e-01, 0, 0, 0, 1.001309137115701908e+00, 0, 1.725410447851861151e-01, 1.995473951467236374e-03, 0, 2.043043419011393320e-03, 7.594073607095458422e-02, 0])
initial_truth = np.array([1.023860, 0, -0.183349, 0, -0.107237, 0, 1.005271672718543563e+00, -4.867201096742823641e-01, 1.039996619651046172e+00, 6.123775200273015029e-01, 5.309094110569010372e-01, 5.410185901250754714e-01])
# initial_truth = np.array([0.869093134528914, 0, 0, 0, 0.471129523484998, 0, 2.728393465768669479e+00,-2.213062564132921128e+00, -1.368370228190690652e+00, 9.316407006623674825e-01, -2.255988376284393804e-01, 3.620024666599501173e-01])
# final_time = 6.185934071545815982
final_time = 2.199798443376498547
# final_time = 0.2
# final_time = 2.346224101773999760
dt = 30*60/3.751903e5
dt = 0.01
dynamics_equation = minimum_energy_ODE
umax = 1
mu = 1.215059e-2

# Measurement parameters
measurement_equation = az_el_sensor
measurement_jacobian = az_el_sensor_jacobian
measurement_noise_covariance = np.eye(2)*np.deg2rad(18/60/60)**2
measurement_dimension = 2
individual_measurement_size = 2
sensor_initial_conditions = np.array([[9.251388163276373922e-01,0,2.188093146262887201e-01,0,1.215781574069972060e-01,0],
                                      [1.082273752962558566e+00,0,-2.023390302053850731e-01,0,-2.003138750012137537e-01,0]])
sensor_dynamics_equation = CR3BP_DEs
earth_exclusion_angle = np.deg2rad(5)
moon_additional_angle = np.deg2rad(5)
sun_exclusion_angle = np.deg2rad(20)
seed = 0

# Filter parameters
initial_covariance = np.vstack((np.hstack((np.eye(6)*0.03**2, np.zeros((6, 6)))), np.hstack((np.zeros((6, 6)), np.eye(6)*1**2))))
initial_estimate = np.concatenate((np.random.default_rng(seed).multivariate_normal(initial_truth[0:6], initial_covariance[0:6, 0:6]), np.array([0, 0, 0, 0, 0, 0])))
EKF_process_noise_covriance = np.vstack((np.hstack((np.eye(3)*1**2, np.zeros((3, 3)))), np.hstack((np.zeros((3, 3)), np.eye(3)*1**2))))
state_process_noise_covariance = np.vstack((np.hstack((np.eye(3)*0.02**2, np.zeros((3, 3)))), np.hstack((np.zeros((3, 3)), np.eye(3)*0.02**2))))
costate_process_noise_covariance = np.vstack((np.hstack((np.eye(3)*0.01**2, np.zeros((3, 3)))), np.hstack((np.zeros((3, 3)), np.eye(3)*0.01**2))))
process_noise_covariance = np.vstack((np.hstack((state_process_noise_covariance, np.zeros((6, 6)))), np.hstack((np.zeros((6, 6)), costate_process_noise_covariance))))
filter_measurement_covariance = measurement_noise_covariance * (1.1)**2