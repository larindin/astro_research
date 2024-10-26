
import numpy as np
from measurement_functions import *
from CR3BP_pontryagin import *

# Truth parameters
# initial_truth = np.array([5.700765369968086027e-01, 0, 0, 0, 1.001309137115701908e+00, 0, 2.218985298179385168e+00, 5.173659581424284309e-03, 0, 1.045748103035356626e-02, 1.002278884441901052e+00, 0])
# initial_truth = np.array([1.023860, 0, -0.183349, 0, -0.107237, 0, 1.886804244746727921e+00, -5.756815826039763939e-01, 4.471842695737356377e-01, 8.068657163041901281e-01, 7.386444568749661599e-01, 5.395507048947811857e-01])
initial_truth = np.array([0.869093134528914, 0, 0, 0, 0.471129523484998, 0, 1.886994686356553652e+00, -1.834644038828200818e+00, -1.353616598123221770e+00, 6.170104458498792965e-01, 1.853008581517654740e-01, 7.634714087832630280e-01])
# final_time = 6.235577707735169284
# final_time = 2.209568031669125077
final_time = 1.992457188219079134
final_time = 0.2
dt = 30*60/3.751903e5
dt = 0.01
dynamics_equation = minimum_fuel_ODE
truth_rho = 1e-4
umax = 1
mu = 1.215059e-2

# Measurement parameters
measurement_equation = az_el_sensor
measurement_jacobian = az_el_sensor_jacobian
measurement_noise_covariance = np.eye(2)*np.deg2rad(1e-3)**2
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
initial_state_covariance =  np.vstack((np.hstack((np.eye(3)*1.30072841e-4**2, np.zeros((3, 3)))), np.hstack((np.zeros((3, 3)), np.eye(3)*9.76041363e-4**2))))
initial_covariance = np.vstack((np.hstack((initial_state_covariance, np.zeros((6, 6)))), np.hstack((np.zeros((6, 6)), np.eye(6)*1**2))))
initial_kernel_covariance = np.vstack((np.hstack((initial_state_covariance, np.zeros((6, 6)))), np.hstack((np.zeros((6, 6)), np.eye(6)*0.2**2))))
initial_estimate = np.concatenate((np.random.default_rng(seed).multivariate_normal(initial_truth[0:6], initial_covariance[0:6, 0:6]), np.array([0, 0, 0, 1e-5, 1e-5, 1e-5])))
# initial_estimate = initial_truth
EKF_process_noise_covriance = np.vstack((np.hstack((np.eye(3)*0.1**2, np.zeros((3, 3)))), np.hstack((np.zeros((3, 3)), np.eye(3)*0.1**2))))
state_process_noise_covariance = np.vstack((np.hstack((np.eye(3)*0.001**2, np.zeros((3, 3)))), np.hstack((np.zeros((3, 3)), np.eye(3)*0.001**2))))
costate_process_noise_covariance = np.vstack((np.hstack((np.eye(3)*0.001**2, np.zeros((3, 3)))), np.hstack((np.zeros((3, 3)), np.eye(3)*0.001**2))))
process_noise_covariance = np.vstack((np.hstack((state_process_noise_covariance, np.zeros((6, 6)))), np.hstack((np.zeros((6, 6)), costate_process_noise_covariance))))
filter_measurement_covariance = measurement_noise_covariance * (1.1)**2
filter_rho = 1e-4
timeout_count = 25
switching_count = 2
filter_index = 0
# durations = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
# durations = np.array([0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24])
durations = np.array([0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44])
# magnitudes = np.linspace(1.2, 1.25, 6)
magnitudes = np.linspace(1, 1.05, 6)
num_kernels = len(durations)*len(magnitudes)
initial_weights = np.ones(num_kernels)/num_kernels