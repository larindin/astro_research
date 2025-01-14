
import numpy as np
from measurement_functions import *
from CR3BP_pontryagin import *
from catalogue import *
from helper_functions import *

# Truth parameters
initial_orbit_index = 1
final_orbit_index = 3
initial_state = boundary_states[initial_orbit_index][0:6]
initial_costate = costates[initial_orbit_index][final_orbit_index]
initial_truth = np.concatenate((initial_state, initial_costate))
final_time = 25*24 / NONDIM_TIME_HR
# dt = 30*60/NONDIM_TIME
dt = 0.01
dynamics_equation = minimum_fuel_ODE
# initial_truth[6:12] = np.array([0, 0, 0, 1e-12, 1e-12, 1e-12])
truth_rho = 1e-4
mass = 1500 # kg
thrust = 0.6 # N
umax = thrust/mass / 1000 / NONDIM_LENGTH * NONDIM_TIME**2
mu = 1.215059e-2
truth_dynamics_args = (mu, umax, truth_rho)

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
initial_state_covariance =  scipy.linalg.block_diag(np.eye(3)*1.30072841e-4**2, np.eye(3)*9.76041363e-4**2)
initial_covariance = scipy.linalg.block_diag(initial_state_covariance, np.eye(6)*1**2)
generator = np.random.default_rng(seed)
initial_estimate = np.concatenate((generator.multivariate_normal(initial_truth[0:6], initial_state_covariance), np.zeros(3), np.ones(3)*1e-9))
# initial_estimate = initial_truth
filter_measurement_covariance = measurement_noise_covariance * (1.1)**2
filter_rho = 1e-4
switching_cutoff = 5

# IMM parameters
coasting_process_noise_covariance = scipy.linalg.block_diag(np.eye(3)*(1e-6)**2, np.eye(3)*(1e-3)**2, np.eye(6)*(1e-12)**2)
thrusting_process_noise_covariance = scipy.linalg.block_diag(np.eye(3)*(1e-6)**2, np.eye(3)*(1e-3)**2, np.eye(6)*(0.5)**2)
process_noise_covariances = np.stack((coasting_process_noise_covariance, thrusting_process_noise_covariance), 2)
initial_mode_probabilities = np.array([0.8, 0.2])
mode_transition_matrix = np.array([[0.8, 0.2], [0.2, 0.8]])

# GMM parameters
initial_kernel_state_covariance = scipy.linalg.block_diag(np.eye(3)*1e-6**2, np.eye(3)*1e-3**2)
initial_kernel_costate_covariance = np.eye(6)*0.01**2
initial_kernel_covariance = scipy.linalg.block_diag(initial_kernel_state_covariance, initial_kernel_costate_covariance)
kernel_process_noise = scipy.linalg.block_diag(np.eye(3)*(1e-6)**2, np.eye(3)*(1e-3)**2, np.eye(3)*(1e-2)**2, np.eye(3)*(3e-2)**2)
# magnitudes = np.linspace(1.15, 1.25, 11)
magnitudes = np.linspace(1.01, 1.05, 2)
magnitudes = [np.linalg.norm(initial_truth[9:12])]
num_kernels = len(magnitudes)
initial_weights = np.ones(num_kernels)/num_kernels
