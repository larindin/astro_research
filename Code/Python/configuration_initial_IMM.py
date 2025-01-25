
import numpy as np
from measurement_functions import *
from CR3BP_pontryagin import *
from catalogue import *
from helper_functions import *

# Truth parameters
initial_orbit_index = 3
final_orbit_index = 2
initial_state = boundary_states[initial_orbit_index][0:6]
initial_costate = costates[initial_orbit_index][final_orbit_index]
initial_truth = np.concatenate((initial_state, initial_costate))
final_time = 25*24 / NONDIM_TIME_HR
# final_time = 2.5
# final_time = 15*24 / NONDIM_TIME_HR
# final_time = 1
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
initial_covariance = initial_state_covariance
generator = np.random.default_rng(seed)
initial_estimate = generator.multivariate_normal(initial_truth[0:6], initial_state_covariance)
# initial_estimate = initial_truth
filter_measurement_covariance = measurement_noise_covariance * (1.5)**2
coasting_process_noise_covariance = scipy.linalg.block_diag(np.eye(3)*1e-9**2, np.eye(3)*1e-9**2)*dt
maneuvering_process_noise_covariance = scipy.linalg.block_diag(np.eye(3)*1e-9**2, np.eye(3)*5e-3**2)*dt
process_noise_covariances = np.stack((coasting_process_noise_covariance, maneuvering_process_noise_covariance), 2)
initial_mode_probabilities = np.array([0.9, 0.1])
mode_transition_matrix = np.array([[0.95, 0.05],
                                   [0.05, 0.95]])