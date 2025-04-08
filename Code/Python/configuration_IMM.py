
import numpy as np
from measurement_functions import *
from CR3BP_pontryagin import *
from CR3BP_pontryagin_reformulated import *
from catalogue import *
from helper_functions import *

# Monte-carlo parameters
seed = 0
generator = np.random.default_rng(seed)
num_runs = 10
save = False
gap = True

#10, 13, 31, 32 are good

# Truth parameters
initial_orbit_index = 3
final_orbit_index = 1
initial_state = boundary_states[initial_orbit_index][0:6]
initial_costate = costates[initial_orbit_index][final_orbit_index]
initial_truth = np.concatenate((initial_state, initial_costate))
final_time = 25*24 / NONDIM_TIME_HR
# final_time *= 0.8
# final_time = 1
backprop_time = 5 * 24 / NONDIM_TIME_HR
additional_time = 5 * 24 / NONDIM_TIME_HR
dt = 1/NONDIM_TIME_HR
# dt = 0.01
dynamics_equation = minimum_fuel_ODE
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
sensor_initial_conditions1 = np.array([[212171.440, 224189.013, 0, 0.1618*NONDIM_TIME, 0.5780*NONDIM_TIME, 0],
                                      [267266.589, 291378.691, 0, 0.4596*NONDIM_TIME, 0.1991*NONDIM_TIME, 0],
                                      [361478.978, 291915.604, 0, 0.5970*NONDIM_TIME, -0.1946*NONDIM_TIME, 0]])/NONDIM_LENGTH
sensor_initial_conditions = np.array([[5.539207919986701700e-01, 0, 0, 0, 1.045674089930031636e+00, 0], 
                                      [0.5459235236258585, 0.23317978705332257, 0, -0.05449366983979726, 0.9513225041719617, 0], 
                                      [0.5382931241100969, 0.43100528150170114, 0, 0.007423897208115144, 0.7621824790096972, 0]])
sensor_dynamics_equation = CR3BP_DEs
earth_exclusion_angle = np.deg2rad(5)
moon_additional_angle = np.deg2rad(5)
sun_exclusion_angle = np.deg2rad(20)


# IMM parameters
initial_state_covariance =  scipy.linalg.block_diag(np.eye(3)*1.30072841e-4**2, np.eye(3)*9.76041363e-4**2)
initial_costate_covariance = np.eye(6)*1e-2**2
initial_acceleration_covariance = np.eye(3)*1e-2**2
# initial_estimate = initial_truth
IMM_measurement_covariance = measurement_noise_covariance * (1)**2
measurement_variances = np.array([np.deg2rad(1e-3)**2, (1e5*np.deg2rad(1e-3))**2])
coasting_costate_process_noise_covariance = scipy.linalg.block_diag(np.eye(3)*(1e-15)**2, np.eye(3)*(1e-15)**2, np.eye(6)*(1e-2)**2)
min_time_process_noise_covariance = scipy.linalg.block_diag(np.eye(3)*(1e-15)**2, np.eye(3)*(1e-9)**2, np.eye(3)*(1e-1)**2, np.eye(3)*(1e-2)**2)
coasting_acceleration_process_noise_covariance = scipy.linalg.block_diag(np.eye(3)*(1e-15)**2, np.eye(3)*(1e-15)**2, np.eye(3)*(1e-6)**2)
acceleration_process_noise_covariance = scipy.linalg.block_diag(np.eye(3)*(1e-15)**2, np.eye(3)*(1e-9)**2, np.eye(3)*(5e-3)**2)
initial_mode_probabilities = np.array([0.99, 0.01])
mode_transition_matrix = np.array([[0.99, 0.01],
                                   [0.01, 0.99]])
