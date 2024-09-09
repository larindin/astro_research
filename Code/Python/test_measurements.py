

import numpy as np
import matplotlib.pyplot as plt
import scipy
from measurement_functions import *
from helper_functions import *
from CR3BP import *

def dynamics_equation(t, state, mu):

    return CR3BP_DEs(state, mu)

mu = 1.215059e-2

target_orbit_ICs = np.array([9.091988819167088343e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,4.853617536824898493e-01,0.000000000000000000e+00])
# observer_orbit_ICs = target_orbit_ICs + np.ones(6)*1e-3
observer_orbit_ICs = np.array([6.825171035620000159e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,7.297344905967444451e-01,0.000000000000000000e+00])
final_time = 5.04
time_vals = np.arange(0, final_time, 0.001)
tspan = np.array([time_vals[0], time_vals[-1]])

target_propagation = scipy.integrate.solve_ivp(dynamics_equation, tspan, target_orbit_ICs, args=(mu,), t_eval=time_vals, atol=1e-12, rtol=1e-12)
target_truth_vals = target_propagation.y

observer_propagation = scipy.integrate.solve_ivp(dynamics_equation, tspan, observer_orbit_ICs, args=(mu,), t_eval=time_vals, atol=1e-12, rtol=1e-12)
observer_truth_vals = observer_propagation.y

moon_vectors = generate_moon_vectors(time_vals, observer_truth_vals[0:3, :])
earth_vectors = generate_earth_vectors(time_vals, observer_truth_vals[0:3, :])
sun_vectors = generate_sun_vectors(time_vals, np.pi)

validity_vector_moon = check_validity(time_vals, target_truth_vals[0:3, :], observer_truth_vals[0:3, :], moon_vectors, check_exclusion_dynamic, (9.0400624349e-3, 0))
validity_vector_earth = check_validity(time_vals, target_truth_vals[0:3, :], observer_truth_vals[0:3, :], earth_vectors, check_exclusion, (np.deg2rad(5), ))
validity_vector_sun = check_validity(time_vals, target_truth_vals[0:3, :], observer_truth_vals[0:3, :], sun_vectors, check_exclusion, (np.deg2rad(20), ))

validity_vector = validity_vector_moon * validity_vector_earth * validity_vector_sun

ax = plt.figure().add_subplot(projection="3d")
ax.plot(target_truth_vals[0], target_truth_vals[1], target_truth_vals[2])
ax.plot(observer_truth_vals[0], observer_truth_vals[1], observer_truth_vals[2])

ax = plt.figure().add_subplot()
ax.plot(earth_vectors[0], earth_vectors[1])
ax.plot(sun_vectors[0], sun_vectors[1])

ax = plt.figure().add_subplot()
ax.plot(validity_vector_earth, alpha=0.25)
ax.plot(validity_vector_sun, alpha=0.25)
ax.plot(validity_vector_moon, alpha=0.25)
# ax.plot(validity_vector)

ax = plt.figure().add_subplot()
ax.plot(np.array([0, 1, 2]), np.array([1, np.nan, 1]))

plt.show()