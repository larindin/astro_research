

import numpy as np
import scipy
from CR3BP import *

def CR3BP_ODE(t, X, mu):
    return CR3BP_DEs(X, mu)

def CR3BP_position_continuity_constraint(X, node_num, num_nodes, mu):

    state_dim = 7
    dt = X[-3]/num_nodes

    previous_post = X[state_dim*(node_num-1):state_dim*(node_num)-1]
    current_post = X[state_dim*(node_num):state_dim*(node_num+1)-1]

    current_pre = scipy.integrate.solve_ivp(CR3BP_ODE, np.array([0, dt]), previous_post, args=(mu,), atol=1e-12, rtol=1e-12).y[:, -1]

    return np.linalg.norm(current_post - current_pre)

def mass_continuity_constraint(X, node_num, num_nodes, Isp, epsilon, mu):

    state_dim = 7
    dt = X[-3]/num_nodes

    previous_post = X[state_dim*(node_num-1):state_dim*node_num]
    current_post = X[state_dim*node_num:state_dim*(node_num+1)]

    current_pre = scipy.integrate.solve_ivp(CR3BP_ODE, np.array([0, dt]), previous_post[0:6], args=(mu,), atol=1e-12, rtol=1e-12).y[:, -1]

    deltaV = np.sqrt((current_post[3:6] - current_pre[3:6])**2 + epsilon**2)
    dim_deltaV = deltaV*3.844000e8/3.751903e5

    previous_post_mass = previous_post[6]
    current_post_mass = current_post[6]

    return current_post_mass - previous_post_mass*np.exp(-dim_deltaV/9.81/Isp)

def initial_mass_continuity_constraint(X, initial_orbit_ICs, m0, Isp, epsilon, mu):

    state_dim = 7

    current_post = X[0:state_dim]
    initial_tau = X[-2]

    current_pre = scipy.integrate.solve_ivp(CR3BP_ODE, np.array([0, initial_tau]), initial_orbit_ICs, args=(mu,), atol=1e-12, rtol=1e-12).y[:, -1]

    deltaV = np.sqrt((current_post[3:6] - current_pre[3:6])**2 + epsilon**2)
    dim_deltaV = deltaV*3.844000e8/3.751903e5

    previous_post_mass = m0
    current_post_mass = current_post[6]

    return current_post_mass - previous_post_mass*np.exp(-dim_deltaV/9.81/Isp)

def thrust_constraint(X, node_num, num_nodes, Tmax, epsilon, mu):

    state_dim = 7
    dt = X[-3]/num_nodes

    previous_post = X[state_dim*(node_num-1):state_dim*node_num]
    current_post = X[state_dim*node_num:state_dim*(node_num+1)]

    current_pre = scipy.integrate.solve_ivp(CR3BP_ODE, np.array([0, dt]), previous_post[0:6], args=(mu,), atol=1e-12, rtol=1e-12).y[:, -1]

    deltaV = np.sqrt((current_post[3:6] - current_pre[3:6])**2 + epsilon**2)
    
    dim_deltaV = deltaV*3.844000e8/3.751903e5
    dim_dt = dt*3.751903e5

    current_post_mass = current_post[6]

    return -(dim_deltaV - Tmax/current_post_mass*dim_dt)

def initial_thrust_constraint(X, num_nodes, initial_orbit_ICs, Tmax, epsilon, mu):

    state_dim = 7
    dt = X[-3]/num_nodes

    current_post = X[0:state_dim]
    initial_tau = X[-2]

    current_pre = scipy.integrate.solve_ivp(CR3BP_ODE, np.array([0, initial_tau]), initial_orbit_ICs, args=(mu,), atol=1e-12, rtol=1e-12).y[:, -1]

    deltaV = np.sqrt((current_post[3:6] - current_pre[3:6])**2 + epsilon**2)
    
    dim_deltaV = deltaV*3.844000e8/3.751903e5
    dim_dt = dt*3.751903e5

    current_post_mass = current_post[6]

    return -(dim_deltaV - Tmax/current_post_mass*dim_dt)

def initial_orbit_constraint(X, initial_orbit_ICs, mu):

    initial_position = X[:3]
    initial_tau = X[-2]

    initial_orbit_position = scipy.integrate.solve_ivp(CR3BP_ODE, np.array([0, initial_tau]), initial_orbit_ICs, args=(mu,), atol=1e-12, rtol=1e-12).y[:3, -1]

    return np.linalg.norm(initial_position - initial_orbit_position)

def initial_tau_constraint(X, initial_orbit_period):
    initial_tau = X[-2]
    return initial_orbit_period - initial_tau

def final_orbit_constraint(X, final_orbit_ICs, mu):

    state_dim = 7

    final_state = X[-(state_dim + 3):-3]
    final_tau = X[-1]

    final_orbit_state = scipy.integrate.solve_ivp(CR3BP_ODE, np.array([final_tau, 0]), final_orbit_ICs, args=(mu,), atol=1e-12, rtol=1e-12).y[:, -1]

    return np.linalg.norm(final_state[0:6] - final_orbit_state)

def final_tau_constraint(X, final_orbit_period):
    final_tau = X[-1]
    return final_orbit_period - final_tau

def initial_mass_constraint(X, m0):

    state_dim = 7
    initial_mass = X[state_dim]

    return m0 - initial_mass

def minimum_mass_objective(X):
    return -X[-4]