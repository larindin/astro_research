
import numpy as np
from CR3BP_pontryagin import *

def reformulated_min_fuel_ODE(t, X, mu, umax, rho):

    state = X[0:6]
    theta = X[9]
    psi = X[10]
    eta = X[11]
    lambda_v = eta * np.array([np.cos(psi)*np.cos(theta), np.cos(psi)*np.sin(theta), np.sin(psi)])
    costate = np.concatenate((X[6:9], lambda_v))

    B = np.vstack((np.zeros((3, 3)), np.eye(3)))

    p = -B.T @ costate
    G = umax/2 * (1 + np.tanh((np.linalg.norm(p) - 1)/rho))
    control = G * p/np.linalg.norm(p)

    ddt_state_kepler = CR3BP_DEs(t, state, mu)
    ddt_state = ddt_state_kepler + B @ control

    ddt_costate = CR3BP_costate_DEs(t, state, costate, mu)

    ddt_theta = np.array([np.cos(theta)**2*(ddt_costate[4]/costate[3] - costate[4]*ddt_costate[3]/costate[3]**2)])
    ddt_eta = np.array([-np.sum(costate[0:3] * lambda_v/eta)])
    ddt_psi = -(costate[2] + ddt_eta*np.sin(psi))/eta/np.cos(psi)

    return np.concatenate((ddt_state, ddt_costate[0:3], ddt_theta, ddt_psi, ddt_eta), 0)

def reforumlated_min_fuel_jacobian(state, costate, mu, umax, rho):

    l1, l2, l3, theta, psi, eta = costate
    x, y, z, vx, vy, vz = state

    l4, l5, l6 = eta*np.array([np.cos(psi)*np.cos(theta), np.cos(psi)*np.sin(theta), np.sin(psi)])
    
    d = np.array([x + mu, y, z])
    r = np.array([x - 1 + mu, y, z])
    dmag = np.linalg.norm(d)
    rmag = np.linalg.norm(r)
    B = np.vstack((np.zeros((3, 3)), np.eye(3)))
    p = -B.T @ costate
    S = eta - 1
    G = umax/2 * (1 + np.tanh(S/rho))
    my_invcosh = lambda x: 1/np.cosh(x) if abs(x) < 700 else 0

    dxdx = 1 - (1 - mu)/(dmag**3) - mu/(rmag**3) + 3*(1 - mu)*(x + mu)**2/(dmag**5) + 3*mu*(x - 1 + mu)**2/(rmag**5)
    dxdy = 3*(1 - mu)*(x + mu)*y/(dmag**5) + 3*mu*(x - 1 + mu)*y/(rmag**5)
    dxdz = 3*(1 - mu)*(x + mu)*z/(dmag**5) + 3*mu*(x - 1 + mu)*z/(rmag**5)
    dydx = dxdy
    dydy = 1 - (1 - mu)/(dmag**3) - mu/(rmag**3) + 3*(1 - mu)*y**2/(dmag**5) + 3*mu*y**2/(rmag**5)
    dydz = 3*(1 - mu)*y*z/(dmag**5) + 3*mu*y*z/(rmag**5)
    dzdx = dxdz
    dzdy = dydz
    dzdz = -(1 - mu)/(dmag**3) - mu/(rmag**3) + 3*(1 - mu)*z**2/(dmag**5) + 3*mu*z**2/(rmag**5)

    dxdxdx = 3*(1 - mu)*(x + mu)/dmag**5 + 3*mu*(x - 1 + mu)/rmag**5 + 6*(1 - mu)*(x + mu)/dmag**5 - 15*(1 - mu)*(x + mu)**3/dmag**7 + 6*mu*(x - 1 + mu)/rmag**5 - 15*mu*(x - 1 + mu)**3/rmag**7
    dxdxdy = 3*(1 - mu)*y/dmag**5 + 3*mu*y/rmag**5 - 15*(1 - mu)*(x + mu)**2*y/dmag**7 - 15*mu*(x - 1 + mu)**2*y/rmag**7
    dxdxdz = 3*(1 - mu)*z/dmag**5 + 3*mu*z/rmag**5 - 15*(1 - mu)*(x + mu)**2*z/dmag**7 - 15*mu*(x - 1 + mu)**2*z/rmag**7
    dxdydx = dxdxdy
    dxdydy = 3*(1 - mu)*(x + mu)/dmag**5 - 15*(1 - mu)*(x + mu)*y**2/dmag**7 + 3*mu*(x - 1 + mu)/rmag**5 - 15*mu*(x - 1 + mu)*y**2/rmag**7
    dxdydz = -15*(1 - mu)*(x + mu)*y*z/dmag**7 - 15*mu*(x - 1 + mu)*y*z/rmag**7
    dxdzdx = dxdxdz
    dxdzdy = dxdydz
    dxdzdz = 3*(1 - mu)*(x + mu)/dmag**5 - 15*(1 - mu)*(x + mu)*z**2/dmag**7 + 3*mu*(x - 1 + mu)/rmag**5 - 15*mu*(x - 1 + mu)*z**2/rmag**7
    dydydx = dxdydy
    dydydy = 3*(1 - mu)*y/dmag**5 + 3*mu*y/rmag**5 + 6*(1 - mu)*y/dmag**5 - 15*(1 - mu)*y**3/dmag**7 + 6*mu*y/rmag**5 - 15*mu*y**3/rmag**7
    dydydz = 3*(1 - mu)*z/dmag**5 + 3*mu*z/rmag**5 - 15*(1 - mu)*y**2*z/dmag**7 - 15*mu*y**2*z/rmag**7
    dydzdx = dxdydz
    dydzdy = dydydz
    dydzdz = 3*(1 - mu)*y/dmag**5 - 15*(1 - mu)*y*z**2/dmag**7 + 3*mu*y/rmag**5 - 15*mu*y*z**2/rmag**7
    dzdzdx = dxdzdz
    dzdzdy = dydzdz
    dzdzdz = 3*(1 - mu)*z/dmag**5 + 3*mu*z/rmag**5 + 6*(1 - mu)*z/dmag**5 - 15*(1 - mu)*z**3/dmag**7 + 6*mu*z/rmag**5 - 15*mu*z**3/rmag**7

    dGdeta = umax/rho/2 * my_invcosh(S/rho)**2

    dxdtheta = np.cos(psi)*np.sin(theta)*G
    dxdpsi = np.sin(psi)*np.cos(theta)*G
    dxdeta = -np.cos(psi)*np.cos(theta)*dGdeta
    dydtheta = -np.cos(psi)*np.cos(theta)*G
    dydpsi = np.sin(psi)*np.sin(theta)*G
    dydeta = -np.cos(psi)*np.sin(theta)*dGdeta
    dzdtheta = 0
    dzdpsi = -np.cos(psi)*G
    dzdeta = -np.sin(psi)*dGdeta

    dl1dtheta = eta*(np.cos(psi)*np.sin(theta)*dxdx - np.cos(psi)*np.cos(theta)*dxdy)
    dl1dpsi = eta*(np.sin(psi)*np.cos(theta)*dxdx + np.sin(psi)*np.sin(theta)*dxdy - np.cos(psi)*dxdz)
    dl1deta = -np.cos(psi)*np.cos(theta)*dxdx - np.cos(psi)*np.sin(theta)*dxdy - np.sin(psi)*dxdz
    dl2dtheta = eta*(np.cos(psi)*np.sin(theta)*dxdy - np.cos(psi)*np.cos(theta)*dydy)
    dl2dpsi = eta*(np.sin(psi)*np.cos(theta)*dxdy + np.sin(psi)*np.sin(theta)*dydy - np.cos(psi)*dydz)
    dl2deta = -np.cos(psi)*np.cos(theta)*dxdy - np.cos(psi)*np.sin(theta)*dydy - np.sin(psi)*dydz
    dl3dtheta =eta*(np.cos(psi)*np.sin(theta)*dxdz - np.cos(psi)*np.cos(theta)*dydz)
    dl3dpsi = eta*(np.sin(psi)*np.cos(theta)*dxdz + np.sin(psi)*np.sin(theta)*dydz - np.cos(psi)*dzdz)
    dl3deta = -np.cos(psi)*np.cos(theta)*dxdz - np.cos(psi)*np.sin(theta)*dydz - np.sin(psi)*dzdz
    dthetadl1 = np.sin(theta)/eta/np.cos(psi)
    dthetadl2 = -np.cos(theta)/eta/np.cos(psi)
    dthetadtheta = l1*np.cos(theta)/eta/np.cos(psi) + l2*np.sin(theta)/eta/np.cos(psi)
    dthetadpsi = l1*np.sin(theta)*np.sin(psi)/eta/np.cos(psi)**2 - l2*np.cos(theta)*np.sin(psi)/eta/np.cos(psi)**2
    dthetadeta = l2*np.cos(theta)/eta**2/np.cos(psi) - l1*np.sin(theta)/eta**2/np.cos(psi)
    dpsidl1 = np.sin(psi)*np.cos(theta)/eta
    dpsidl2 = np.sin(psi)*np.sin(theta)/eta
    dpsidl3 = np.tan(psi)*np.sin(psi)/eta - 1/eta/np.cos(psi)
    dpsidtheta = -l1*np.sin(psi)*np.sin(theta)/eta + l2*np.sin(psi)*np.cos(theta)/eta
    dpsidpsi = -l3*np.sin(psi)/eta/np.cos(psi)**2 + l1*np.cos(psi)*np.cos(theta)/eta - l2*np.cos(psi)*np.sin(theta)/eta + l3*(np.tan(psi)*np.cos(psi) + np.tan(psi)/np.cos(psi))/eta
    dpsideta = l3/eta**2/np.cos(psi) - l1*np.sin(psi)*np.cos(theta)/eta**2 - l2*np.sin(psi)*np.sin(theta)/eta**2 - l3*np.tan(psi)*np.sin(psi)/eta**2
    detadl1 = -np.cos(psi)*np.cos(theta)
    detadl2 = -np.cos(psi)*np.sin(theta)
    detadl3 = -np.sin(psi)
    detadtheta = l1*np.cos(psi)*np.sin(theta) - l2*np.cos(psi)*np.cos(theta)
    detadpsi = l1*np.sin(psi)*np.cos(theta) + l2*np.sin(psi)*np.sin(theta) - l3*np.cos(psi)
    
    dXdX = np.array([[0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                    [dxdx, dxdy, dxdz, 0, 2, 0],
                    [dydx, dydy, dydz, -2, 0, 0],
                    [dzdx, dzdy, dzdz, 0, 0, 0]])
    
    dXdL_small = np.array([[dxdtheta, dxdpsi, dxdeta],
                           [dydtheta, dydpsi, dydeta],
                           [dzdtheta, dzdpsi, dzdeta]])
    dXdL = scipy.linalg.block_diag(np.zeros((3, 3)), dXdL_small)

    dLdX_small = -np.array([[l4*dxdxdx + l5*dxdydx + l6*dxdzdx, l4*dxdxdy + l5*dxdydy + l6*dxdzdy, l4*dxdxdz + l5*dxdydz + l6*dxdzdz],
                            [l4*dxdydx + l5*dydydx + l6*dydzdx, l4*dxdydy + l5*dydydy + l6*dydzdy, l4*dxdydz + l5*dydydz + l6*dydzdz],
                            [l4*dxdzdx + l5*dydzdx + l6*dzdzdx, l4*dxdzdy + l5*dydzdy + l6*dzdzdy, l4*dxdzdz + l5*dydzdz + l6*dzdzdz]])
    dLdX = scipy.linalg.block_diag(dLdX_small, np.zeros((3, 3)))
    
    dLdL = np.array([[0, 0, 0, dl1dtheta, dl1dpsi, dl1deta],
                     [0, 0, 0, dl2dtheta, dl2dpsi, dl2deta],
                     [0, 0, 0, dl3dtheta, dl3dpsi, dl3deta],
                     [dthetadl1, dthetadl2, 0, dthetadtheta, dthetadpsi, dthetadeta],
                     [dpsidl1, dpsidl2, dpsidl3, dpsidtheta, dpsidpsi, dpsideta],
                     [detadl1, detadl2, detadl3, detadtheta, detadpsi, 0]])

    jacobian = np.vstack((np.hstack((dXdX, dXdL)), np.hstack((dLdX, dLdL))))

    return jacobian

def reformulated_min_energy_ODE(t, X, mu, umax):

    state = X[0:6]
    theta = X[9]
    psi = X[10]
    eta = X[11]
    lambda_v = eta * np.array([np.cos(psi)*np.cos(theta), np.cos(psi)*np.sin(theta), np.sin(psi)])
    costate = np.concatenate((X[6:9], lambda_v))

    B = np.vstack((np.zeros((3, 3)), np.eye(3)))

    p = -B.T @ costate
    p_mag = np.linalg.norm(p)

    if p_mag > 2*umax:
        control = umax * p/p_mag
    else:
        control = p/2

    ddt_state_kepler = CR3BP_DEs(t, state, mu)
    ddt_state = ddt_state_kepler + B @ control

    ddt_costate = CR3BP_costate_DEs(t, state, costate, mu)

    ddt_theta = np.array([np.cos(theta)**2*(ddt_costate[4]/costate[3] - costate[4]*ddt_costate[3]/costate[3]**2)])
    ddt_eta = np.array([-np.sum(costate[0:3] * lambda_v/eta)])
    ddt_psi = -(costate[2] + ddt_eta*np.sin(psi))/eta/np.cos(psi)
    
    return np.concatenate((ddt_state, ddt_costate[0:3], ddt_theta, ddt_psi, ddt_eta), 0)

def reformulated_min_energy_jacobian(state, costate, mu, umax):

    l1, l2, l3, theta, psi, eta = costate
    x, y, z, vx, vy, vz = state
    
    d = np.array([x + mu, y, z])
    r = np.array([x - 1 + mu, y, z])
    dmag = np.linalg.norm(d)
    rmag = np.linalg.norm(r)
    l4, l5, l6 = eta*np.array([np.cos(psi)*np.cos(theta), np.cos(psi)*np.sin(theta), np.sin(psi)])

    dxdx = 1 - (1 - mu)/(dmag**3) - mu/(rmag**3) + 3*(1 - mu)*(x + mu)**2/(dmag**5) + 3*mu*(x - 1 + mu)**2/(rmag**5)
    dxdy = 3*(1 - mu)*(x + mu)*y/(dmag**5) + 3*mu*(x - 1 + mu)*y/(rmag**5)
    dxdz = 3*(1 - mu)*(x + mu)*z/(dmag**5) + 3*mu*(x - 1 + mu)*z/(rmag**5)
    dydx = dxdy
    dydy = 1 - (1 - mu)/(dmag**3) - mu/(rmag**3) + 3*(1 - mu)*y**2/(dmag**5) + 3*mu*y**2/(rmag**5)
    dydz = 3*(1 - mu)*y*z/(dmag**5) + 3*mu*y*z/(rmag**5)
    dzdx = dxdz
    dzdy = dydz
    dzdz = -(1 - mu)/(dmag**3) - mu/(rmag**3) + 3*(1 - mu)*z**2/(dmag**5) + 3*mu*z**2/(rmag**5)

    dxdxdx = 3*(1 - mu)*(x + mu)/dmag**5 + 3*mu*(x - 1 + mu)/rmag**5 + 6*(1 - mu)*(x + mu)/dmag**5 - 15*(1 - mu)*(x + mu)**3/dmag**7 + 6*mu*(x - 1 + mu)/rmag**5 - 15*mu*(x - 1 + mu)**3/rmag**7
    dxdxdy = 3*(1 - mu)*y/dmag**5 + 3*mu*y/rmag**5 - 15*(1 - mu)*(x + mu)**2*y/dmag**7 - 15*mu*(x - 1 + mu)**2*y/rmag**7
    dxdxdz = 3*(1 - mu)*z/dmag**5 + 3*mu*z/rmag**5 - 15*(1 - mu)*(x + mu)**2*z/dmag**7 - 15*mu*(x - 1 + mu)**2*z/rmag**7
    dxdydx = dxdxdy
    dxdydy = 3*(1 - mu)*(x + mu)/dmag**5 - 15*(1 - mu)*(x + mu)*y**2/dmag**7 + 3*mu*(x - 1 + mu)/rmag**5 - 15*mu*(x - 1 + mu)*y**2/rmag**7
    dxdydz = -15*(1 - mu)*(x + mu)*y*z/dmag**7 - 15*mu*(x - 1 + mu)*y*z/rmag**7
    dxdzdx = dxdxdz
    dxdzdy = dxdydz
    dxdzdz = 3*(1 - mu)*(x + mu)/dmag**5 - 15*(1 - mu)*(x + mu)*z**2/dmag**7 + 3*mu*(x - 1 + mu)/rmag**5 - 15*mu*(x - 1 + mu)*z**2/rmag**7
    dydydx = dxdydy
    dydydy = 3*(1 - mu)*y/dmag**5 + 3*mu*y/rmag**5 + 6*(1 - mu)*y/dmag**5 - 15*(1 - mu)*y**3/dmag**7 + 6*mu*y/rmag**5 - 15*mu*y**3/rmag**7
    dydydz = 3*(1 - mu)*z/dmag**5 + 3*mu*z/rmag**5 - 15*(1 - mu)*y**2*z/dmag**7 - 15*mu*y**2*z/rmag**7
    dydzdx = dxdydz
    dydzdy = dydydz
    dydzdz = 3*(1 - mu)*y/dmag**5 - 15*(1 - mu)*y*z**2/dmag**7 + 3*mu*y/rmag**5 - 15*mu*y*z**2/rmag**7
    dzdzdx = dxdzdz
    dzdzdy = dydzdz
    dzdzdz = 3*(1 - mu)*z/dmag**5 + 3*mu*z/rmag**5 + 6*(1 - mu)*z/dmag**5 - 15*(1 - mu)*z**3/dmag**7 + 6*mu*z/rmag**5 - 15*mu*z**3/rmag**7

    dl1dtheta = eta*(np.cos(psi)*np.sin(theta)*dxdx - np.cos(psi)*np.cos(theta)*dxdy)
    dl1dpsi = eta*(np.sin(psi)*np.cos(theta)*dxdx + np.sin(psi)*np.sin(theta)*dxdy - np.cos(psi)*dxdz)
    dl1deta = -np.cos(psi)*np.cos(theta)*dxdx - np.cos(psi)*np.sin(theta)*dxdy - np.sin(psi)*dxdz
    dl2dtheta = eta*(np.cos(psi)*np.sin(theta)*dxdy - np.cos(psi)*np.cos(theta)*dydy)
    dl2dpsi = eta*(np.sin(psi)*np.cos(theta)*dxdy + np.sin(psi)*np.sin(theta)*dydy - np.cos(psi)*dydz)
    dl2deta = -np.cos(psi)*np.cos(theta)*dxdy - np.cos(psi)*np.sin(theta)*dydy - np.sin(psi)*dydz
    dl3dtheta =eta*(np.cos(psi)*np.sin(theta)*dxdz - np.cos(psi)*np.cos(theta)*dydz)
    dl3dpsi = eta*(np.sin(psi)*np.cos(theta)*dxdz + np.sin(psi)*np.sin(theta)*dydz - np.cos(psi)*dzdz)
    dl3deta = -np.cos(psi)*np.cos(theta)*dxdz - np.cos(psi)*np.sin(theta)*dydz - np.sin(psi)*dzdz
    dthetadl1 = np.sin(theta)/eta/np.cos(psi)
    dthetadl2 = -np.cos(theta)/eta/np.cos(psi)
    dthetadtheta = l1*np.cos(theta)/eta/np.cos(psi) + l2*np.sin(theta)/eta/np.cos(psi)
    dthetadpsi = l1*np.sin(theta)*np.sin(psi)/eta/np.cos(psi)**2 - l2*np.cos(theta)*np.sin(psi)/eta/np.cos(psi)**2
    dthetadeta = l2*np.cos(theta)/eta**2/np.cos(psi) - l1*np.sin(theta)/eta**2/np.cos(psi)
    dpsidl1 = np.sin(psi)*np.cos(theta)/eta
    dpsidl2 = np.sin(psi)*np.sin(theta)/eta
    dpsidl3 = np.tan(psi)*np.sin(psi)/eta - 1/eta/np.cos(psi)
    dpsidtheta = -l1*np.sin(psi)*np.sin(theta)/eta + l2*np.sin(psi)*np.cos(theta)/eta
    dpsidpsi = -l3*np.sin(psi)/eta/np.cos(psi)**2 + l1*np.cos(psi)*np.cos(theta)/eta - l2*np.cos(psi)*np.sin(theta)/eta + l3*(np.tan(psi)*np.cos(psi) + np.tan(psi)/np.cos(psi))/eta
    dpsideta = l3/eta**2/np.cos(psi) - l1*np.sin(psi)*np.cos(theta)/eta**2 - l2*np.sin(psi)*np.sin(theta)/eta**2 - l3*np.tan(psi)*np.sin(psi)/eta**2
    detadl1 = -np.cos(psi)*np.cos(theta)
    detadl2 = -np.cos(psi)*np.sin(theta)
    detadl3 = -np.sin(psi)
    detadtheta = l1*np.cos(psi)*np.sin(theta) - l2*np.cos(psi)*np.cos(theta)
    detadpsi = l1*np.sin(psi)*np.cos(theta) + l2*np.sin(psi)*np.sin(theta) - l3*np.cos(psi)

    dXdX = np.array([[0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                    [dxdx, dxdy, dxdz, 0, 2, 0],
                    [dydx, dydy, dydz, -2, 0, 0],
                    [dzdx, dzdy, dzdz, 0, 0, 0]])
    
    if eta <= 2*umax:
        dXdL_small = np.array([[eta*np.cos(psi)*np.sin(theta), eta*np.sin(psi)*np.cos(theta), -np.cos(psi)*np.cos(theta)],
                               [-eta*np.cos(psi)*np.cos(theta), eta*np.sin(psi)*np.sin(theta), -np.cos(psi)*np.sin(theta)],
                               [0, -eta*np.cos(psi), -np.sin(psi)]]) / 2
        dXdL = scipy.linalg.block_diag(np.zeros((3, 3)), dXdL_small)
    else:
        dXdL_small = np.array([[np.cos(psi)*np.sin(theta), np.sin(psi)*np.cos(theta), 0],
                               [-np.cos(psi)*np.cos(theta), np.sin(psi)*np.sin(theta), 0],
                               [0, -np.cos(psi), 0]]) * umax
        dXdL = scipy.linalg.block_diag(np.zeros((3, 3)), dXdL_small)
    
    dLdX_small = -np.array([[l4*dxdxdx + l5*dxdydx + l6*dxdzdx, l4*dxdxdy + l5*dxdydy + l6*dxdzdy, l4*dxdxdz + l5*dxdydz + l6*dxdzdz],
                            [l4*dxdydx + l5*dydydx + l6*dydzdx, l4*dxdydy + l5*dydydy + l6*dydzdy, l4*dxdydz + l5*dydydz + l6*dydzdz],
                            [l4*dxdzdx + l5*dydzdx + l6*dzdzdx, l4*dxdzdy + l5*dydzdy + l6*dzdzdy, l4*dxdzdz + l5*dydzdz + l6*dzdzdz]])
    dLdX = scipy.linalg.block_diag(dLdX_small, np.zeros((3, 3)))
    
    dLdL = np.array([[0, 0, 0, dl1dtheta, dl1dpsi, dl1deta],
                     [0, 0, 0, dl2dtheta, dl2dpsi, dl2deta],
                     [0, 0, 0, dl3dtheta, dl3dpsi, dl3deta],
                     [dthetadl1, dthetadl2, 0, dthetadtheta, dthetadpsi, dthetadeta],
                     [dpsidl1, dpsidl2, dpsidl3, dpsidtheta, dpsidpsi, dpsideta],
                     [detadl1, detadl2, detadl3, detadtheta, detadpsi, 0]])

    jacobian = np.vstack((np.hstack((dXdX, dXdL)), np.hstack((dLdX, dLdL))))

    return jacobian

def reformulated2standard(costate):
    theta, psi, eta = costate[3:6]
    new_costate = costate.copy()
    new_costate[3:6] = eta * np.array([np.cos(psi)*np.cos(theta), np.cos(psi)*np.sin(theta), np.sin(psi)])
    return new_costate

def standard2reformulated(costate):
    l4, l5, l6 = costate[3:6]
    eta = np.linalg.norm(costate[3:6])
    psi = np.arcsin(l6/eta)
    theta = np.arctan2(l5, l4)
    new_costate = costate.copy()
    new_costate[3:6] = np.array([theta, psi, eta])
    return new_costate

def get_reformulated_min_fuel_control(costate_output, umax, rho):

    B = np.vstack((np.zeros((3, 3)), np.eye(3)))

    control = costate_output[0:3]*0
    for time_index in range(len(costate_output[0])):

        theta, psi, eta = costate_output[3:6, time_index]

        control[:, time_index] = -umax/2 * (1 + np.tanh((eta - 1)/rho)) * np.array([np.cos(psi)*np.cos(theta), np.cos(psi)*np.sin(theta), np.sin(psi)])
    
    return control

def get_reformulated_min_energy_control(costate_output, umax):

    control = costate_output[0:3]*0
    for time_index in range(len(costate_output[0])):

        theta, psi, eta = costate_output[3:6, time_index]
        p = -eta * np.array([np.cos(psi)*np.cos(theta), np.cos(psi)*np.sin(theta), np.sin(psi)])
        p_mag = np.linalg.norm(p)

        if p_mag > 2*umax:
            control[:, time_index] = umax * p/p_mag
        else:
            control[:, time_index] = p/2
    
    return control
   