

import numpy as np

def calculate_JC(IC):
    mu = 1.215059e-2
    
    x, y, z, vx, vy, vz = IC
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)

    return x**2 + y**2 + 2*((1 - mu)/r1 + (mu)/r2) - (vx**2 + vy**2 + vz**2)

def CR3BP_DEs(t, state, mu):

    x, y, z, vx, vy, vz = state

    d = np.array([x+mu, y, z])
    r = np.array([x+mu-1, y, z])
    dmag = np.linalg.norm(d)
    rmag = np.linalg.norm(r)

    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = -(1 - mu)*(x + mu)/dmag**3 - mu*(x - 1 + mu)/rmag**3 + 2*vy + x
    dvydt = -(1 - mu)*y/dmag**3 - mu*y/rmag**3 - 2*vx + y
    dvzdt = -(1 - mu)*z/dmag**3 - mu*z/rmag**3

    dXdt = np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt])

    return dXdt

def CR3BP_jacobian(state, mu):

    x, y, z, vx, vy, vz = state

    # Intermediate values
    d = np.array([x + mu, y, z])
    r = np.array([x - 1 + mu, y, z])
    dmag = np.linalg.norm(d)
    rmag = np.linalg.norm(r)

    dxdx = 1 - (1 - mu)/(dmag**3) - mu/(rmag**3) + 3*(1 - mu)*(x + mu)**2/(dmag**5) + 3*mu*(x - 1 + mu)**2/(rmag**5)
    dxdy = 3*(1 - mu)*(x + mu)*y/(dmag**5) + 3*mu*(x - 1 + mu)*y/(rmag**5)
    dxdz = 3*(1 - mu)*(x + mu)*z/(dmag**5) + 3*mu*(x - 1 + mu)*z/(rmag**5)
    dydx = 3*(1 - mu)*(x + mu)*y/(dmag**5) + 3*mu*(x - 1 + mu)*y/(rmag**5)
    dydy = 1 - (1 - mu)/(dmag**3) - mu/(rmag**3) + 3*(1 - mu)*y**2/(dmag**5) + 3*mu*y**2/(rmag**5)
    dydz = 3*(1 - mu)*y*z/(dmag**5) + 3*mu*y*z/(rmag**5)
    dzdx = 3*(1 - mu)*(x + mu)*z/(dmag**5) + 3*mu*(x - 1 + mu)*z/(rmag**5)
    dzdy = 3*(1 - mu)*y*z/(dmag**5) + 3*mu*y*z/(rmag**5)
    dzdz = -(1 - mu)/(dmag**3) - mu/(rmag**3) + 3*(1 - mu)*z**2/(dmag**5) + 3*mu*z**2/(rmag**5)

    jacobian = np.array([[0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1],
                         [dxdx, dxdy, dxdz, 0, 2, 0],
                         [dydx, dydy, dydz, -2, 0, 0],
                         [dzdx, dzdy, dzdz, 0, 0, 0]])
    
    return jacobian

def CR3BP_costate_DEs(t, state, costate, mu):

    jacobian = CR3BP_jacobian(state, mu)
    ddt_costate = -jacobian.T @ costate
    return ddt_costate

def CR3BP_costate_jacobian(state, costate, mu, umax):

    l1, l2, l3, l4, l5, l6 = costate
    x, y, z, vx, vy, vz = state
    
    d = np.array([x + mu, y, z])
    r = np.array([x - 1 + mu, y, z])
    dmag = np.linalg.norm(d)
    rmag = np.linalg.norm(r)
    lnorm = np.linalg.norm(np.array([l4, l5, l6]))

    B = np.vstack((np.zeros((3, 3)), np.eye(3)))

    dxdx = 1 - (1 - mu)/(dmag**3) - mu/(rmag**3) + 3*(1 - mu)*(x + mu)**2/(dmag**5) + 3*mu*(x - 1 + mu)**2/(rmag**5)
    dxdy = 3*(1 - mu)*(x + mu)*y/(dmag**5) + 3*mu*(x - 1 + mu)*y/(rmag**5)
    dxdz = 3*(1 - mu)*(x + mu)*z/(dmag**5) + 3*mu*(x - 1 + mu)*z/(rmag**5)
    dydx = 3*(1 - mu)*(x + mu)*y/(dmag**5) + 3*mu*(x - 1 + mu)*y/(rmag**5)
    dydy = 1 - (1 - mu)/(dmag**3) - mu/(rmag**3) + 3*(1 - mu)*y**2/(dmag**5) + 3*mu*y**2/(rmag**5)
    dydz = 3*(1 - mu)*y*z/(dmag**5) + 3*mu*y*z/(rmag**5)
    dzdx = 3*(1 - mu)*(x + mu)*z/(dmag**5) + 3*mu*(x - 1 + mu)*z/(rmag**5)
    dzdy = 3*(1 - mu)*y*z/(dmag**5) + 3*mu*y*z/(rmag**5)
    dzdz = -(1 - mu)/(dmag**3) - mu/(rmag**3) + 3*(1 - mu)*z**2/(dmag**5) + 3*mu*z**2/(rmag**5)

    dxdxdx = 3*(1 - mu)*(x + mu)/dmag**5 + 3*mu*(x - 1 + mu)/rmag**5 + 6*(1 - mu)*(x + mu)/dmag**5 - 15*(1 - mu)*(x + mu)**3/dmag**7 + 6*mu*(x - 1 + mu)/rmag**5 - 15*mu*(x - 1 + mu)**3/rmag**7
    dxdxdy = 3*(1 - mu)*y/dmag**5 + 3*mu*y/rmag**5 - 15*(1 - mu)*(x + mu)**2*y/dmag**7 - 15*mu*(x - 1 + mu)**2*y/rmag**7
    dxdxdz = 3*(1 - mu)*z/dmag**5 + 3*mu*z/rmag**5 - 15*(1 - mu)*(x + mu)**2*z/dmag**7 - 15*mu*(x - 1 + mu)**2*z/rmag**7
    dxdydx = 3*(1 - mu)*y/dmag**5 - 15*(1 - mu)*(x + mu)**2*y/dmag**7 + 3*mu*y/rmag**5 - 15*mu*(x - 1 + mu)**2*y/rmag**7
    dxdydy = 3*(1 - mu)*(x + mu)/dmag**5 - 15*(1 - mu)*(x + mu)*y**2/dmag**7 + 3*mu*(x - 1 + mu)/rmag**5 - 15*mu*(x - 1 + mu)*y**2/rmag**7
    dxdydz = -15*(1 - mu)*(x + mu)*y*z/dmag**7 - 15*mu*(x - 1 + mu)*y*z/rmag**7
    dxdzdx = 3*(1 - mu)*z/dmag**5 - 15*(1 - mu)*(x + mu)**2*z/dmag**7 + 3*mu*z/rmag**5 - 15*mu*(x - 1 + mu)**2*z/rmag**7
    dxdzdy = -15*(1 - mu)*(x + mu)*y*z/dmag**7 - 15*mu*(x - 1 + mu)*y*z/rmag**7
    dxdzdz = 3*(1 - mu)*(x + mu)/dmag**5 - 15*(1 - mu)*(x + mu)*z**2/dmag**7 + 3*mu*(x - 1 + mu)/rmag**5 - 15*mu*(x - 1 + mu)*z**2/rmag**7
    dydydx = 3*(1 - mu)*(x + mu)/dmag**5 + 3*mu*(x - 1 + mu)/rmag**5 - 15*(1 - mu)*y**2*(x + mu)/dmag**7 - 15*mu*(x - 1 + mu)*y**2/rmag**7
    dydydy = 3*(1 - mu)*y/dmag**5 + 3*mu*y/rmag**5 + 6*(1 - mu)*y/dmag**5 - 15*(1 - mu)*y**3/dmag**7 + 6*mu*y/rmag**5 - 15*mu*y**3/rmag**7
    dydydz = 3*(1 - mu)*z/dmag**5 + 3*mu*z/rmag**5 - 15*(1 - mu)*y**2*z/dmag**7 - 15*mu*y**2*z/rmag**7
    dydzdx = -15*(1 - mu)*y*z*(x + mu)/dmag**7 - 15*mu*y*z*(x + mu)/rmag**7
    dydzdy = 3*(1 - mu)*z/dmag**5 - 15*(1 - mu)*y**2*z/dmag**7 + 3*mu*z/rmag**5 - 15*mu*y**2*z/rmag**7
    dydzdz = 3*(1 - mu)*y/dmag**5 - 15*(1 - mu)*y*z**2/dmag**7 + 3*mu*y/rmag**5 - 15*mu*y*z**2/rmag**7
    dzdzdx = 3*(1 - mu)*(x + mu)/dmag**5 + 3*mu*(x - 1 + mu)/rmag**5 - 15*(1 - mu)*z**2*(x + mu)/dmag**7 - 15*mu*z**2*(x - 1 + mu)/rmag**7
    dzdzdy = 3*(1 - mu)*y/dmag**5 + 3*mu*y/rmag**5 - 15*(1 - mu)*z**2*y/dmag**7 - 15*mu*z**2*(x - 1 + mu)/rmag**7
    dzdzdz = 3*(1 - mu)*z/dmag**5 + 3*mu*z/rmag**5 + 6*(1 - mu)*z/dmag**5 - 15*(1 - mu)*z**3/dmag**7 + 6*mu*z/rmag**5 - 15*mu*z**3/rmag**7


    dXdX = np.array([[0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                    [dxdx, dxdy, dxdz, 0, 2, 0],
                    [dydx, dydy, dydz, -2, 0, 0],
                    [dzdx, dzdy, dzdz, 0, 0, 0]])
    
    if np.linalg.norm(-B.T @ costate) <= 2*umax:
        dXdL = np.vstack((np.zeros((3, 6)), np.hstack((np.zeros((3, 3)), -np.eye(3)*0.5))))
    else:
        dXdL_small = np.array([[-(1/lnorm - l4**2/lnorm**3), l4*l5/lnorm**3, l4*l6/lnorm**3],
                                 [l4*l5/lnorm**3, -(1/lnorm - l5**2/lnorm**3), l5*l6/lnorm**3],
                                 [l4*l6/lnorm**3, l5*l6/lnorm**3, -(1/lnorm - l6**2/lnorm**3)]]) * umax
        dXdL = np.vstack((np.zeros((3, 6)), np.hstack((np.zeros((3, 3)), dXdL_small))))
    
    dLdX_small = -np.array([[l4*dxdxdx + l5*dxdydx + l6*dxdzdx, l4*dxdxdy + l5*dxdydy + l6*dxdzdy, l4*dxdxdz + l5*dxdydy + l6*dxdzdy],
                            [l4*dxdydx + l5*dydydx + l6*dydzdx, l4*dxdydy + l5*dydydy + l6*dydzdy, l4*dxdydz + l5*dydydz + l6*dydzdz],
                            [l4*dxdzdx + l5*dydzdx + l6*dzdzdx, l4*dxdzdy + l5*dydzdy + l6*dzdzdy, l4*dxdzdz + l5*dydzdz + l6*dzdzdz]])
    dLdX = np.vstack((np.hstack((dLdX_small, np.zeros((3, 3)))), np.zeros((3, 6))))

    dLdL_small = -np.array([[dxdx, dxdy, dxdz],
                            [dxdy, dydy, dydz],
                            [dxdz, dydz, dzdz]])
    dLdL_small2 = np.array([[0, 2, 0],
                            [-2, 0, 0],
                            [0, 0, 0]])
    dLdL = np.vstack((np.hstack((np.zeros((3, 3)), dLdL_small)), np.hstack((-np.eye(3), dLdL_small2))))

    jacobian = np.vstack((np.hstack((dXdX, dXdL)), np.hstack((dLdX, dLdL))))

    return jacobian