

import numpy as np
import scipy

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

def CR3BP_accel_DEs(t, state, mu):

    x, y, z, vx, vy, vz, ax, ay, az = state

    d = np.array([x+mu, y, z])
    r = np.array([x+mu-1, y, z])
    dmag = np.linalg.norm(d)
    rmag = np.linalg.norm(r)

    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = -(1 - mu)*(x + mu)/dmag**3 - mu*(x - 1 + mu)/rmag**3 + 2*vy + x + ax
    dvydt = -(1 - mu)*y/dmag**3 - mu*y/rmag**3 - 2*vx + y + ay
    dvzdt = -(1 - mu)*z/dmag**3 - mu*z/rmag**3 + az

    dXdt = np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt, 0, 0, 0])

    return dXdt

def CR3BP_accel_umax_DEs(t, state, mu, umax):

    x, y, z, vx, vy, vz, ax, ay, az = state

    d = np.array([x+mu, y, z])
    r = np.array([x+mu-1, y, z])
    a = np.array([ax, ay, az])
    dmag = np.linalg.norm(d)
    rmag = np.linalg.norm(r)
    ahat = a / np.linalg.norm(a)

    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = -(1 - mu)*(x + mu)/dmag**3 - mu*(x - 1 + mu)/rmag**3 + 2*vy + x + ahat[0]*umax
    dvydt = -(1 - mu)*y/dmag**3 - mu*y/rmag**3 - 2*vx + y + ahat[1]*umax
    dvzdt = -(1 - mu)*z/dmag**3 - mu*z/rmag**3 + ahat[2]*umax

    dXdt = np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt, 0, 0, 0])

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
    dydx = dxdy
    dydy = 1 - (1 - mu)/(dmag**3) - mu/(rmag**3) + 3*(1 - mu)*y**2/(dmag**5) + 3*mu*y**2/(rmag**5)
    dydz = 3*(1 - mu)*y*z/(dmag**5) + 3*mu*y*z/(rmag**5)
    dzdx = dxdz
    dzdy = dydz
    dzdz = -(1 - mu)/(dmag**3) - mu/(rmag**3) + 3*(1 - mu)*z**2/(dmag**5) + 3*mu*z**2/(rmag**5)

    jacobian = np.array([[0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1],
                         [dxdx, dxdy, dxdz, 0, 2, 0],
                         [dydx, dydy, dydz, -2, 0, 0],
                         [dzdx, dzdy, dzdz, 0, 0, 0]])
    
    return jacobian

def CR3BP_accel_jacobian(state, mu):

    x, y, z, vx, vy, vz, ax, ay, az = state

    # Intermediate values
    d = np.array([x + mu, y, z])
    r = np.array([x - 1 + mu, y, z])
    dmag = np.linalg.norm(d)
    rmag = np.linalg.norm(r)

    dxdx = 1 - (1 - mu)/(dmag**3) - mu/(rmag**3) + 3*(1 - mu)*(x + mu)**2/(dmag**5) + 3*mu*(x - 1 + mu)**2/(rmag**5)
    dxdy = 3*(1 - mu)*(x + mu)*y/(dmag**5) + 3*mu*(x - 1 + mu)*y/(rmag**5)
    dxdz = 3*(1 - mu)*(x + mu)*z/(dmag**5) + 3*mu*(x - 1 + mu)*z/(rmag**5)
    dydx = dxdy
    dydy = 1 - (1 - mu)/(dmag**3) - mu/(rmag**3) + 3*(1 - mu)*y**2/(dmag**5) + 3*mu*y**2/(rmag**5)
    dydz = 3*(1 - mu)*y*z/(dmag**5) + 3*mu*y*z/(rmag**5)
    dzdx = dxdz
    dzdy = dydz
    dzdz = -(1 - mu)/(dmag**3) - mu/(rmag**3) + 3*(1 - mu)*z**2/(dmag**5) + 3*mu*z**2/(rmag**5)

    jacobian = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [dxdx, dxdy, dxdz, 0, 2, 0, 1, 0, 0],
                         [dydx, dydy, dydz, -2, 0, 0, 0, 1, 0],
                         [dzdx, dzdy, dzdz, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    return jacobian

def CR3BP_accel_umax_jacobian(state, mu, umax):

    x, y, z, vx, vy, vz, ax, ay, az = state

    # Intermediate values
    d = np.array([x + mu, y, z])
    r = np.array([x - 1 + mu, y, z])
    a = np.array([ax, ay, az])
    dmag = np.linalg.norm(d)
    rmag = np.linalg.norm(r)
    amag = np.linalg.norm(a)

    dxdx = 1 - (1 - mu)/(dmag**3) - mu/(rmag**3) + 3*(1 - mu)*(x + mu)**2/(dmag**5) + 3*mu*(x - 1 + mu)**2/(rmag**5)
    dxdy = 3*(1 - mu)*(x + mu)*y/(dmag**5) + 3*mu*(x - 1 + mu)*y/(rmag**5)
    dxdz = 3*(1 - mu)*(x + mu)*z/(dmag**5) + 3*mu*(x - 1 + mu)*z/(rmag**5)
    dydx = dxdy
    dydy = 1 - (1 - mu)/(dmag**3) - mu/(rmag**3) + 3*(1 - mu)*y**2/(dmag**5) + 3*mu*y**2/(rmag**5)
    dydz = 3*(1 - mu)*y*z/(dmag**5) + 3*mu*y*z/(rmag**5)
    dzdx = dxdz
    dzdy = dydz
    dzdz = -(1 - mu)/(dmag**3) - mu/(rmag**3) + 3*(1 - mu)*z**2/(dmag**5) + 3*mu*z**2/(rmag**5)

    daxdax = umax*(1/amag - ax**2/amag**3)
    daxday = -umax*ax*ay/amag**3
    daxdaz = -umax*ax*az/amag**3
    daydax = daxday
    dayday = umax*(1/amag - ay**2/amag**3)
    daydaz = -umax*ay*az/amag**3
    dazdax = daxdaz
    dazday = daydaz
    dazdaz = umax*(1/amag - az**2/amag**3)

    jacobian = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [dxdx, dxdy, dxdz, 0, 2, 0, daxdax, daxday, daxdaz],
                         [dydx, dydy, dydz, -2, 0, 0, daydax, dayday, daydaz],
                         [dzdx, dzdy, dzdz, 0, 0, 0, dazdax, dazday, dazdaz],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    return jacobian
    
def CR3BP_costate_DEs(t, state, costate, mu):

    jacobian = CR3BP_jacobian(state, mu)
    ddt_costate = -jacobian.T @ costate
    return ddt_costate

def get_accel_umax_control(a_vecs, umax):

    control = np.empty(np.shape(a_vecs))

    for ax_index in range(3):
        control[ax_index] = umax * a_vecs[ax_index] / np.linalg.norm(a_vecs, axis=0)

    return control

def get_accel_umax_ctrl_cov(a_vecs, posterior_covariance_vals, umax):
    
    num_timesteps = np.size(a_vecs, 1)
    control_covariance_vals = np.empty((3, 3, num_timesteps))

    for timestep_index in range(num_timesteps):
        
        ax, ay, az = a_vecs[:, timestep_index]
        amag = np.linalg.norm([ax, ay, az])

        transform = np.array([[1/amag - ax**2/amag**3, -ax*ay/amag**3, -ax*az/amag**3],
                              [-ax*ay/amag**3, 1/amag - ay**2/amag**3, -ay*az/amag**3],
                              [-ax*az/amag**3, -ay*az/amag**3, 1/amag - az**2/amag**3]]) * umax
        
        control_covariance_vals[:, :, timestep_index] = transform @ posterior_covariance_vals[6:9, 6:9, timestep_index] @ transform.T
    
    return control_covariance_vals