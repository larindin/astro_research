

import numpy as np



def CR3BP_DEs(x, y, z, vx, vy, vz, mu):

    dXdt = np.zeros((6))

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

def CR3BP_jacobian(x, y, z, vx, vy, vz, mu):

    jacobian = np.zeros((6, 6))

    # Intermediate values
    d = np.array([x+mu, y, z])
    r = np.array([x+mu-1, y, z])
    dmag = np.linalg.norm(d)
    rmag = np.linalg.norm(r)


    jacobian[0] = np.array([0, 0, 0, 1, 0, 0])
    jacobian[1] = np.array([0, 0, 0, 0, 1, 0])
    jacobian[2] = np.array([0, 0, 0, 0, 0, 1])
    jacobian[3] = np.array([1 - (1 - mu)/dmag**3 + 3*(1 - mu)*(x + mu)**2/dmag**5 - mu/rmag**3 + 3*mu*(x - 1 + mu)**2/rmag**5,
                            3*(1 - mu)*(x + mu)*y/dmag**5 + 3*mu*(x - 1 + mu)*y/rmag**5,
                            3*(1 - mu)*(x + mu)*z/dmag**5 + 3*mu*(x - 1 + mu)*z/rmag**5, 0, 2, 0])
    jacobian[4] = np.array([3*(1 - mu)*(x + mu)*y/dmag**5 + 3*mu*(x - 1 + mu)*y/rmag**5,
                            1 - (1 - mu)/dmag**3 - mu/rmag**3 + 3*(1 - mu)*y**2/dmag**5 + 3*mu*y**2/rmag**5,
                            3*(1 - mu)*y*z/dmag**5 + 3*mu*y*z/rmag**5, -2, 0, 0])
    jacobian[5] = np.array([3*(1 - mu)*(x + mu)*z/dmag**5 + 3*mu*(x - 1 + mu)*z/rmag**5,
                            3*(1 - mu)*y*z/dmag**5 + 3*mu*y*z/rmag**5,
                            1 - (1 - mu)/dmag**3 - mu/rmag**3 + 3*(1 - mu)*z**2/dmag**5 + 3*mu*z**2/rmag**5, 0, 0, 0])
    
    return jacobian