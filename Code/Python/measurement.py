

import numpy as np

def azimuth_elevation(x, y, z):

    azimuth = np.arctan2(y, x)
    elevation = np.arctan(z, np.sqrt(x**2 + y**2))

    return azimuth, elevation