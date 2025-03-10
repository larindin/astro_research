
import numpy as np
import scipy
import matplotlib.pyplot as plt
from CR3BP import *
from helper_functions import *

initial_conditions = np.array([L2, 0, 0, 0, 0, 0])
final_time = 50*24/NONDIM_TIME_HR
timespan = [0, final_time]

args = (mu,)
prop = scipy.integrate.solve_ivp(CR3BP_DEs, timespan, initial_conditions, args=args, atol=1e-12, rtol=1e-12)

vals = prop.y

ax = plt.figure().add_subplot(projection="3d")
ax.plot(vals[0], vals[1], vals[2])
ax.set_aspect("equal")

plt.show()