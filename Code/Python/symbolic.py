import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import sympy as sp

x, y, z, vx, vy, vz, mu = sp.symbols("x y z vx vy vz mu")

X = sp.Matrix([vx, 
               vy, 
               vz, 
               -(1 - mu)*(x + mu)/sp.sqrt((x + mu)**2 + y**2 + z**2)**3 - mu*(x - 1 + mu)/sp.sqrt((x + mu -1)**2 + y**2 + z**2)**3 + 2*vy + x,
               -(1 - mu)*y/sp.sqrt((x + mu)**2 + y**2 + z**2)**3 - mu*y/sp.sqrt((x - 1 + mu)**2 + y**2 + z**2)**3 - 2*vx + y
               -(1 - mu)*z/sp.sqrt((x + mu)**2 + y**2 + z**2)**3 - mu*z/sp.sqrt((x - 1 + mu)**2 + y**2 + z**2)**3])

Y = sp.Matrix([x, y, z, vx, vy, vz])

sp.pprint(X.jacobian(Y))