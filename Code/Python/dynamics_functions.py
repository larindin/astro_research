
import numpy as np
import scipy.integrate



class Truth:
    def __init__(self, time_vals, truth_vals):
        assert np.size(time_vals, 0) == np.size(truth_vals, 1)

        self.t = time_vals
        self.truth_vals = truth_vals

