import numpy as np
from copy import deepcopy
from gp import GaussianProcess

class BayesianOpt(object):

    def __init__(self, kernel):
        self.kernel_init = kernel
        self.gp = None
        self.X = []
        self.Y = []

    def init(self):
        self.gp = None
        self.X = []
        self.Y = []

    def train(self, x, y,
            doOptmization = False, params_min = None, params_max = None):

        self.X.append(x)
        self.Y.append(y)
        self.gp = GaussianProcess(self.X, self.Y, self.gp.kernel)
        if doOptmization:
            self.gp.optimize(params_min = params_min, params_max = params_max)

    def pick_next_input(self): 
        raise NotImplementedError("overwrite this")

