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

    def train(self, x, y):
        self.X.append(x)
        self.Y.append(y)
        self.gp = GaussianProcess(self.X, self.Y, deepcopy(self.kernel))

    def pick_next_input(self): 
        raise NotImplementedError()









