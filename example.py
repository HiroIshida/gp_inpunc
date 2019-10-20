import numpy as np
from gp import GaussianProcess 
from kernel import matern23_uncertain

kern = matern23_uncertain(l = 0.5, noise = 0.01, dim = 1)
cov = np.zeros((1, 1))
X = [(np.array(e), cov) for e in [[0.0], [0.3],[1.0]]]
Y = [0, 2, 1]
gp = GaussianProcess(X, Y, kern)
