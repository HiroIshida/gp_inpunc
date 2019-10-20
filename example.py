import numpy as np
import matplotlib.pyplot as plt
from gp import GaussianProcess 
from kernel import matern23_uncertain

def example_1d():
    kern = matern23_uncertain(1, l = 0.5, noise = 0.01)
    cov = np.zeros((1, 1))
    X = [(np.array(e), cov) for e in [[0.0], [0.3],[1.0]]]
    Y = [5.0, 7.0, 6.0]
    gp = GaussianProcess(X, Y, kern)
    gp.show(N_grid = 200, bmin = -5, bmax = 5)
    plt.show()

def example_2d():
    kern = matern23_uncertain(dim = 2, l = 0.5, noise = 0.01)
    cov = np.zeros((2, 2))
    X = [(np.array(e), cov) for e in [[0, 0.0], [0.3, 0.7],[0.6, 0.2], [1.0, 1.0]]]
    Y = [2, 3, 4, 1]
    gp = GaussianProcess(X, Y, kern)
    return gp

if __name__=="__main__":
    gp = example_2d()
    gp.show()
    plt.show()



