import numpy as np
import matplotlib.pyplot as plt
from gp import GaussianProcess 
from classification import GPClassifier
from kernel import Matern23
import pickle

def example_1d():
    kern = Matern23(1, l = 0.5, noise = 0.01)
    cov = np.zeros((1, 1))
    X = [(np.array(e), cov) for e in [[0.0], [0.3],[1.0]]]
    Y = [5.0, 7.0, 6.0]
    gp = GaussianProcess(X, Y, kern)
    gp.show(N_grid = 200, bmin = -5, bmax = 5)
    plt.show()

def example_2d():
    kern = Matern23(dim = 2, l = 0.5, noise = 0.01)
    cov = np.zeros((2, 2))
    X = [(np.array(e), cov) for e in [[0, 0.0], [0.3, 0.7],[0.6, 0.2], [1.0, 1.0]]]
    Y = [2, 3, 4, 1]
    gp = GaussianProcess(X, Y, kern)
    return gp

def example_2d_gpc():
    gp = example_2d()
    gpc = GPClassifier(gp)
    return gpc

if __name__=="__main__":
    gpc = example_2d_gpc()



