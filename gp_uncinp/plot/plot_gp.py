import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import utils

def show(self, **kwargs):
    raise NotImplementedError()
    if self.dim == 1:
        self._show1d(**kwargs)
    elif self.dim == 2:
        self._show2d(**kwargs)
    elif self.dim == 3:
        self._show3d(**kwargs)
    else:
        raise NotImplementedError()

def _show1d(self, bmin = None, bmax = None, margin = 0.2, N_grid = 20):
    if bmin is None or bmax is None:
        bmin_, bmax_ = self.get_boundary(margin = margin)
        bmin = bmin.item()
        bmax = bmax.item()

    x_lin = np.linspace(bmin, bmax, N_grid)
    mu_lin = np.zeros(N_grid)
    std_lin = np.zeros(N_grid)
    for i in range(N_grid):
        x = (np.array([[x_lin[i]]]), 0)
        mu, var = self.predict(x)
        mu_lin[i] = mu
        std_lin[i] = math.sqrt(var)
    plt.plot(x_lin, mu_lin, c = 'b')
    plt.plot(x_lin, mu_lin + std_lin * 2, c = 'r')
    plt.plot(x_lin, mu_lin - std_lin * 2, c = 'r')

def _show2d(self, bmin = None, bmax = None, margin = 0.2, N_grid = 20, levels = None,
        scatter_rule = None):
    """
    scatter_rule: Y -> [(idxes_1, "blue"), ..., (idxes_n, "red")]
    """

    def func(x):
        mu, var = self.predict(x)
        return mu, mu

    if (bmin is None) or (bmax is None):
        bmin, bmax = self.get_boundary(margin = margin)

    fig, ax = plt.subplots() 
    fax = (fig, ax)
    utils.show2d(func, bmin, bmax, fax = fax, levels = levels, N = N_grid) 

    if scatter_rule is None:
        ## Y -> [(idxes_1, c1), ..., (idxes_n, cn)]
        def f(Y):
            idxes = range(len(Y))
            color = "red"
            return [(idxes, color)]
        scatter_rule = f

    utils.scatter(scatter_rule, self.X, self.Y, fax)

def _show3d(self, bmin = None, bmax = None, margin = 0.2, N_grid = 20, levels = None,
        scatter_rule = None):

    if (bmin is None) or (bmax is None):
        bmin, bmax = self.get_boundary(margin = margin)

    fig = plt.figure()
    ax = Axes3D(fig)
    fax = (fig, ax)

    if scatter_rule is None:
        ## Y -> [(idxes_1, c1), ..., (idxes_n, cn)]
        def f(Y):
            idxes = range(len(Y))
            color = "red"
            return [(idxes, color)]
        scatter_rule = f

    utils.scatter3d(scatter_rule, self.X, self.Y, fax)

