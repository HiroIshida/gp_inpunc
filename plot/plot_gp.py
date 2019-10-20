import numpy as np
import matplotlib.pyplot as plt
import math
from utils import show2d

def show(self, **kwargs):
    if self.dim == 1:
        self._show1d(**kwargs)
    elif self.dim == 2:
        self._show2d(**kwargs)
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
    example:
        def f(Y):
            idxes_1 = idxes_2 = []
            for idx in range(len(Y)):
                y = Y[idx]
                if y > 0.0:
                    idxes_1.append(idx)
                else:
                    idxes_2.append(idx)
            return [(idxes_1, "blue"), (idxes_2, "red")]
    """

    def func(x):
        mu, var = self.predict(x)
        return mu, mu

    if (bmin is None) or (bmax is None):
        bmin, bmax = self.get_boundary(margin = margin)

    fig, ax = plt.subplots() 
    fax = (fig, ax)
    show2d(func, bmin, bmax, fax = fax, levels = levels) 

    if scatter_rule is None:
        ## Y -> [(idxes_1, c1), ..., (idxes_n, cn)]
        def f(Y):
            idxes = range(len(Y))
            color = "red"
            return [(idxes, color)]
        scatter_rule = f

    pair_list = scatter_rule(self.Y)
    for pair in pair_list:
        idxes, color = pair
        if idxes is not None:
            x1, x2 = [[self.X[idx][0][i] for idx in idxes] for i in range(2)]
            ax.scatter(x1, x2, c = color)

