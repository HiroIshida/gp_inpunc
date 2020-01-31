import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import utils

def show(self, showType = "occ", **kwargs):

    def func(x):
        occ, f, var = self.predict(x)
        # return contour, contourf
        if showType == "occ":
            return occ, occ
        elif showType == "var":
            return occ, var
        else:
            raise NotImplementedError()

    if self.gp.dim == 2:
        self._show2d(func, **kwargs)
    elif self.gp.dim == 3:
        self._show3d(func, **kwargs)
    else:
        raise NotImplementedError()

def _show2d(self, func, bmin = None, bmax = None, margin = 0.2, levels = None):

    if (bmin is None) or (bmax is None):
        bmin, bmax = self.gp.get_boundary(margin = margin)

    fig, ax = plt.subplots() 
    fax = (fig, ax)
    utils.show2d(func, bmin, bmax, N = 30, fax = fax, levels = levels)  

    """
    def rule(Y):
        idxes_1 = []
        idxes_2 = []
        for idx in range(len(Y)):
            y = Y[idx]
            if y > 0.0:
                idxes_1.append(idx)
            else:
                idxes_2.append(idx)
        return [(idxes_1, "blue"), (idxes_2, "red")]

    utils.scatter(rule, self.gp.X, self.gp.Y, fax)
    """

def _show3d(self, bmin = None, bmax = None, margin = 0.2, N_grid = 20, levels = None,
        scatter_rule = None):

    if (bmin is None) or (bmax is None):
        bmin, bmax = self.gp.get_boundary(margin = margin)

    fig = plt.figure()
    ax = Axes3D(fig)
    fax = (fig, ax)

    def rule(Y):
        idxes_1 = []
        idxes_2 = []
        for idx in range(len(Y)):
            y = Y[idx]
            if y > 0.0:
                idxes_1.append(idx)
            else:
                idxes_2.append(idx)
        return [(idxes_1, "blue"), (idxes_2, "red")]

    utils.scatter3d(rule, self.gp.X, self.gp.Y, fax)







