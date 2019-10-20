import numpy as np
import matplotlib.pyplot as plt

def show2d(func, bmin, bmax, N = 20, fax = None, levels = None):
    # fax is (fix, ax)
    # func: np.array([x1, x2]) list -> scalar
    # func cbar specifies the same height curve
    if fax is None:
        fig, ax = plt.subplots() 
    else:
        fig = fax[0]
        ax = fax[1]

    mat_contour_ = np.zeros((N, N))
    mat_contourf_ = np.zeros((N, N))
    x1_lin, x2_lin = [np.linspace(bmin[i], bmax[i], N) for i in range(bmin.size)]
    for i in range(N):
        for j in range(N):
            x = np.array([x1_lin[i], x2_lin[j]])
            val_c, val_cf = func(x)
            mat_contour_[i, j] = val_c
            mat_contourf_[i, j] = val_cf
    mat_contour = mat_contour_.T
    mat_contourf = mat_contourf_.T
    X, Y = np.meshgrid(x1_lin, x2_lin)

    cs = ax.contour(X, Y, mat_contour, levels = levels, cmap = 'jet')
    zc = cs.collections[0]
    plt.setp(zc, linewidth=4)
    ax.clabel(cs, fontsize=10)
    cf = ax.contourf(X, Y, mat_contourf, cmap = 'gray_r')
    fig.colorbar(cf)

def scatter(rule, X, Y, fax = None):
    if fax is None:
        fig, ax = plt.subplots() 
    else:
        fig = fax[0]
        ax = fax[1]

    dim = X[0][0].size
    pair_list = rule(Y)
    for pair in pair_list:
        idxes, color = pair
        if idxes is not None:
            x1, x2 = [[X[idx][0][i] for idx in idxes] for i in range(dim)]
            ax.scatter(x1, x2, c = color)

