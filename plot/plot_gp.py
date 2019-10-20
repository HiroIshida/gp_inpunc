import numpy as np
import matplotlib.pyplot as plt
import math

def show(self, **kwargs):
    if self.dim == 1:
        self._show1d(**kwargs)
    else:
        raise NotImplementedError()

def _show1d(self, bmin = None, bmax = None, N_grid = 20):
    if bmin or bmax is None:
        bmin, bmax = self.get_boundary()

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





