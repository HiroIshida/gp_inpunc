import numpy as np
import warnings
from numpy.linalg import inv, det
from numpy import trace
from math import *
from time import time
from scipy.optimize import fmin_l_bfgs_b
from copy import deepcopy

class GaussianProcess:
    def __init__(self, X, Y, kernel):
        self.dim = kernel.dim
        self.kernel = kernel
        self.X = X
        self.Y = Y
        self.n_train = len(X)
        self.construct_matrixes()

    ## import methods
    from plot.plot_gp import show, _show1d

    def predict(self, x):

        withoutInpUnc = (type(x) is not tuple)
        if withoutInpUnc:
            x = (x, np.zeros((self.dim, self.dim)))

        K_s = self._construct_K_s(x)
        K_s_tr = K_s.transpose()
        K_ss = np.matrix(self.kernel.k(x, x))

        Y_vec = np.matrix(self.Y).transpose()
        y_mean = Y_vec.mean(axis = 0)

        gp_mean = (y_mean + np.dot(np.dot(K_s_tr, self.K_inv), Y_vec - y_mean)).item()
        gp_var = (K_ss - np.dot(np.dot(K_s_tr, self.K_inv), K_s)).item()
        if gp_var < 0:
            gp_var = 0
        return gp_mean, gp_var

    def _construct_K_s(self, x):
        K_s = np.zeros((self.n_train, 1))
        for i in range(self.n_train):
            x_i = self.X[i]
            K_s[i] = self.kernel.k(x_i, x)
        return K_s

    def optimize(self, params_min = None, params_max = None, dump_log = False):
        fun = lambda x: - self.compute_likelihood(x, params_min, params_max) # to maximize it via minimize function
        grad = lambda x: - self.compute_likelihood_grad(x) # to maximize it via minimize function
        tes = fmin_l_bfgs_b(fun, self.kernel.params, fprime=grad)

    def compute_likelihood_grad(self, params_ = None, params_min = None, params_max = None):
        K_inv_Y = self.K_inv.dot(np.matrix(self.Y).transpose())
        n_params = self.kernel.n_params

        grad = np.zeros(n_params)
        for n in range(n_params):
            K_grad = self.K_grad_lst[n]
            grad[n] = - trace(self.K_inv.dot(K_grad)) + K_inv_Y.transpose().dot(K_grad).dot(K_inv_Y)
        return grad

    def compute_likelihood(self, params_ = None, params_min = None, params_max = None):

        if params_ is not None:
            # when this function is called in the optimizer (e.g. scipy.optimize.fmin_l_bfgs()) ,
            # params_ is sets, which tells the kernel that parameters are changed.
            # Then reconstruct the covariance matrix again.
            #if GaussianProcess

            params = list(params_) # in case given in numpy.array

            def isValidParams():
                for i in range(len(params)):
                    if params_min is not None:
                        if params[i] < log(params_min[i]):
                            return False
                    if params_max is not None:
                        if params[i] > log(params_max[i]):
                            return False
                return True

            if isValidParams():
                self.kernel.set_param(params) 
                self.construct_matrixes()
            else:
                print [exp(p) for p in params]
                print "PARAMETERS HIT THE BOUND"


        Y_vec = np.matrix(self.Y).transpose()
        datafit = - Y_vec.transpose().dot(self.K_inv).dot(Y_vec).item()
        simpleness = - log(det(self.K))
        likelihood = datafit + simpleness
        return likelihood

    def construct_matrixes(self):
        # K and K_grad (derivative of K w.r.t hyperparameters theta) will be constructed and cached
        # the reason why constructed both K and K_grad in the same method is that 
        # some terms appears in the computation of K will also appear in K_grad, so 
        # if we devide this function into two, then we have to do the same computation for
        # some terms twice

        K = np.zeros((self.n_train, self.n_train))

        # note that gradient matrix of K must be constructed for each parameter; that's why the following is a list
        K_grad_lst = [np.zeros((self.n_train, self.n_train)) for i in range(self.kernel.n_params)]

        for i in range(self.n_train):
            for j in range(self.n_train):
                k, k_grad = self.kernel.k(self.X[i], self.X[j], with_grad = True)

                K[i][j] = k
                for n in range(self.kernel.n_params):
                    K_grad_lst[n][i][j] = k_grad[n]

        K_inv = inv(K)

        # Note that it is worth cacheng K_inv; computation of this takes time, and will be used entirely in this class
        self.K_inv = K_inv 
        self.K = K
        self.K_grad_lst = K_grad_lst

    def get_boundary(self, margin = 0.2):
        x_lst_lst = [[self.X[i][0][j] for i in range(self.n_train)] for j in range(self.dim)]
        bmin_ = np.array([min(x_lst) for x_lst in x_lst_lst])
        bmax_ = np.array([max(x_lst) for x_lst in x_lst_lst])
        dif = bmax_ - bmin_
        bmin = bmin_ - margin * dif
        bmax = bmax_ + margin * dif
        return bmin, bmax
