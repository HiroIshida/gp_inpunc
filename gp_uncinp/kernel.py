import numpy as np
from numpy.random import rand
import warnings
import matplotlib.pyplot as plt
from numpy.linalg import inv, det
import numpy.linalg as la
from numpy import trace
from math import *
from copy import deepcopy

class Kernel(object):
    def __init__(self, dim, l = 1.0, noise = 0.01, N = 100, aniso = None):
        # aniso (list) is anisotropic factor 
        params = [log(l), log(noise)]
        self.set_param(params)
        self.n_params = len(params)
        self.N = N # mc sample number
        self.dim = dim

        if aniso is None:
            self.aniso = np.ones(dim)
        else:
            if dim > 1:
                self.aniso = np.array(aniso)
            else:
                error(" ")

        ## pre-generate random vector for a faster online computation
        if dim == 1:
            self.rns1 = np.random.randn(N)
            self.rns2 = np.random.randn(N)
        else:
            self.rns1 = np.random.randn(dim, N)
            self.rns2 = np.random.randn(dim, N)

    def set_param(self, params):
        self.params = params # is not used inside 
        self.l = exp(params[0])
        self.noise = exp(params[1])

    def _format_input(self, x):
        withUnc = (type(x) == tuple)
        if withUnc:
            return x
        cov = np.zeros((self.dim, self.dim))
        return (x, cov)

    def k(self, x1, x2, with_grad = False):
        mu1, cov1 = self._format_input(x1)
        mu2, cov2 = self._format_input(x2)

        boolean = np.array_equal(mu1, mu2)


        if self.dim == 1:
            pts1 = sqrt(cov1)*self.rns1 + mu1
            pts2 = sqrt(cov2)*self.rns2 + mu2 
            dif = pts1 - pts2
            r_vec = np.abs(dif)
        else:
            if np.sum(cov1) == 0.0:
                pts1 = np.tile(mu1, [self.N, 1]).T
            else:
                L1 = np.linalg.cholesky(cov1)
                pts1 = np.dot(L1, self.rns1) + np.tile(mu1, [self.N, 1]).T

            if np.sum(cov2) == 0.0:
                pts2 = np.tile(mu2, [self.N, 1]).T
            else:
                L2 = np.linalg.cholesky(cov2)
                pts2 = np.dot(L2, self.rns2) + np.tile(mu2, [self.N, 1]).T

            dif = pts1 - pts2
            # add anisotopic factor
            for i in range(self.dim):
                dif[i, :] /= self.aniso[i]
            r_vec = np.sqrt(np.sum(dif**2, axis = 0))

        kern, exps = self._k(r_vec)

        if not with_grad:
            return kern

        left = self._left(r_vec)
        grad0 = np.mean(left * exps)
        grad1 = self.noise * boolean
        kern_grad = [grad0, grad1]
        return kern, kern_grad

    def _k(self, r_vec):
        raise NotImplementedError()

    def _left(self, r_vec):
        raise NotImplementedError()

class Matern23(Kernel):
    def __init__(self, dim, l = 1.0, noise = 0.01, N = 100, aniso = None):
        super(Matern23, self).__init__(dim, l = l, noise = noise, N = N, aniso = aniso)

    def _k(self, r_vec):
        exps = np.exp(-sqrt(3)*r_vec/self.l)
        kern_vector =  (1 + sqrt(3)*r_vec/self.l) * exps
        kern = np.mean(kern_vector)
        return kern, exps

    def _left(self, r_vec):
        left = -sqrt(3)*r_vec/self.l**2 + (1 + sqrt(3)*r_vec/self.l) * (sqrt(3)*r_vec/self.l**2)
        return left

class RBF(Kernel):
    def __init__(self, dim, l = 1.0, noise = 0.01, N = 100, aniso = None):
        super(RBF, self).__init__(dim, l = l, noise = noise, N = N, aniso = aniso)

    def _k(self, r_vec):
        exps = np.exp(- r_vec ** 2/ (2 * self.l **2))
        kern_vector = exps
        kern = np.mean(kern_vector)
        return kern, exps

    def _left(self, r_vec):
        left = r_vec**2 / (self.l ** 3)
        return left





        





