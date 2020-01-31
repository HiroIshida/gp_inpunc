import scipy.stats as sst
import numpy as np
import numpy.linalg as la
from gp import GaussianProcess
import math

# for analiticall tractability I used only Cumurative gaussian as a likelihood function
class GPClassifier:

    def __init__(self, gp):
        self.gp = gp
        self._construct_f_mode() # NOTE set member variables there 

    from plot.plot_gpc import show, _show2d, _show3d

    def _construct_f_mode(self):
        # called only once in constructor
        f = np.zeros(self.gp.n_train)

        for i in range(10):
            hessian_diag = self._compute_hessian_diag(f)
            W = np.diag(- hessian_diag)
            W_half = np.diag(np.sqrt(- hessian_diag ))
            I = np.eye(self.gp.n_train)
            L = la.cholesky(I + W_half.dot(self.gp.K).dot(W_half))
            grad = self._compute_gradient(f)
            b = W.dot(f) + grad
            a = b - la.inv(W_half.dot(L.T)).dot( la.inv(L).dot(W_half.dot(self.gp.K).dot(b)))
            f = self.gp.K.dot(a)

        self.f_mode = f
        self.grad_mode = grad
        self.hessian_diag_mode = hessian_diag
        self.L_mode = L

    def predict(self, x):
        hessian_diag = self.hessian_diag_mode
        W = np.diag(-hessian_diag)
        W_half = np.diag(np.sqrt(- hessian_diag))
        L = self.L_mode
        k_s = self.gp._construct_K_s(x)

        f_bar= k_s.T.dot(self.grad_mode) 

        v = la.inv(L).dot(W_half.dot(k_s))
        V_var = self.gp.kernel.k(x, x) - la.norm(v) ** 2

        probit = sst.norm.cdf(f_bar/math.sqrt(1 + V_var))
        return probit, f_bar, V_var
        
    def _compute_gradient(self, f):
        yf = self.gp.Y * f
        pdf_f = sst.norm.pdf(f)
        cdf_yf = sst.norm.cdf(yf)
        return self.gp.Y * pdf_f / cdf_yf

    def _compute_hessian_diag(self, f):
        # f is a vector of latent function evaluations
        diag_elements = np.zeros(self.gp.n_train)

        # see: eq 3.16
        yf = self.gp.Y * f
        pdf_f = sst.norm.pdf(f)
        cdf_yf = sst.norm.cdf(yf)

        hessian_diag = (- (pdf_f**2)/(cdf_yf**2) - yf*pdf_f/cdf_yf)
        return hessian_diag
    
