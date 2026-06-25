import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.special import binom, factorial

QE_LOCATION = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "qe.csv"))
MAX_D = 4
MAX_ELECTRONS = 100#8200

class QuantumEfficiency:
    """Class to store information about the QE"""
    def __init__(self):
        data = np.loadtxt(QE_LOCATION, delimiter=',')
        self.interp = interp1d(data[:,0], data[:,1], fill_value=(data[0,1], data[-1,1]), bounds_error=False)

        self.lambda_interp = self._get_lambda_interp()

        self.d_interps = [self._get_d_interp(0)]
        for k in range(2, MAX_D+1):
            self.d_interps.append(self._get_d_interp(k))

    def __call__(self, n):
        return self.interp(n)
    
    def get_inverse(self, n):
        if type(n) is np.ndarray:
            neg_mask = n < 0
            lambdas = self.lambda_interp(n)
            lambdas[neg_mask] = self.lambda_interp(1) * n[neg_mask]
        else:
            if n < 0:
                return self.lambda_interp(1) * n
            else:
                return self.lambda_interp(n)
        return lambdas
    
    def get_d(self, k, n):
        if k == 0: return self.d_interps[0](n)
        if k == 1: return np.zeros_like(n)
        if k >= MAX_D: return np.zeros_like(n)
        return self.d_interps[k-1](n)
    
    def _get_lambda_interp(self):
        ns = np.arange(0, MAX_ELECTRONS).astype(np.float64)
        values = []
        for n in ns:
            l = n / self(n)
            ms = np.arange(n, MAX_ELECTRONS)
            coefficient = self(ms)**n * (1 - self(ms))**(ms-n) * binom(ms, n)
            for iteration in range(10):
                d1 = np.sum(self._get_poisson_deriv(l, ms, 1)*coefficient)
                d2 = np.sum(self._get_poisson_deriv(l, ms, 2)*coefficient)
                l -= d1 / d2
            values.append(l)
        return interp1d(ns, values, bounds_error=False)

    def _get_poisson_deriv(self, l, m, k):
        """
        Get the kth derivative of the Poisson distribution wrt lambda evaluated at m counts. m is a vector, but not l and k.
        """
        ms, js = np.meshgrid(m, np.arange(k+1), indexing="ij")
        values = binom(k, js) * l**(ms-js) * (-1)**(k-js) / factorial(ms-js) * np.exp(-l)
        values[js>ms] = 0
        return np.sum(values, axis=1)
    
    def _get_d_interp(self, k):
        ns = np.arange(0, MAX_ELECTRONS).astype(float)
        values = []
        for n in ns:
            l = self.lambda_interp(n)
            ms = np.arange(n, MAX_ELECTRONS)
            coefficient = self(ms)**n * (1 - self(ms))**(ms-n) * binom(ms, n)
            values.append(np.sum(self._get_poisson_deriv(l, ms, k)*coefficient))
        return interp1d(ns, values)