import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.special import binom, factorial
import tqdm

QE_LOCATION = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "qe.csv"))
P_EPSILON_GAMMA_LOCATION = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "tmp", "p_epsilon_gamma.npy"))
MAX_ELECTRONS = 8200
MAX_CALC_N = 100
TRAP_N = 4
TRAP_P = 0.11

class QuantumEfficiency:
    """Class to store information about the QE"""
    def __init__(self):
        data = np.loadtxt(QE_LOCATION, delimiter=',')
        self.qe_interp = interp1d(data[:,0], data[:,1], fill_value=(data[0,1], data[-1,1]), bounds_error=False)
        self.p_epsilon_gamma = self._get_p_epsilon_gamma()
        self.lambda_interp = self._get_lambda_interp()

    def __call__(self, n):
        return self.qe_interp(n)

    def _get_p_epsilon_gamma(self):
        if not os.path.exists(P_EPSILON_GAMMA_LOCATION):
            p_epsilon_gamma = np.zeros((MAX_ELECTRONS, MAX_ELECTRONS))
            for gamma in tqdm.tqdm(range(MAX_CALC_N), colour="green"):
                n_caught = np.zeros(TRAP_N+1)
                n_trial = 50_000
                sites_used = np.zeros((n_trial, TRAP_N), bool)
                for _ in range(gamma):
                    catch = (np.random.random((n_trial, TRAP_N)) < TRAP_P) & (~sites_used)
                    photoelectron_caught = np.any(catch, axis=1)
                    trap_index = np.argmax(catch, axis=1)[photoelectron_caught]
                    sites_used[np.where(photoelectron_caught)[0], trap_index] = True
                n_caught = np.sum(sites_used, axis=1)
                catch_prob = np.histogram(n_caught, np.arange(TRAP_N+2)-0.5, density=True)[0]
                for n in range(TRAP_N + 1):
                    p_epsilon_gamma[gamma-n, gamma] = catch_prob[n]

            for gamma in range(MAX_CALC_N, MAX_ELECTRONS):
                p_epsilon_gamma[gamma - TRAP_N, gamma] = 1

            np.save(P_EPSILON_GAMMA_LOCATION, p_epsilon_gamma)
        return np.load(P_EPSILON_GAMMA_LOCATION)
    
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
    
    def _get_lambda_interp(self):
        gammas = np.arange(MAX_CALC_N)
        values = [0]
        
        for n in range(1, MAX_CALC_N//2):
            l = n / self(n)
            p_epsilon_gamma = self.p_epsilon_gamma[n,:len(gammas)]
            for iteration in range(8):
                d1 = p_epsilon_gamma @ self._p_gamma_lambda_deriv(l, gammas, 1)
                d2 = p_epsilon_gamma @ self._p_gamma_lambda_deriv(l, gammas, 2)
                l -= d1 / d2
                l = max(l, 0)
            values.append(l)
        for n in range(MAX_CALC_N//2, MAX_ELECTRONS):
            values.append(n / self(n))
        ns = np.arange(0, MAX_ELECTRONS).astype(np.float64)
        return interp1d(ns, values, bounds_error=False)

    def _p_gamma_lambda_deriv(self, lamb, gamma, k):
        """
        Get the kth derivative of the Poisson distribution wrt lambda evaluated at gamma counts. gamma is a vector, but not lamb and k.
        """
        gammas, js = np.meshgrid(gamma, np.arange(k+1), indexing="ij")
        values = binom(k, js) * (-1)**(k-js) * lamb**(gammas-js) / factorial(gammas-js) * np.exp(-lamb)
        values[js>gammas] = 0
        return np.sum(values, axis=1)