import numpy as np
from scipy.special import factorial, binom
from .qe import QuantumEfficiency, MAX_D
from .util import EnormousArray

class Weighter:
    """
    A class to perform weighted analyses. After initialization, add pixels using the add_pixels method and perform the fit using get_fluxes.

    Parameters
    ----------
    data_set : DataSet
        Data set from which the data was collected
    n_outputs : int
        Number of flux values that are fitted for
    max_n : int, optional
        Max numbers of electrons per pixel to model (default: 3)
    blur : bool, optional
        Set to False to guarantee that each pixel contributes to only one flux value. If you set this flag, then the weights you pass to add_pixels needs to be a tuple of the flux indices and the weight value.
    """
    def __init__(self, data_set, n_outputs, max_n, blur=True):
        self.weights_list = EnormousArray() # Stores the w_{ai} matrix. Shape: a, i
        self.probs_list = EnormousArray() # Shape: a, max_n
        self.qe = QuantumEfficiency()
        self.epsilons = np.arange(max_n+1)
        self.pixel_properties = data_set.get_pixel_properties(True)
        self.fluxes = np.zeros(n_outputs)
        self.n_outputs = n_outputs
        self.blur = blur
        self.n_epochs_added = 0

        p_epsilon_gamma = self.qe.p_epsilon_gamma[:max_n+1, :max_n+1]
        gamma, gamma_prime = np.meshgrid(self.epsilons, self.epsilons, indexing="ij")
        self.p_epsilon_gamma_primes = []
        for k in range(3):
            m_gamma_gamma_prime = (-1)**(k + gamma + gamma_prime) * binom(k, gamma - gamma_prime)
            m_gamma_gamma_prime[gamma_prime > gamma] = 0
            m_gamma_gamma_prime[gamma_prime < gamma-k] = 0
            self.p_epsilon_gamma_primes.append(p_epsilon_gamma @ m_gamma_gamma_prime)

    def clear(self):
        self.weights_list.clear()
        self.probs_list.clear()
        self.fluxes *= 0
        self.n_epochs_added = 0

    def pinv(self, weights):
        if self.blur:
            if len(weights.shape) == 1:
                return weights / np.sum(weights**2)
            elif np.all(np.sum(weights != 0, axis=1) == 1):
                # There's a simplification for calculating the MPI
                weights_pinv = np.copy(weights)
                weights_pinv[weights_pinv != 0] = 1/np.sum(weights)
                return weights_pinv
            else:
                return np.transpose(np.linalg.pinv(weights))
        else:
            output = np.copy(weights)
            indices = weights[:,0].astype(int)
            denom = np.bincount(indices, weights=weights[:,1]**2, minlength=self.n_outputs)[indices]
            output[:,1] = np.where(denom > 0, weights[:,1] / denom, 0.0)
            return output
        
    def multiply(self, weights, fluxes):
        if self.blur:
            return np.einsum("ai,i->a", weights, fluxes)
        else:
            return fluxes[weights[:,0].astype(int)] * weights[:,1]
    def reverse_multiply(self, weights, lamb):
        if self.blur:
            return np.einsum("ai,a->i", weights, lamb)
        else:
            return np.bincount(weights[:,0].astype(int), weights=weights[:,1] * lamb, minlength=self.n_outputs)
    def reverse_multiply_2(self, weights, lamb):
        if self.blur:
            return np.einsum("ai,aj,a->ij", weights, weights, lamb)
        else:
            diag = np.bincount(weights[:,0].astype(int), weights=weights[:,1]**2 * lamb, minlength=self.n_outputs)
            return np.diag(diag)

    def add_pixels(self, image, weights, mask=None):
        """
        Add some pixels to the fit.
        
        Parameters
        ----------
        image : array-like (a,)
            The frame to add. If you are using a mask, ensure this array is masked. This array must be 1-D
        weights : array-like (a, i,) or tuple(j, w)
            Weight matrix that connect the observed flux to the parameters. If blur is False, then weights is a tuple of indices and the weight values
        mask : array-like
            The mask used to make the image
        """

        # Calculate the noise probabilities
        all_probs = np.array([self.pixel_properties.get_prob(image, n, mask) for n in self.epsilons]).transpose()

        self.probs_list.concatenate(all_probs)
        self.weights_list.concatenate(weights)

        self.weights_list.max_data_len = min(self.weights_list.max_data_len, self.probs_list.max_data_len)
        self.probs_list.max_data_len = self.weights_list.max_data_len

        self.fluxes = (self.fluxes * self.n_epochs_added + self.reverse_multiply(self.pinv(weights), image)) / (self.n_epochs_added + 1)
        self.n_epochs_added += 1

    def get_fluxes(self, n_iterations=10):
        self.fluxes = np.ones_like(self.fluxes)
        for iteration in range(n_iterations):
            frac_shift = self.iterate()
            print(f"Iteration {iteration+1}: fractional shift of {frac_shift*100:.2f}%")
            if frac_shift < 0.01:
                break

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.step(np.arange(len(self.fluxes)), self.fluxes)
            fig.savefig("fluxes.png")

        return self.fluxes

    def iterate(self):
        # Perform an iteration
        self.fluxes = np.maximum(self.fluxes, 1e-5)
        self.fluxes[np.isnan(self.fluxes)] = 1
        like = 0
        gradient = np.zeros(len(self.fluxes))
        hessian = np.zeros((len(self.fluxes), len(self.fluxes)))

        for chunk_probs, chunk_weights in zip(self.probs_list, self.weights_list):
            lambdas = self.multiply(chunk_weights, self.fluxes)
            gamma_grid, lambda_grid = np.meshgrid(self.epsilons, lambdas, indexing="ij")
            p_gamma_lambdas = lambda_grid**gamma_grid / factorial(gamma_grid)*np.exp(-lambdas)

            d0 = np.einsum("ax,xy,ya->a", chunk_probs, self.p_epsilon_gamma_primes[0], p_gamma_lambdas)
            d1 = np.einsum("ax,xy,ya->a", chunk_probs, self.p_epsilon_gamma_primes[1], p_gamma_lambdas)
            d2 = np.einsum("ax,xy,ya->a", chunk_probs, self.p_epsilon_gamma_primes[2], p_gamma_lambdas)

            bad_mask = (~np.isfinite(d0)) | (d0 == 0)
            grad_summand = d1/d0
            hess_summand = d2/d0 - grad_summand**2
            grad_summand[bad_mask] = 0
            hess_summand[bad_mask] = 0
            like += np.sum(np.log(d0[~bad_mask]))

            gradient += self.reverse_multiply(chunk_weights, grad_summand)
            hessian += self.reverse_multiply_2(chunk_weights, hess_summand)

        if self.blur:
            inverse_hessian = np.linalg.inv(hessian)
        else:
            inverse_hessian = np.diag(1/np.diagonal(hessian))

        old_fluxes = np.copy(self.fluxes)
        self.fluxes -= inverse_hessian @ gradient
        # self.fluxes = self.check_boundaries(self.fluxes)

        fractional_shift = np.sqrt(np.nanmean((self.fluxes - old_fluxes)**2)) / np.abs(np.nanmean(old_fluxes))
        return fractional_shift

    def check_boundaries(self, fluxes):
        min_lambda = 0
        max_lambda = self.epsilons[-1]
        shift = np.zeros_like(fluxes)
        n_chunks = 0
        for chunk_weights in self.weights_list:
            lambdas = self.multiply(chunk_weights, fluxes)
            normal = -np.minimum(lambdas, min_lambda)
            normal += max_lambda - np.maximum(lambdas, max_lambda)
            shift += self.reverse_multiply(self.pinv(chunk_weights), normal)
            n_chunks += 1
        shift /= n_chunks
        return fluxes + shift