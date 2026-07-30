import numpy as np
from scipy.special import factorial
from .qe import QuantumEfficiency, MAX_D
from .util import EnormousArray

MAX_ARRAY_SIZE = 5_000_000

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
    def __init__(self, data_set, n_outputs, max_n=3, blur=True):
        self.weights_list = EnormousArray(MAX_ARRAY_SIZE) # Stores the w_{ai} matrix. Shape: a, i
        self.probs_list = EnormousArray(MAX_ARRAY_SIZE) # Shape: a, max_n
        self.qe = QuantumEfficiency()
        self.epsilons = np.arange(max_n)
        self.pixel_properties = data_set.get_pixel_properties(True)
        self.initial = np.zeros(n_outputs)
        self.data_set_n_frames = data_set.num_frames()
        self.n_outputs = n_outputs

        self.blur = blur

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
            cpy = np.copy(weights)
            cpy[:,1] = 1/cpy[:,1]
            return cpy
        
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
        self.initial += np.nanmean(image)

    def get_fluxes(self):
        fluxes = self.initial / self.data_set_n_frames
        fluxes = np.maximum(fluxes, 100) # TODO

        # Perform the iterations
        print("Beginning weight iterations")
        for iteration in range(10):
            fluxes = np.maximum(fluxes, 0)

            gradient = np.zeros(len(fluxes))
            if self.blur:
                hessian = np.zeros((len(fluxes)))
            else:
                hessian = np.zeros((len(fluxes), len(fluxes)))

            for chunk_probs, chunk_weights in zip(self.probs_list, self.weights_list):
                lambdas = self.multiply(chunk_weights, fluxes)
                print(np.min(lambdas), np.max(lambdas))

                sum_d0 = np.zeros_like(lambdas)
                sum_d1 = np.zeros_like(lambdas)
                sum_d2 = np.zeros_like(lambdas)
                for epsilon in self.epsilons:
                    d_lambda = lambdas - self.qe.get_inverse(epsilon)
                    for n in range(0, MAX_D+1):
                        prefactor = d_lambda**n / factorial(n)
                        sum_d0 += chunk_probs[:,epsilon] * prefactor * self.qe.get_d(n, epsilon)
                        sum_d1 += chunk_probs[:,epsilon] * prefactor * self.qe.get_d(n+1, epsilon)
                        sum_d2 += chunk_probs[:,epsilon] * prefactor * self.qe.get_d(n+2, epsilon)
                bad_mask = (~np.isfinite(sum_d0)) | (sum_d0 == 0)
                grad_summand = sum_d1/sum_d0
                hess_summand = sum_d2/sum_d0 - grad_summand**2
                grad_summand[bad_mask] = 0
                hess_summand[bad_mask] = 0

                gradient += self.reverse_multiply(chunk_weights, grad_summand)
                hessian += self.reverse_multiply_2(chunk_weights, hess_summand)

            if self.blur:
                inverse_hessian = np.linalg.inv(hessian)
            else:
                inverse_hessian = np.diag(1/np.diagonal(hessian))

            old_fluxes = np.copy(fluxes)
            fluxes -= inverse_hessian @ gradient

            fluxes = self.check_boundaries(fluxes)

            fractional_shift = np.sqrt(np.nanmean((fluxes - old_fluxes)**2)) / np.abs(np.nanmean(fluxes))
            print(f"Iteration {iteration+1}: fractional shift of {fractional_shift*100:.2f}%")
            if fractional_shift < 0.01: break

        return fluxes

    def check_boundaries(self, fluxes):
        normals = EnormousArray(MAX_ARRAY_SIZE)
        min_lambda = 0
        max_lambda = self.epsilons[-1]
        norm_dot_norm = 0
        lambda_dot_norm = 0
        normal_d = 0
        for chunk_weights in self.weights_list:
            lambdas = self.multiply(chunk_weights, fluxes)
            minimum_normal = -np.minimum(lambdas, min_lambda)
            maximum_normal = max_lambda - np.maximum(lambdas, max_lambda)
            normal = minimum_normal + maximum_normal
            normal_d += np.sum(max_lambda * maximum_normal)
            norm_dot_norm += np.sum(normal**2)
            lambda_dot_norm += normal @ lambdas
            normals.concatenate(normal)
        if norm_dot_norm == 0:
            return fluxes
        
        alpha = (normal_d - lambda_dot_norm) / norm_dot_norm
        shift = np.zeros_like(fluxes)
        for chunk_weights, chunk_normal in zip(self.weights_list, normals):
            shift += self.reverse_multiply(self.pinv(chunk_weights), alpha * chunk_normal)
        return fluxes + shift