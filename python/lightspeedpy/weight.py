import numpy as np
from scipy.special import factorial
from .qe import QuantumEfficiency, MAX_D
from .util import EnormousArray

def fast_pinv(m):
    if len(m.shape) == 1:
        return m / np.sum(m**2)
    if np.all(np.sum(m != 0, axis=1) == 1):
        # There's a simplification for calculating the MPI
        weights_pinv = np.copy(np.transpose(m))
        weights_pinv[weights_pinv != 0] = 1/np.sum(m)
        return weights_pinv
    return np.linalg.pinv(m)


class Weighter:
    """
    A class to perform weighted analyses. After initialization, add pixels using the add_pixels method and perform the fit using get_fluxes.

    Parameters
    ----------
    data_set : DataSet
        Data set from which the data was collected
    max_n : int, optional
        Max numbers of electrons per pixel to model (default: 3)
    one_to_one : bool, optional
        Set to True to guarantee that the weights you are using are everywhere zero except at one index, at which the weight is 1, and that index is unique. If you set this flag, then the weights you pass to add_pixels needs to be the index of the non-zero item.
    """
    def __init__(self, data_set, n_outputs, max_n=3, one_to_one=False):
        self.weights_list = EnormousArray(5_000_000) # Stores the w_{ai} matrix. Shape: a, i
        self.probs_list = EnormousArray(5_000_000) # Shape: a, max_n
        self.qe = QuantumEfficiency()
        self.ns = np.arange(max_n)
        self.pixel_properties = data_set.get_pixel_properties(True)
        self.initial = np.zeros(n_outputs)
        self.data_set_n_frames = data_set.num_frames()

        self.one_to_one = one_to_one

    def finish(self):
        self.weights_list.finish()
        self.probs_list.finish()

    def add_pixels(self, image, weights, mask=None):
        """
        Add pixels to the fit.
        
        Parameters
        ----------
        image : array-like (a,)
            The frame to add. If you are using a mask, ensure this array is masked. This array must be 1-D
        weights : array-like (a, i,)
            Weight matrix that connect the observed flux to the parameters. If one_to_one is set, then weights is a list of length a corresponding containing the flux index constrained by a.
        mask : array-like
            The mask used to make the image
        """

        # Calculate the noise probabilities
        all_probs = np.array([self.pixel_properties.get_prob(image, n, mask) for n in self.ns]).transpose()

        # Add each pixel individually
        if len(weights.shape) == len(all_probs.shape):
            w_iter = weights.reshape(-1, weights.shape[-1])
        else:
            w_iter = weights.reshape(-1)

        for probs, weight in zip(all_probs.reshape(-1, all_probs.shape[-1]), w_iter):
            self.probs_list.append(probs)
            self.weights_list.append(weight)
            qe_corrected_image = self.qe.get_inverse(image)
            qe_corrected_image[np.isnan(qe_corrected_image)] = 0

            if self.one_to_one:
                np.add.at(self.initial, weights, qe_corrected_image)
            else:
                self.initial += np.einsum("ia,a->i", fast_pinv(weights), qe_corrected_image)

    def get_fluxes(self):
        fluxes = self.initial / self.data_set_n_frames
        fluxes -= np.min(fluxes)

        # Check boundaries
        for chunk_probs, chunk_weights in zip(self.probs_list, self.weights_list):
            if self.one_to_one:
                lambdas = fluxes[chunk_weights]
            else:
                lambdas = np.einsum("ai,i->a", chunk_weights, fluxes)
            if np.nanmin(lambdas) < 0:
                normal = -np.minimum(lambdas, 0)
                normal /= np.sqrt(np.sum(normal**2))
                fluxes -= fast_pinv(chunk_weights) @ normal * (normal @ lambdas)

        # Perform the iterations
        print("Beginning weight iterations")
        for iteration in range(10):
            gradient = np.zeros(len(fluxes))
            if self.one_to_one:
                hessian = np.zeros((len(fluxes)))
            else:
                hessian = np.zeros((len(fluxes), len(fluxes)))

            for chunk_probs, chunk_weights in zip(self.probs_list, self.weights_list):
                if self.one_to_one:
                    lambdas = fluxes[chunk_weights]
                else:
                    lambdas = np.einsum("ai,i->a", chunk_weights, fluxes)

                sum_d0 = np.zeros_like(lambdas)
                sum_d1 = np.zeros_like(lambdas)
                sum_d2 = np.zeros_like(lambdas)
                for n in self.ns:
                    d_lambda = lambdas - self.qe.get_inverse(n)
                    for k in range(0, MAX_D+1):
                        prefactor = d_lambda**k / factorial(k)
                        sum_d0 += chunk_probs[:,n] * prefactor * self.qe.get_d(k, n)
                        sum_d1 += chunk_probs[:,n] * prefactor * self.qe.get_d(k+1, n)
                        sum_d2 += chunk_probs[:,n] * prefactor * self.qe.get_d(k+2, n)
                grad_summand = sum_d1/sum_d0
                hess_summand = sum_d2/sum_d0 - grad_summand**2
                bad_mask = (~np.isfinite(sum_d0)) | (sum_d0 == 0)

                grad_summand[bad_mask] = 0
                hess_summand[bad_mask] = 0

                if self.one_to_one:
                    gradient[chunk_weights] += grad_summand
                    hessian[chunk_weights] += hess_summand
                else:
                    gradient += np.einsum("ai,a->i", chunk_weights, grad_summand)
                    hessian += np.einsum("ai,aj,a->ij", chunk_weights, chunk_weights, hess_summand)


            if self.one_to_one:
                inverse_hessian = np.diag(1/hessian)
            else:
                inverse_hessian = np.linalg.inv(hessian)

            old_fluxes = np.copy(fluxes)
            fluxes -= inverse_hessian @ gradient
            print(fluxes)

            # Check boundaries
            for chunk_probs, chunk_weights in zip(self.probs_list, self.weights_list):
                if self.one_to_one:
                    lambdas = fluxes[chunk_weights]
                else:
                    lambdas = np.einsum("ai,i->a", chunk_weights, fluxes)
                if np.nanmin(lambdas) < 0:
                    normal = -np.minimum(lambdas, 0)
                    normal /= np.sqrt(np.sum(normal**2))
                    fluxes -= fast_pinv(chunk_weights) @ normal * (normal @ lambdas)

            fractional_shift = np.sqrt(np.nanmean((fluxes - old_fluxes)**2)) / np.abs(np.nanmean(fluxes))
            print(f"Iteration {iteration+1}: fractional shift of {fractional_shift*100:.2f}%")
            if fractional_shift < 0.01: break

        return fluxes