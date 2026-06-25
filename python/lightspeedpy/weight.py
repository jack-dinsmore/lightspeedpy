import numpy as np
from scipy.special import factorial
from .qe import QuantumEfficiency, MAX_D
from .util import EnormousArray


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
    def __init__(self, data_set, max_n=3, one_to_one=False):
        self.weights_list = EnormousArray(5_000_000) # Stores the w_{ai} matrix. Shape: a, i
        self.probs_list = EnormousArray(5_000_000) # Shape: a, max_n
        self.qe = QuantumEfficiency()
        self.ns = np.arange(max_n)
        self.pixel_properties = data_set.get_pixel_properties(True)
        self.initial = None

        self.one_to_one = one_to_one

    def add_pixels(self, image, weights, mask=None):
        """
        Add pixels to the fit.
        
        Parameters
        ----------
        image : array-like (a,)
            The frame to add. If you are using a mask, ensure this array is masked.
        weights : array-like (a, i,)
            Weight matrix that connect the observed flux to the parameters. If one_to_one is set, then weights is a list of length a corresponding containing the flux index constrained by a.
        mask : array-like
            The mask used to make the image
        """

        # Calculate the noise probabilities
        all_probs = np.array([self.pixel_properties.get_prob(image, n, mask) for n in self.ns])

        # Add each pixel individually
        for probs, weight in zip(all_probs.reshape(len(self.ns), -1), weights.reshape(-1, weights.shape[-1])):
            self.probs_list.append(probs)
            self.weights_list.append(weight)
            qe_corrected_image = self.qe.get_inverse(image)

            if self.one_to_one:
                new_lambda_estimate = qe_corrected_image[weights]
            else:
                new_lambda_estimate = np.einsum("ia,a->i", np.linalg.pinv(weights), qe_corrected_image)
            if self.initial is None:
                self.initial = new_lambda_estimate 
            else:
                self.initial += new_lambda_estimate

    def get_fluxes(self):
        fluxes = self.initial / self.qe.get_inverse(1)

        # Perform the iterations
        fractional_shift = 1
        print("Beginning weight iterations")
        for iteration in range(10):
            gradient = np.zeros(len(fluxes))
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
                    for k in range(0, MAX_D):
                        prefactor = d_lambda**k / factorial(k)
                        sum_d0 += chunk_probs[:,n] * prefactor * self.qe.get_d(k, n)
                        sum_d1 += chunk_probs[:,n] * prefactor * self.qe.get_d(k+1, n)
                        sum_d2 += chunk_probs[:,n] * prefactor * self.qe.get_d(k+2, n)
                d1_d0_ratio = sum_d1/sum_d0
                if self.one_to_one:
                    gradient = np.zeros(len(fluxes))
                    gradient[chunk_weights] = d1_d0_ratio
                    hessian = np.zeros((len(fluxes)))
                    hessian[chunk_weights] = sum_d2/sum_d0 - d1_d0_ratio**2
                    inverse_hessian = np.diag(1/hessian)
                else:
                    gradient = np.einsum("ai,a->i", chunk_weights, d1_d0_ratio)
                    hessian = np.einsum("ai,aj,a->i", chunk_weights, chunk_weights, sum_d2/sum_d0 - d1_d0_ratio**2)
                    inverse_hessian = np.linalg.inv(hessian)
            shift = -inverse_hessian @ gradient
            fluxes += shift
            fractional_shift = np.sqrt(np.mean(shift**2)) / np.abs(np.mean(fluxes))
            print(f"Iteration {iteration+1}: fractional shift of {fractional_shift*100:.2f}%")
            if fractional_shift < 0.01: break

        return fluxes