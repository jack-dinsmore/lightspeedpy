import numpy as np
from scipy.special import factorial, binom
from .qe import QuantumEfficiency, MAX_D
from .constants import ADU_PER_ELECTRON

class PixelLayout:
    """
    The weighter stores count histograms for all data expected to have the same flux and noise properties. For imaging, that means it stores one histogram per pixel. For light curves, it stores one histogram per pixel per bin. This class defines the layout used for a given fit.

    Parameters
    ----------
    data_set : DataSet
        Data set from which the data was collected
    pixel_indices : array-like
        An array containing the pixel index of each histogram
    """
    def __init__(self, data_set, pixel_indices, mask=None):
        self.pixel_indices = pixel_indices
        self.pixel_properties = data_set.get_pixel_properties(True)
        self.mask = mask
        self.n_pixels = np.prod(data_set.image_shape) if mask is None else np.sum(mask)

    def image(data_set, mask=None):
        """
        Create the pixel layout for an imaging fit

        Parameters
        ----------
        data_set : DataSet
            Data set from which the data was collected
        mask : array-like (optional)
            Mask indicating the section of the detector used. If not provided, the fitter will treat every pixel in the frame.
        """
        n_pixels = np.prod(data_set.image_shape) if mask is None else np.sum(mask)
        return PixelLayout(data_set, np.arange(n_pixels), mask)

    def light_curve(data_set, n_bins, mask=None):
        """
        Create the pixel layout for a light curve fit

        Parameters
        ----------
        data_set : DataSet
            Data set from which the data was collected
        n_bins : int
            Number of light curve bins
        mask : array-like (optional)
            Mask indicating the section of the detector used. If not provided, the fitter will treat every pixel in the frame.
        """
        n_pixels = np.prod(data_set.image_shape) if mask is None else np.sum(mask)
        return PixelLayout(data_set, np.concatenate([np.arange(n_pixels)] * n_bins), mask)

    def n_histograms(self):
        return len(self.pixel_indices)

    def _get_p_u_epsilon(self, u_bins, epsilons):
        pixel_probs = np.zeros((self.n_pixels, len(u_bins)-1, len(epsilons)))
        image = np.ones(self.n_pixels)

        u_centers = (u_bins[:-1] + u_bins[1:]) / 2
        for (i, u) in enumerate(u_centers):
            for (j, epsilon) in enumerate(epsilons):
                pixel_probs[:,i,j] = self.pixel_properties.get_prob(image * u, epsilon, self.mask)
        pixel_probs = np.einsum("aue,ae->aue", pixel_probs, 1/np.sum(pixel_probs, axis=1))

        output = np.zeros((len(self.pixel_indices), len(u_bins)-1, len(epsilons)))
        for (i, pixel_index) in enumerate(self.pixel_indices):
            output[i] = pixel_probs[pixel_index]

        return output

class WeighterChiSquared:
    """
    A class to perform weighted analyses. After initialization, add pixels using the add_pixels method and perform the fit using get_fluxes.

    Parameters
    ----------
    data_set : DataSet
        Data set from which the data was collected
    max_electrons : int
        Max numbers of electrons per pixel to model
    weight_matrix: array-like
        The pixel weights to apply. The weight matrix M is defined such that lambda = M f, where f is the vector of fluxes which will be returned by this fitter and lambda is the incident photon rate for each pixel. Note: If you provide a 1D array, it will be assumed that you meant a 2D array where the second axis had length 1. If your weight_matrix smaller than the number of pixels listed in the pixel layout, it will be assumed that you provided a per-pixel weight which you want to be the same for all pixels regardless of the time bin.
    """
    def __init__(self, pixel_layout, max_electrons, weight_matrix):
        self.pixel_layout = pixel_layout
        self.u_edges = np.arange(-2, max_electrons, 1/ADU_PER_ELECTRON)
        self.histograms = np.zeros((pixel_layout.n_histograms(), len(self.u_edges)-1))
        self.max_electrons = max_electrons
        self.prep_fit(max_electrons, weight_matrix)

    def prep_fit(self, max_electrons, weight_matrix):
        self.epsilons = np.arange(max_electrons+1)
        self.p_u_epsilons = self.pixel_layout._get_p_u_epsilon(self.u_edges, self.epsilons)

        # Make the weight matrix
        if weight_matrix.ndim == 1:
            weight_matrix = weight_matrix.reshape(-1,1)
        if weight_matrix.shape[0] == self.pixel_layout.n_histograms():
            self.weight_matrix = weight_matrix
        else:
            # Assume this is a light curve, so the pixel layout should just be duplicated for each bin.
            n_bins = self.pixel_layout.n_histograms() // weight_matrix.shape[0]
            self.weight_matrix = np.zeros((self.pixel_layout.n_histograms(), n_bins))
            for i in range(n_bins):
                self.weight_matrix[self.pixel_layout.n_pixels*i:self.pixel_layout.n_pixels*(i + 1), i] = weight_matrix[:,0]

        # Make the product of p_epsilon_gamma with p_epsilon_gamma_prime
        qe = QuantumEfficiency()
        p_epsilon_gamma = qe.p_epsilon_gamma[:max_electrons+1, :max_electrons+1]
        gamma, gamma_prime = np.meshgrid(self.epsilons, self.epsilons, indexing="ij")
        self.p_epsilon_gamma_primes = []
        for k in range(3):
            m_gamma_gamma_prime = (-1)**(k + gamma + gamma_prime) * binom(k, gamma - gamma_prime)
            m_gamma_gamma_prime[gamma_prime > gamma] = 0
            m_gamma_gamma_prime[gamma_prime < gamma-k] = 0
            self.p_epsilon_gamma_primes.append(p_epsilon_gamma @ m_gamma_gamma_prime)

    def clear(self):
        """
        Clear the weighter of all previous data
        """
        self.histograms *= 0

    def add_pixels(self, image, time_index=0):
        """
        Add some pixels to the fit.
        
        Parameters
        ----------
        image : array-like (a,)
            The frame to add. If you are using a mask, ensure this array is masked. This array must be 1-D
        mask : array-like
            The mask used to make the image
        time_index: int, optional
            For light curves, this argument provides index of the relevant light curve bin.
        """

        histogram_indices = np.arange(len(image))
        bin_indices = np.digitize(image, self.u_edges)
        acceptable_mask = (bin_indices >= 1) & (bin_indices < len(self.u_edges))
        histogram_indices = histogram_indices[acceptable_mask]
        histogram_indices += self.pixel_layout.n_pixels * time_index
        bin_indices = bin_indices[acceptable_mask] - 1
        self.histograms[histogram_indices, bin_indices] += 1

    def get_fluxes(self, n_iterations=10):
        self.histograms = np.einsum("au,a->au", self.histograms, 1/np.sum(self.histograms, axis=1))
        estimated_pixel_fluxes = np.sum(self.histograms * (self.u_edges[1:] + self.u_edges[:-1])/2, axis=1)
        fluxes = np.linalg.pinv(self.weight_matrix) @ estimated_pixel_fluxes

        for iteration in range(n_iterations):
            fluxes = np.maximum(fluxes, 1e-5)
            fluxes[np.isnan(fluxes)] = 1

            lambdas = self.weight_matrix @ fluxes
            gamma_grid, lambda_grid = np.meshgrid(self.epsilons, lambdas, indexing="ij")
            p_gamma_lambdas = lambda_grid**gamma_grid / factorial(gamma_grid)*np.exp(-lambdas)

            # Get the likelihood derivatives at each bin position
            d0 = np.einsum("aue,eg,ga->au", self.p_u_epsilons, self.p_epsilon_gamma_primes[0], p_gamma_lambdas)
            d1 = np.einsum("aue,eg,ga->au", self.p_u_epsilons, self.p_epsilon_gamma_primes[1], p_gamma_lambdas)
            d2 = np.einsum("aue,eg,ga->au", self.p_u_epsilons, self.p_epsilon_gamma_primes[2], p_gamma_lambdas)

            bad_mask = (d0 == 0) | np.isnan(d0)
            d0[bad_mask] = np.nan
            d1[bad_mask] = np.nan
            d2[bad_mask] = np.nan

            grad_summand = np.nansum(self.histograms * (d1 / d0) - d1, axis=1)
            hess_summand = np.nansum(self.histograms * (d2 / d0 - d1**2 / d0**2) - d2, axis=1)
            gradient = np.einsum("ai,a->i", self.weight_matrix, grad_summand)
            hessian = np.einsum("ai,aj,a->ij", self.weight_matrix, self.weight_matrix, hess_summand)
            inverse_hessian = np.linalg.inv(hessian)
            
            old_fluxes = np.copy(fluxes)
            fluxes -= inverse_hessian @ gradient
            fractional_shift = np.sqrt(np.nanmean((fluxes - old_fluxes)**2)) / np.abs(np.nanmean(old_fluxes))
            print(f"Iteration {iteration+1}: fractional shift of {fractional_shift*100:.2f}%")
            if fractional_shift < 0.01:
                break

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.step(np.arange(len(fluxes)), fluxes)
            fig.savefig("fluxes.png")
            plt.close()

        return fluxes
