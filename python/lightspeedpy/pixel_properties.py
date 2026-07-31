import numpy as np
import os, tqdm
from astropy.io import fits
from .util import trim_image
from .constants import FORBIDDEN_KEYWORDS, ADU_PER_ELECTRON

GRID_LOCATION = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "moments.npy"))

class PixelProperties:
    """
    Biases and noise of each pixel in the data set. Use :meth:`PixelProperties.default` or :meth:`PixelProperties.from_bias` to create it.
    
    Attributes
    ----------
    bias : array-like
        Image of biases of each pixel
    widths : array-like
        Noises in each pixel, defined as the standard deviation of the Gaussian error approximation.
    """
    def __init__(self, bias, widths, params, source_data_set, dest_data_set):
        self.params = params
        if source_data_set is not None:
            self.bias = trim_image(bias, source_data_set, dest_data_set)
            self.widths = trim_image(widths, source_data_set, dest_data_set)
            self.header0 = source_data_set.header0
            self.header1 = source_data_set.header1
        else:
            self.bias = bias
            self.widths = widths

    def has_noise_distro(self):
        return self.params is not None

    def crop(self, bbox):
        self.params = self.params[bbox[0]:bbox[1],bbox[2]:bbox[3],:]
        self.bias = self.bias[bbox[0]:bbox[1],bbox[2]:bbox[3]]
        self.widths = self.widths[bbox[0]:bbox[1],bbox[2]:bbox[3]]

    def save(self, filename, clobber):
        """
        Save the pixel properties to a file

        Parameters
        ----------
        filename : str
            Name of the output file
        """
        h0 = fits.PrimaryHDU()
        h1 = fits.ImageHDU(data=self.bias)
        h2 = fits.ImageHDU(data=self.widths)
        if self.params is not None:
            h3 = fits.ImageHDU(data=self.params)

        for key, value in self.header0.items():
            if key not in FORBIDDEN_KEYWORDS:
                if len(key) > 8: key = f"HIERARCH {key}"
                h0.header[key] = value

        for key, value in self.header1.items():
            if key not in FORBIDDEN_KEYWORDS:
                if len(key) > 8: key = f"HIERARCH {key}"
                h1.header[key] = value
                h2.header[key] = value
                if self.params is not None:
                    h3.header[key] = value

        h0.header["PIXPROP"] = "T"
        h1.header["PIXPROP"] = "T"
        h2.header["PIXPROP"] = "T"
        if self.params is not None:
            h3.header["PIXPROP"] = "T"

        hdul = [h0, h1, h2]
        if self.params is not None:
            hdul.append(h3)


        fits.HDUList(hdul).writeto(filename, overwrite=clobber)

    def load(filename):
        with fits.open(filename) as hdul:
            if "PIXPROP" not in hdul[1].header or hdul[1].header["PIXPROP"] != "T":
                raise Exception(f"The file {filename} is not a PixelProperties object")
            
            bias = np.array(hdul[1].data)
            widths = np.array(hdul[2].data)
            if len(hdul) == 4:
                params = np.array(hdul[3].data)
            else:
                params = None
            pp = PixelProperties(bias, widths, params, None, None)
            pp.header0 = hdul[0].header
            pp.header1 = hdul[1].header
        return pp
    
    def get_prob(self, image, true_n, mask=None):
        """
        Get the probability for the observed counts to have been produced given a true source count.
        
        Parameters
        ----------
        image : array-like
            Detected image (e.g. from frame.image)
        true_n : int
            True number of counts
        mask : array of bool, optional
            If you only wish to get probabilities for a subset of the full image, provide that subset as the iamge argument and supply the pixel mask here.
        
        Returns
        -------
            array-like
        The probability for each pixel to have originated from the given true source count.
        """

        if self.params is None:
            raise Exception("You cannot get a noise probability unless you first map the noise distribution")
        if mask is None:
            denom = 1 / (2*self.params[:,0]**2)
            pdf = np.exp(-(image-self.params[:,1] - true_n)**2 * denom) * self.params[:,2]
            pdf += np.exp(-(image-self.params[:,1]-self.params[:,3] - true_n)**2 * denom) * self.params[:,4]
            pdf += np.exp(-(image-self.params[:,1]+self.params[:,5] - true_n)**2 * denom) * self.params[:,6]
            pdf /= np.sqrt(2*np.pi*self.params[:,0]**2)
        else:
            denom = 1 / (2*self.params[mask,0]**2)
            pdf = np.exp(-(image-self.params[mask,1] - true_n)**2 * denom) * self.params[mask,2]
            pdf += np.exp(-(image-self.params[mask,1]-self.params[mask,3] - true_n)**2 * denom) * self.params[mask,4]
            pdf += np.exp(-(image-self.params[mask,1]+self.params[mask,5] - true_n)**2 * denom) * self.params[mask,6]
            pdf /= np.sqrt(2*np.pi*self.params[mask,0]**2)

        return pdf

    def default(data_set):
        """
        Get the default pixel properties for a data set with no bias.
        """
        return PixelProperties(
            np.zeros(data_set.image_shape),
            np.ones(data_set.image_shape) * 0.3,
            None,
            data_set,
            data_set
        )

    def from_bias(source_data_set, dest_data_set, map_noise, max_frames=10_000):
        """
        Get the pixel properties of a bias data set
        """
        m1 = np.zeros(source_data_set.image_shape)
        m2 = np.zeros(source_data_set.image_shape)
        n_frames = np.zeros(source_data_set.image_shape)
        edges = np.arange(-2, 2, 1/ADU_PER_ELECTRON)
        n_pixels = np.prod(source_data_set.image_shape)
        counts = np.zeros((len(edges)+1, n_pixels), int)
        arange = np.arange(n_pixels)
        
        # Get mean, stdev, and histograms
        for frame in source_data_set.iterator(max_frames=max_frames):
            good_mask = ~np.isnan(frame.image)
            masked_image = frame.image[good_mask]
            m1[good_mask] += masked_image
            m2[good_mask] += masked_image**2
            n_frames[good_mask] += 1
            digits = np.digitize(frame.image.reshape(-1), edges)
            counts[digits, arange] += 1
        m1 /= n_frames
        m2 /= n_frames
        counts = counts[1:-1,:]

        # Fix gaps
        gap_mask = (counts[:-2,:] > 0) & (counts[2:,:] > 0) & (counts[1:-1,:] == 0)
        count_gap_mask = np.zeros(counts.shape, bool)
        count_gap_mask[1:-1,:] = gap_mask
        local_average = (counts[:-2,:] + counts[2:,:]) / 2
        counts[count_gap_mask] = local_average[gap_mask]

        bias = m1
        widths = np.sqrt(m2 - m1**2)

        # Get fit parameters
        if map_noise:
            params = fit_gaussians(edges, counts)
            params = params.transpose().reshape((bias.shape[0], bias.shape[1], 7))
        else:
            params = None

        return PixelProperties(bias, widths, params, source_data_set, dest_data_set)
    
def fit_gaussians(edges, counts):
    """
    Fit a triple Gaussian to a list of histograms by minimizing the Cash statistic

    Parameters
    ----------
    edges : array-like 
        Edges of the bins (shape (e,))
    counts : array-like
        Data (shape (e-1, p) for p pixels.)

    Returns an array of parameters (7, p)
    """
    n_bins, n_pixels = counts.shape
    centers = (edges[1:] + edges[:-1]) / 2
    n_counts = np.sum(counts, axis=0)
    total_area  = n_counts * (centers[1] - centers[0])
    x0 = np.array([0.2, 0., 0.9, 0.6, 0.05, 0.6, 0.05])
    params = np.repeat(x0[:, None], n_pixels, axis=1)
    gradient = np.zeros((7, n_bins, n_pixels))
    old_gradient = None
    old_params = None
    excess = 1e-7 * n_counts

    for iteration in tqdm.tqdm(range(100), colour="yellow"):
        normalization = total_area / np.sqrt(2*np.pi * params[0]**2)
        x01 = np.subtract.outer(centers, params[1])/params[0] # Shape e-1, p
        x02 = x01 - params[3]/params[0]
        x03 = x01 + params[5]/params[0]
        gauss_1 = np.exp(-x01**2 / 2) * params[2] * normalization
        gauss_2 = np.exp(-x02**2 / 2) * params[4] * normalization
        gauss_3 = np.exp(-x03**2 / 2) * params[6] * normalization
        model = gauss_1 + gauss_2 + gauss_3 + excess # Extra bit to avoid divide by zero errors
        gradient[0,:,:] = (gauss_1*(x01**2-1) + gauss_2*(x02**2-1) + gauss_3*(x03**2-1)) / params[0]
        gradient[1,:,:] = (gauss_1*x01 + gauss_2*x02 + gauss_3*x03) / params[0]
        gradient[2,:,:] = gauss_1 / params[2]
        gradient[3,:,:] = gauss_2 * x02 / params[0]
        gradient[4,:,:] = gauss_2 / params[4]
        gradient[5,:,:] = -gauss_3 * x03 / params[0]
        gradient[6,:,:] = gauss_3 / params[6]
        gradient *= 2 * (1 - counts / model)
        collapsed_gradient = np.sum(gradient, axis=1)

        # Perform gradient descent
        learning_rate = 1e-2 / n_counts.astype(float)
        if iteration > 0:
            accel = collapsed_gradient - old_gradient
            bb_learning_rate = np.sum((params - old_params) * accel, axis=0) / np.sum(accel**2, axis=0)
            bb_learning_rate = np.clip(bb_learning_rate, 5e-5 / n_counts.astype(float), 3e-1 / n_counts.astype(float))
            mask = np.isfinite(bb_learning_rate)
            learning_rate[mask] = bb_learning_rate[mask]

        old_params = np.copy(params)
        params -= collapsed_gradient * learning_rate
        old_gradient = collapsed_gradient

        # Implement bounds
        params[0] = np.clip(params[0], 0.08, 0.75)
        params[1] = np.clip(params[1], -1., 1.)
        params[2] = np.clip(params[2], 0.001, 10)
        params[3] = np.clip(params[3], 0.05, 1)
        params[4] = np.clip(params[4], 0.001, 0.2)
        params[5] = np.clip(params[5], 0.05, 1)
        params[6] = np.clip(params[6], 0.001, 0.2)


    total_amp = params[2] + params[4] + params[6]
    params[2] /= total_amp
    params[4] /= total_amp
    params[6] /= total_amp

    return params