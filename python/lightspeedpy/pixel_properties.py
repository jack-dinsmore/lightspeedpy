import numpy as np
from scipy.optimize import least_squares
from multiprocessing import Pool
import os, tqdm
from astropy.io import fits
from .util import trim_image
from .constants import FORBIDDEN_KEYWORDS, ADU_PER_ELECTRON

GRID_LOCATION = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "moments.npy"))

class PixelProperties:
    """
    Biases and noise of each pixel in the data set. Use :meth:`PixelProperties.from_data` or :meth:`PixelProperties.from_bias` to create it.
    
    Attributes
    ----------
    bias : array-like
        Image of biases of each pixel
    widths : array-like
        Noises in each pixel, defined as the standard deviation of the Gaussian error approximation.
    """
    def __init__(self, bias, widths, params, source_data_set, dest_data_set):
        if bias is not None: # Cheat code so that I can construct an empty PixelProperties: set bias=None
            if source_data_set is not None:
                self.bias = trim_image(bias, source_data_set, dest_data_set)
                self.widths = trim_image(widths, source_data_set, dest_data_set)
                self.params = np.zeros((self.bias.shape[0], self.bias.shape[1], 7))
                for i in range(params.shape[2]):
                    self.params[:,:,i] = trim_image(params[:,:,i], source_data_set, dest_data_set=dest_data_set)
                self.header0 = source_data_set.header0
                self.header1 = source_data_set.header1
            else:
                self.bias = bias
                self.widths = widths
                self.params = params

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
                h3.header[key] = value

        h0.header["PIXPROP"] = "T"
        h1.header["PIXPROP"] = "T"
        h2.header["PIXPROP"] = "T"
        h3.header["PIXPROP"] = "T"

        fits.HDUList([h0, h1, h2, h3]).writeto(filename, overwrite=clobber)

    def load(filename):
        with fits.open(filename) as hdul:
            if "PIXPROP" not in hdul[1].header or hdul[1].header["PIXPROP"] != "T":
                raise Exception(f"The file {filename} is not a PixelProperties object")
            
            bias = np.array(hdul[1].data)
            widths = np.array(hdul[2].data)
            params = np.array(hdul[3].data)
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
        Get the default pixel properties for a data set with no bias and not self-biased.
        """
        return PixelProperties(
            np.zeros(data_set.image_shape),
            np.ones(data_set.image_shape) * 0.3,
            np.zeros((data_set.image_shape[0], data_set.image_shape[1], 6)),
            data_set,
            data_set
        )

    def from_bias(source_data_set, dest_data_set, max_frames=10_000, use_pool=True):
        """
        Get the pixel properties of a bias data set
        """
        m1 = np.zeros(source_data_set.image_shape)
        m2 = np.zeros(source_data_set.image_shape)
        n_frames = np.zeros(source_data_set.image_shape)
        edges = np.arange(-2, 2, 1/ADU_PER_ELECTRON)
        n_pixels = np.prod(source_data_set.image_shape)
        counts = np.zeros((n_pixels, len(edges)+1), int)
        arange = np.arange(n_pixels)
        
        # Get mean, stdev, and histograms
        for frame in source_data_set.iterator(max_frames=max_frames):
            good_mask = ~np.isnan(frame.image)
            masked_image = frame.image[good_mask]
            m1[good_mask] += masked_image
            m2[good_mask] += masked_image**2
            n_frames[good_mask] += 1
            digits = np.digitize(frame.image.reshape(-1), edges)
            counts[arange,digits] += 1
        m1 /= n_frames
        m2 /= n_frames
        counts = counts[:,1:-1]

        bias = m1
        widths = np.sqrt(m2 - m1**2)

        # Get fit parameters
        centers = (edges[1:] + edges[:-1]) / 2
        args = [(c, centers) for c in counts]
        if use_pool:
            with Pool() as pool:
                params = list(tqdm.tqdm(pool.map(fit_gaussians, args), total=len(args)))
        else:
            params = []
            for arg in tqdm.tqdm(args):
                params.append(fit_gaussians(arg))
        params = np.array(params)

        params = params.reshape((bias.shape[0], bias.shape[1], 7))
        return PixelProperties(bias, widths, params, source_data_set, dest_data_set)
    
def fit_gaussians(args):
    """
    Fit a triple Gaussian to the series of data given by data_points, which should be (n, m) for m data points from n pixels.

    Returns a list of parameters. sigma, mu1, amp1, mu2, amp2, ...
    """
    return [0.4, 0., 0.9, 0.5, 0.05, 0.5, 0.05] # TODO short-circuited the Gaussian fitting
    counts, centers = args
    errors = np.sqrt(counts)
    errors[errors==0] = np.inf
    total_area  = np.sum(counts) * (centers[1]-centers[0]) 
    def score(params):
        denom = 1 / (2*params[0]**2)
        norm = total_area / np.sqrt(2*np.pi*params[0]**2)
        ys = np.exp(-(centers-params[1])**2 * denom) * params[2] * norm
        ys += np.exp(-(centers-params[1]-params[3])**2 * denom) * params[4] * norm
        ys += np.exp(-(centers-params[1]+params[5])**2 * denom) * params[6] * norm
        return (ys - counts) / errors
    # TODO jacobian

    x0 = [0.4, 0., 0.9, 0.5, 0.05, 0.5, 0.05]
    bounds = np.array([(0.02, 1.5),
        (-2, 2), (0., 10),
        (0.05, 1), (0, 0.2),
        (0.05, 1), (0, 0.2)
    ]).transpose()
    result = least_squares(score, x0, bounds=bounds)
    params = result.x

    total_amp = params[2] + params[4] + params[6]
    params[2] /= total_amp
    params[4] /= total_amp
    params[6] /= total_amp

    return params