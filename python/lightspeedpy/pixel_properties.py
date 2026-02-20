import numpy as np
from scipy.interpolate import LinearNDInterpolator
import os
from astropy.io import fits
from .util import trim_image
from .constants import ADU_PER_ELECTRON, FORBIDDEN_KEYWORDS

GRID_LOCATION = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "moments.npy"))

def make_m1m2_grid():
    if not os.path.exists(GRID_LOCATION):
        sigmas = np.linspace(0, 1, 100)[1:]
        means = np.linspace(-0.5, 0.5, 100)
        line = np.linspace(-3, 3, 100)
        xs = line - np.floor(line + 0.5)
        x2s = xs**2
        moments = np.zeros((2, len(sigmas), len(means)))
        for i, sigma in enumerate(sigmas):
            for j, mean in enumerate(means):
                gauss = np.exp(-(line - mean)**2 / (2*sigma**2))
                gauss /= np.sum(gauss)
                moments[0,i,j] = np.sum(gauss*xs)
                moments[1,i,j] = np.sum(gauss*x2s)

        sigma_grid, mean_grid = np.meshgrid(sigmas, means, indexing="ij")
        sigma_grid = sigma_grid.reshape(-1)
        mean_grid = mean_grid.reshape(-1)
        moments = moments.reshape(2, -1)
        if not os.path.exists(os.path.dirname(GRID_LOCATION)):
            os.mkdir(os.path.dirname(GRID_LOCATION))
        np.save(GRID_LOCATION, np.concatenate([moments, [sigma_grid], [mean_grid]]))
    m1, m2, sigmas, means = np.load(GRID_LOCATION)

    return LinearNDInterpolator((m1,m2), np.transpose([means, sigmas]))

GRID_INTERPOLATOR = make_m1m2_grid()

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
    def __init__(self, bias, widths, source_data_set, dest_data_set):
        if bias is not None: # Cheat code so that I can construct an empty PixelProperties: set bias=None
            self.bias = trim_image(bias, source_data_set, dest_data_set)
            self.widths = trim_image(widths, source_data_set, dest_data_set)
            self.header0 = source_data_set.header0
            self.header1 = source_data_set.header1

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

        for key, value in self.header0.items():
            if key not in FORBIDDEN_KEYWORDS:
                if len(key) > 8: key = f"HIERARCH {key}"
                h0.header[key] = value

        for key, value in self.header1.items():
            if key not in FORBIDDEN_KEYWORDS:
                if len(key) > 8: key = f"HIERARCH {key}"
                h1.header[key] = value
                h2.header[key] = value

        h0.header["PIXPROP"] = "T"
        h1.header["PIXPROP"] = "T"
        h2.header["PIXPROP"] = "T"

        fits.HDUList([h0, h1, h2]).writeto(filename, overwrite=clobber)

    def load(filename):
        with fits.open(filename) as hdul:
            if "PIXPROP" not in hdul[1].header or hdul[1].header["PIXPROP"] != "T":
                raise Exception(f"The file {filename} is not a PixelProperties object")
            
            pp = PixelProperties(None,None,None,None)
            pp.bias = np.array(hdul[1].data)
            pp.widths = np.array(hdul[2].data)
            pp.header0 = hdul[0].header
            pp.header1 = hdul[1].header
        return pp

    def _get_moments(data_set):
        """
        Return images of the first and second moments of the data set
        """
        m1_image = np.zeros(data_set.image_shape)
        m2_image = np.zeros(data_set.image_shape)
        n_frames = np.zeros(data_set.image_shape)
        
        for frame in data_set.iterator(max_frames=10_000):
            clipped_image = frame.image - np.floor(frame.image + 0.5)
            good_mask = ~np.isnan(frame.image)
            m1_image[good_mask] += clipped_image[good_mask]
            m2_image[good_mask] += clipped_image[good_mask]**2
            n_frames[good_mask] += 1

        m1_image /= n_frames
        m2_image /= n_frames
        return m1_image, m2_image

    def default(data_set):
        """
        Get the default pixel properties for a data set with no bias and not self-biased.
        """
        return PixelProperties(
            np.zeros(data_set.image_shape) * 199.5 / ADU_PER_ELECTRON,
            np.zeros(data_set.image_shape) * 0.3,
            data_set,
            data_set
        )

    def from_data(source_data_set, dest_data_set):
        """
        Get the pixel properties of a faint, rapid readout data set
        """
        m1_image, m2_image = PixelProperties._get_moments(source_data_set)
        output = GRID_INTERPOLATOR((m1_image, m2_image))
        bias = output[:,:,0]
        widths = output[:,:,1]

        widths = np.sqrt(m2_image - m1_image**2) # TODO

        return PixelProperties(bias, widths, source_data_set, dest_data_set)

    def from_bias(source_data_set, dest_data_set):
        """
        Get the pixel properties of a bias data set
        """
        m1_image, m2_image = PixelProperties._get_moments(source_data_set)

        bias = m1_image
        widths = np.sqrt(m2_image - m1_image**2)

        return PixelProperties(bias, widths, source_data_set, dest_data_set)