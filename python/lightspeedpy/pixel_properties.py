import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.special import gamma, beta
import os
from astropy.io import fits
from .util import trim_image
from .constants import FORBIDDEN_KEYWORDS

GRID_LOCATION = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "moments.npy"))

def make_m1m2_grid():
    if not os.path.exists(GRID_LOCATION):
        sigmas = np.linspace(0, 0.7, 300)[1:]
        means = np.linspace(-0.5, 0.5, 101)
        line = np.linspace(-4, 4, 100)
        xs = line - np.floor(line + 0.5)
        measured_moments = np.zeros((2, len(sigmas), len(means)))
        true_moments = np.zeros((2, len(sigmas), len(means)))
        point_mask = np.ones((2, len(sigmas), len(means))).astype(bool)

        for i, sigma in enumerate(sigmas):
            if 0 <= sigma < 0.2:
                m = 7
                k = 2.11
            if 0.2 <= sigma < 0.22:
                m = 6.5
                k = 1.84
            if 0.22 <= sigma < 0.233:
                m = 6.5
                k = 1.97
            if 0.233 <= sigma < 0.25:
                m = 6
                k = 2.14
            if 0.25 <= sigma < 0.27:
                m = 5.5
                k = 2.33
            if 0.27 <= sigma < 0.31:
                m = 5
                k = 2.70
            if 0.31 <= sigma < 0.38:
                m = 4.5
                k = 2.52
            if 0.38 <= sigma:
                m = 4
                k = 1.69

            m = 4
            k = 0

            a = sigma * np.sqrt((2*m - 3) / (1 + k**2/(4*(m - 1)**2))) # Such that variance is matched
            b = -a * k / (2*m - 2) # Such that mean is matched

            for j, mean in enumerate(means):
                xi = (line - mean - b) / a
                pdf = np.exp(k * np.arctan(xi)) / (1 + xi**2)**m
                pdf /= np.sum(pdf)
                true_moments[0,i,j] = np.sum(pdf*line)
                true_moments[1,i,j] = np.sum(pdf*line**2)
                measured_moments[0,i,j] = np.sum(pdf*xs)
                measured_moments[1,i,j] = np.sum(pdf*xs**2)

        # Mask points where the mean can't be measured
        delta_mean = measured_moments[0,:,1:] - measured_moments[0,:,:-1]
        point_mask = delta_mean > 0
        point_mask = np.hstack([point_mask, np.zeros((point_mask.shape[0],1), bool)])

        measured_moments = measured_moments[:,point_mask].reshape(2, -1)
        true_moments = true_moments[:,point_mask].reshape(2, -1)

        import matplotlib.pyplot as plt
        plt.style.use("root")
        fig, axs = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True)
        cmap = plt.colormaps["inferno"]
        random_mask = np.random.random(measured_moments.shape[1]) < 0.08
        for true, measured in zip(true_moments[:,random_mask].transpose(), measured_moments[:,random_mask].transpose()):
            true_std = np.sqrt(true[1] - true[0]**2)
            measured_std = np.sqrt(measured[1] - measured[0]**2)
            color = cmap(measured[0]+0.5)
            axs[0,0].scatter(true[0], true_std, color=color, s=2, edgecolor='none')
            color = cmap(measured_std * 2) # M2
            axs[0,1].scatter(true[0], true_std, color=color, s=2, edgecolor='none')
            color = cmap(true[0]+0.5)
            axs[1,0].scatter(measured[0], measured_std, color=color, s=2, edgecolor='none')
            color = cmap(true_std * 2) # M2
            axs[1,1].scatter(measured[0], measured_std, color=color, s=2, edgecolor='none')
        axs[0,0].set_title("Measured Mean")
        axs[0,1].set_title("Measured Std")
        axs[1,0].set_title("True Mean")
        axs[1,1].set_title("True Std")
        fig.savefig(f"{GRID_LOCATION}.png")

        if not os.path.exists(os.path.dirname(GRID_LOCATION)):
            os.mkdir(os.path.dirname(GRID_LOCATION))
        np.save(GRID_LOCATION, np.concatenate([measured_moments, true_moments]))

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
    def __init__(self, bias, widths, skew, source_data_set, dest_data_set):
        if bias is not None: # Cheat code so that I can construct an empty PixelProperties: set bias=None
            if source_data_set is not None:
                self.bias = trim_image(bias, source_data_set, dest_data_set)
                self.widths = trim_image(widths, source_data_set, dest_data_set)
                self.skews = trim_image(skew, source_data_set, dest_data_set)
                self.header0 = source_data_set.header0
                self.header1 = source_data_set.header1
            else:
                self.bias = bias
                self.widths = widths
                self.skews = skew

            self.ms = np.ones_like(self.skews)
            self.ms[self.widths < 0.2] = 7
            self.ms[(0.2 <= self.widths) & (self.widths < 0.22)] = 6.5
            self.ms[(0.22 <= self.widths) & (self.widths < 0.233)] = 6.5
            self.ms[(0.233 <= self.widths) & (self.widths < 0.25)] = 6
            self.ms[(0.25 <= self.widths) & (self.widths < 0.27)] = 5.5
            self.ms[(0.27 <= self.widths) & (self.widths < 0.31)] = 5
            self.ms[(0.31 <= self.widths) & (self.widths < 0.38)] = 4.5
            self.ms[(0.38 <= self.widths)] = 4

            self.ks = 2 * self.skews * np.sqrt(-(4 - 12*self.ms + 13*self.ms**2 - 6*self.ms**3 + self.ms**4) / (12 - 8*self.ms + (4 - 4*self.ms + self.ms**2)*self.skews**2))
            self.alphas = self.widths * np.sqrt((2*self.ms - 3) / (1 + self.ks**2/(4*(self.ms - 1)**2)))
            self.betas = -self.alphas * self.ks / (2*self.ms - 2)
            self.norms = np.abs(gamma(self.ms - 0.5j*self.ks) / gamma(self.ms))**2 / beta(self.ms-0.5, 0.5) / self.alphas

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
        h3 = fits.ImageHDU(data=self.skews)

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
            skews = np.array(hdul[3].data)
            pp = PixelProperties(bias, widths, skews, None, None)
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
            xi = (image - true_n - self.betas)/self.alphas
            pdf = np.exp(self.ks * np.arctan(xi)) / (1 + xi**2)**self.ms * self.norms
        else:
            xi = (image - true_n - self.betas[mask])/self.alphas[mask]
            pdf = np.exp(self.ks[mask] * np.arctan(xi)) / (1 + xi**2)**self.ms[mask] * self.norms[mask]

        return pdf * 0.99 + 0.01

    def default(data_set):
        """
        Get the default pixel properties for a data set with no bias and not self-biased.
        """
        return PixelProperties(
            np.zeros(data_set.image_shape),
            np.ones(data_set.image_shape) * 0.3,
            np.ones(data_set.image_shape)*0.35,
            data_set,
            data_set
        )

    def from_data(source_data_set, dest_data_set):
        """
        Get the pixel properties of a faint, rapid readout data set
        """
        m1 = np.zeros(source_data_set.image_shape)
        m2 = np.zeros(source_data_set.image_shape)
        n_frames = np.zeros(source_data_set.image_shape)
        
        for frame in source_data_set.iterator(max_frames=10_000):
            good_mask = ~np.isnan(frame.image)
            masked_image = (frame.image - np.floor(frame.image + 0.5))[good_mask]
            m1[good_mask] += masked_image
            m2[good_mask] += masked_image**2
            n_frames[good_mask] += 1
        m1 /= n_frames
        m2 /= n_frames

        output = GRID_INTERPOLATOR((m1, m2))
        m1 = output[:,:,0]
        m2 = output[:,:,1]

        bias = m1
        widths = np.sqrt(m2 - m1**2)

        skews = np.ones_like(bias)
        skews[widths < 0.2] = 0.23
        skews[(0.2 <= widths) & (widths < 0.22)] = 0.23
        skews[(0.22 <= widths) & (widths < 0.233)] = 0.25
        skews[(0.233 <= widths) & (widths < 0.25)] = 0.31
        skews[(0.25 <= widths) & (widths < 0.27)] = 0.41
        skews[(0.27 <= widths) & (widths < 0.31)] = 0.56
        skews[(0.31 <= widths) & (widths < 0.38)] = 0.66
        skews[(0.38 <= widths)] = 0.61

        return PixelProperties(bias, widths, skews, source_data_set, dest_data_set)

    def from_bias(source_data_set, dest_data_set):
        """
        Get the pixel properties of a bias data set
        """
        m1 = np.zeros(source_data_set.image_shape)
        m2 = np.zeros(source_data_set.image_shape)
        m3 = np.zeros(source_data_set.image_shape)
        n_frames = np.zeros(source_data_set.image_shape)
        
        for frame in source_data_set.iterator(max_frames=10_000):
            good_mask = ~np.isnan(frame.image)
            masked_image = frame.image[good_mask]
            m1[good_mask] += masked_image
            m2[good_mask] += masked_image**2
            m3[good_mask] += masked_image**3
            n_frames[good_mask] += 1
        m1 /= n_frames
        m2 /= n_frames
        m3 /= n_frames

        bias = m1
        widths = np.sqrt(m2 - m1**2)
        skews = (m3 - 3*m1*m2) / widths**3

        return PixelProperties(bias, widths, skews, source_data_set, dest_data_set)