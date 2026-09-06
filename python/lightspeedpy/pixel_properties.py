import numpy as np
import os, tqdm, logging
from astropy.io import fits
from scipy.special import loggamma, digamma
from .util import trim_image
from .constants import FORBIDDEN_KEYWORDS, ADU_PER_ELECTRON, N_BIAS_FRAMES

CASH_THRESHOLD = 0.5
GRID_LOCATION = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "moments.npy"))
logger = logging.getLogger("lightspeedpy")

def pearson(x, mean, sigma, k, nu):
    a = sigma * np.sqrt(2 * k)
    z = (x - mean) / a
    log_k = (2*k - 2) * np.log(2) + 2 * np.real(loggamma(k + 0.5j * nu)) - np.log(np.pi) - loggamma(2*k - 1) - np.log(a)
    log_f = log_k - k * np.log(1 + z**2) - nu * np.arctan(z)
    return np.exp(log_f)

def pearson_grad(x, mean, sigma, k, nu):
    a = sigma * np.sqrt(2 * k)
    z = (x - mean) / a
    denom = 1 + z**2

    f = pearson(x, mean, sigma, k, nu)
    psi_c = digamma(k + 0.5j * nu)

    dlog_dk = 2*np.log(2) + 2*np.real(psi_c) - 2*digamma(2*k - 1) - 1/(2*k) - np.log(denom) + z**2/denom + nu*z/(2*k*denom)
    
    dlog_dnu = -np.imag(psi_c) - np.arctan(z)
    dlog_dsigma = ((2*k - 1)*z**2 + nu*z - 1) / (sigma * denom)
    dlog_dmean = (2*k*z + nu) / (a * denom)

    return f*np.array([dlog_dmean, dlog_dsigma, dlog_dk, dlog_dnu])

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
    def __init__(self, bias, widths, params, single_mask, source_data_set, dest_data_set):
        self.params = params
        self.single_mask = single_mask
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
        self.single_mask = self.params[bbox[0]:bbox[1],bbox[2]:bbox[3]]
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
            h4 = fits.ImageHDU(data=self.single_mask.astype(np.uint8))

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
                    h4.header[key] = value

        h0.header["PIXPROP"] = "T"
        h1.header["PIXPROP"] = "T"
        h2.header["PIXPROP"] = "T"
        if self.params is not None:
            h3.header["PIXPROP"] = "T"
            h4.header["PIXPROP"] = "T"

        hdul = [h0, h1, h2]
        if self.params is not None:
            hdul.append(h3)
            hdul.append(h4)

        fits.HDUList(hdul).writeto(filename, overwrite=clobber)

    def load(filename):
        with fits.open(filename) as hdul:
            if "PIXPROP" not in hdul[1].header or hdul[1].header["PIXPROP"] != "T":
                raise Exception(f"The file {filename} is not a PixelProperties object")
            
            bias = np.array(hdul[1].data)
            widths = np.array(hdul[2].data)
            if len(hdul) == 5:
                params = np.array(hdul[3].data)
                single_mask = np.array(hdul[4].data).astype(bool)
            else:
                params = None
                single_mask = None
            pp = PixelProperties(bias, widths, params, single_mask, None, None)
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
            single_mask = self.single_mask
            triple_mask = ~self.single_mask
            image_single_mask = self.single_mask
        else:
            single_mask = self.single_mask & mask
            triple_mask = (~self.single_mask) & mask
            image_single_mask = self.single_mask[mask]
        
        pdf = np.zeros((self.params.shape[0], self.params.shape[1]))
        pdf[single_mask] = pearson(
            image[image_single_mask]-true_n,
            self.params[single_mask,0], self.params[single_mask,1],
            self.params[single_mask,2], self.params[single_mask,3]
        )

        pdf[triple_mask] = pearson(
            image[~image_single_mask]-true_n,
            self.params[triple_mask,0], self.params[triple_mask,1],
            self.params[triple_mask,2], self.params[triple_mask,3]
        ) * self.params[triple_mask,8]
        pdf[triple_mask] += pearson(
            image[~image_single_mask]-true_n,
            self.params[triple_mask,0] + self.params[triple_mask,4], self.params[triple_mask,1],
            self.params[triple_mask,2], self.params[triple_mask,3]
        ) * self.params[triple_mask,5]
        pdf[triple_mask] += pearson(
            image[~image_single_mask]-true_n,
            self.params[triple_mask,0] - self.params[triple_mask,6],self.params[triple_mask,1],
            self.params[triple_mask,2], self.params[triple_mask,3]
        ) * self.params[triple_mask,7]

        if mask is None:
            return pdf
        else:
            return pdf[mask]

    def default(data_set):
        """
        Get the default pixel properties for a data set with no bias.
        """
        return PixelProperties(
            np.zeros(data_set.image_shape),
            np.ones(data_set.image_shape) * 0.3,
            None,
            None,
            data_set,
            data_set
        )

    def from_bias(source_data_set, dest_data_set, map_noise, max_frames=N_BIAS_FRAMES):
        """
        Get the pixel properties of a bias data set
        """
        m1 = np.zeros(source_data_set.image_shape)
        m2 = np.zeros(source_data_set.image_shape)
        n_frames = np.zeros(source_data_set.image_shape)

        # Make edges so that the zero bin is centered at exactly zero
        edges = np.arange(-2, 3, 1/ADU_PER_ELECTRON)
        edges -= edges[np.argmin(np.abs(edges))]

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
            params_single, cash_single = fit_single(edges, counts)
            params_single = params_single.transpose().reshape((bias.shape[0], bias.shape[1], params_single.shape[0]))

            params, cash_triple = fit_triple(edges, counts)
            params = params.transpose().reshape((bias.shape[0], bias.shape[1], params.shape[0]))

            single_mask = (cash_triple - cash_single < 8).reshape((bias.shape[0], bias.shape[1]))
            print("Rate of singles", np.mean(single_mask))
            params[single_mask,:4] = params_single[single_mask,:]
        else:
            params = None
            single_mask = None

        return PixelProperties(bias, widths, params, single_mask, source_data_set, dest_data_set)
    
def fit_single(edges, counts):
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
    centers = (edges[1:] + edges[:-1]) / 2
    n_counts = np.sum(counts, axis=0).astype(float)
    x0 = np.array([0, 0.22, 6, 0, 1])
    params = np.repeat(x0[:, None], counts.shape[1], axis=1)
    normalization = n_counts / ADU_PER_ELECTRON

    for iteration in tqdm.tqdm(range(300), colour="yellow"):
        outers = np.subtract.outer(centers, params[0])
        gradient = pearson_grad(outers, 0, params[1], params[2], params[3]) * normalization * params[4]
        model = pearson(outers, 0, params[1], params[2], params[3]) * normalization
        model += 1e-7 * n_counts
        gradient = np.concatenate([
            gradient,
            [model]
        ])
        model *= params[4]

        gradient *= 2 * (1 - counts / model)
        collapsed_gradient = np.sum(gradient, axis=1)
        print(params[:,0])
        print(collapsed_gradient[:,0])

        # Perform gradient descent
        learning_rate = 1e-1 / n_counts

        params -= collapsed_gradient * learning_rate

        # Implement bounds
        params[0] = np.clip(params[0], -1, 1) # Mean
        params[1] = np.clip(params[1], 0.05, 0.8) # Sigma
        params[2] = np.clip(params[2], 1, 1000) # k
        params[3] = np.clip(params[3], -3, 3) # Nu
        params[4] = np.clip(params[4], 0.5, 1.5) # Norm

    params[4] = 1
    outers = np.subtract.outer(centers, params[0])
    model = pearson(outers, 0, params[1], params[2], params[3]) * normalization
    cash = 2*np.sum((model - counts * np.log(model))[np.abs(centers) > CASH_THRESHOLD], axis=0)

    return params[:4], cash

def fit_triple(edges, counts):
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
    centers = (edges[1:] + edges[:-1]) / 2
    n_counts = np.sum(counts, axis=0)
    x0 = np.array([0, 0.22, 6, 0, 0.6, 0.05, 0.6, 0.05, 0.9])
    params = np.repeat(x0[:, None], counts.shape[1], axis=1)
    normalization = n_counts / ADU_PER_ELECTRON

    for iteration in tqdm.tqdm(range(300), colour="yellow"):
        outers = np.subtract.outer(centers, params[0])
        m1 = pearson(outers, 0, params[1], params[2], params[3]) * normalization
        g1 = pearson_grad(outers, 0, params[1], params[2], params[3]) * normalization
        m2 = pearson(outers, 0 + params[4], params[1], params[2], params[3]) * normalization
        g2 = pearson_grad(outers, 0 + params[4], params[1], params[2], params[3]) * normalization
        m3 = pearson(outers, 0 - params[6], params[1], params[2], params[3]) * normalization
        g3 = pearson_grad(outers, 0 - params[6], params[1], params[2], params[3]) * normalization
        model = m1 * params[8] + m2 * params[5] + m3 * params[7]
        gradient = g1 * params[8] + g2 * params[5] + g3 * params[7]
        gradient = np.concatenate([gradient,
            [g2[0] * params[5],
            m2,
            -g3[0] * params[7],
            m3,
            m1,
            ],
        ])
        model += 1e-7 * n_counts
        gradient *= (1 - counts / model)
        collapsed_gradient = np.sum(gradient, axis=1)
        collapsed_gradient = np.sum(gradient, axis=1)

        # Perform gradient descent
        learning_rate = 2e-2 / n_counts.astype(float)
        params -= collapsed_gradient * learning_rate

        # Implement bounds
        params[0] = np.clip(params[0], -1, 1) # Mean
        params[1] = np.clip(params[1], 0.05, 0.8) # Sigma
        params[2] = np.clip(params[2], 1, 1000) # k
        params[3] = np.clip(params[3], -3, 3) # nu
        params[4] = np.clip(params[4], 0.05, 1) # Delta high
        params[5] = np.clip(params[5], 0.001, 0.3) # Amp high
        params[6] = np.clip(params[6], 0.05, 1) # Delta low
        params[7] = np.clip(params[7], 0.001, 0.3) # Amp low
        params[8] = np.clip(params[8], 0.5, 1.5) # Amp mid

    total = params[8] + params[5] + params[7]
    params[8] /= total
    params[5] /= total
    params[7] /= total

    outers = np.subtract.outer(centers, params[0])
    m1 = pearson(outers, 0, params[1], params[2], params[3]) * normalization
    m2 = pearson(outers, 0 + params[4], params[1], params[2], params[3]) * normalization
    m3 = pearson(outers, 0 - params[6], params[1], params[2], params[3]) * normalization
    model = m1 * params[8] + m2 * params[5] + m3 * params[7]
    cash = 2*np.sum((model - counts * np.log(model))[np.abs(centers) > CASH_THRESHOLD], axis=0)

    return params, cash