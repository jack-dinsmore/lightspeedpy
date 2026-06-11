import numpy as np
from scipy.special import factorial
from scipy.ndimage import convolve
from scipy.ndimage import rotate
import copy
from astropy.io import fits
from astropy.wcs import WCS
from ..qe import get_qe
from ..constants import PIXEL_SIZE, FORBIDDEN_KEYWORDS
from ..util import from_hms, from_dms

FLAT_NAN_THRESHOLD = 0.1

class Image:
    """
    Create an image. The input should be in units of total electrons detected.
    
    Parameters
    ----------
    image : array-like
        Image array
    data_set : DataSet
        Data set from which the image was derived
    n_frames : int or array-like
        Number of frames for which each pixel was exposed. If all pixels were exposed equally, you may supply a single integer. For observations in which pixels have different exposures, provide an image of integers.
    offset : (int, int), optiona;
        ra, dec offset in arcseconds for the WCS
    """
    def __init__(self, image, data_set, n_frames, offset=None, correct_qe=True):
        self.header0 = data_set.header0
        self.header1 = data_set.header1
        for frame in data_set.iterator(use_bar=False):
            self.frame_duration = frame.duration
            break
        if type(n_frames) is int:
            self.n_frames = n_frames * np.ones(image.shape, int)
        else:
            self.n_frames = n_frames

        qe = get_qe()
        electrons_per_frame = image / self.n_frames
        if correct_qe:
            photons_per_frame = electrons_per_frame / qe(electrons_per_frame)
        else:
            photons_per_frame = electrons_per_frame
        self.photons_per_second = photons_per_frame / self.frame_duration
        self.flat_corrected = False
        if data_set.flat is not None:
            self.photons_per_second /= data_set.flat
            self.photons_per_second[data_set.flat < FLAT_NAN_THRESHOLD] = np.nan
            self.flat_corrected = True
        self.pixel_properties = copy.deepcopy(data_set.pixel_properties)

        self.rot_angle = float(data_set.header1["TELPA"]) + float(data_set.header1["ROTENC"]) # deg
        pixscale = PIXEL_SIZE / 3600
        self.wcs = WCS(naxis=2)
        self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        ra = from_hms(data_set.header1["TELRA"])
        dec = from_dms(data_set.header1["TELDEC"])
        if offset is not None:
            ra -= offset[0]-offset[1]/3600
            dec -= offset[1]/3600*np.cos(dec*np.pi/180)
        self.wcs.wcs.crval = [ra, dec]
        self.wcs.wcs.crpix = [image.shape[0] / 2., image.shape[1] / 2.]
        self.wcs.wcs.cdelt = [-pixscale, pixscale]

    def nan_remove(self):
        """
        Replace all nans with averages of the surrounding pixels
        """
        valid = ~np.isnan(self.photons_per_second)
        kernel = np.ones((3, 3))
        image_filled = np.where(valid, self.photons_per_second, 0.0)
        neighbor_sum = convolve(image_filled, kernel, mode="constant", cval=0.0)
        neighbor_count = convolve(valid.astype(float), kernel, mode="constant", cval=0.0)
        local_mean = neighbor_sum / neighbor_count
        self.photons_per_second[~valid] = local_mean[~valid]
        
    def smooth(self, sigma):
        line = np.arange(-np.ceil(sigma*3), np.ceil(sigma*3)+1)
        xs, ys = np.meshgrid(line,line)
        gauss = np.exp(-(xs**2 + ys**2) / (2*sigma**2))
        gauss /= np.sum(gauss)

        bad_mask = ~np.isfinite(self.photons_per_second)
        self.photons_per_second[bad_mask] = np.median(self.photons_per_second[~bad_mask])
        weight_image = 1/self.pixel_properties.widths**2
        blurred_image = convolve(self.photons_per_second * weight_image, gauss)
        blurred_weights = convolve(weight_image, gauss)
        self.photons_per_second = blurred_image / blurred_weights
        self.photons_per_second[bad_mask] = np.nan

    def save(self, filename, apply_wcs, clobber=False, save_kwargs=None):
        """
        Save the image to a file
        
        Parameters
        ----------
        filename : str
            The file name to which the light curve should be saved
        clobber : bool, optional
            Set to True to allow overwriting
        save_kwargs : dict, optional
            Dictionary of keywords to write to the light curve header
        """

        if apply_wcs:
            save_image = np.flip(self.photons_per_second, axis=0)
            bad_mask = ~np.isfinite(save_image)
            save_image[bad_mask] = np.nanmedian(save_image)
            save_image = rotate(save_image, self.rot_angle)
            bad_mask = rotate(bad_mask.astype(float), self.rot_angle)
            save_image[bad_mask > 0.5] = np.nan
        else:
            save_image = np.copy(self.photons_per_second)
        hdu = fits.PrimaryHDU(save_image)
        exposure = np.max(self.n_frames * self.frame_duration)
        
        # Write header
        if "GPSSTART" in self.header0:
            hdu.header["GPSSTART"] = self.header0["GPSSTART"]
            
        for key, value in self.header1.items():
            if key not in FORBIDDEN_KEYWORDS:
                if len(key) > 8: key = f"HIERARCH {key}"
                hdu.header[key] = value

        hdu.header["EXPTIME"] = exposure

        if save_kwargs is not None:
            for key, value in save_kwargs.items():
                if type(value) == list:
                    for v in value:
                        key = f"{key}i"
                        if len(key) > 8: key = f"HIERARCH {key}"
                        hdu.header[f"{key}i"] = v
                    continue
                if len(key) > 8: key = f"HIERARCH {key}"
                hdu.header[key] = value

        hdu.header["BUNIT"] = "phot/s"
        hdu.header["QECORR"] = "T"
        hdu.header["FLATCORR"] = "T" if self.flat_corrected else "F"

        if apply_wcs:
            hdu.header.update(self.wcs.to_header())
        
        # Write to file
        hdu.writeto(filename, overwrite=clobber)

def load_image(image, assert_items=None):
    with fits.open(image) as hdul:
        if assert_items is not None:
            for key in assert_items:
                assert(key in hdul[0].header)
                assert(hdul[0].header[key] == assert_items[key])
        return np.array(hdul[0].data)
    
def get_clipped_image(data_set):
    """
    Get a bias, dark, flat corrected image from a :class:`DataSet` by summing all the detected photons per frame, clipped to zero or 1.
    
    Parameters
    ----------
    data_set : DataSet
        The proto-Lightspeed data set

    Returns
    -------
    array-like
        The image, crrected for flat and quantum efficiency
    """
    image = np.zeros(data_set.image_shape)
    n_frames = np.zeros(data_set.image_shape)
    for frame in data_set:
        good_mask = ~np.isnan(frame.image)
        image[good_mask] += np.round(frame.image[good_mask])
        n_frames[good_mask] += 1

    return Image(image, data_set, n_frames)

def get_summed_image(data_set):
    """
    Get a bias, dark, flat corrected image from a :class:`DataSet` by summing all the detected photons per frame.
    
    Parameters
    ----------
    data_set : DataSet
        The proto-Lightspeed data set

    Returns
    -------
    array-like
        The image, crrected for flat and quantum efficiency
    """
    image = np.zeros(data_set.image_shape)
    n_frames = np.zeros(data_set.image_shape)
    for frame in data_set:
        good_mask = ~np.isnan(frame.image)
        image[good_mask] += frame.image[good_mask]
        n_frames[good_mask] += 1

    return Image(image, data_set, n_frames)

def get_weighted_image_linearized(data_set):
    """
    Get a bias, dark, flat corrected image from a :class:`DataSet` after weighting by the probability of each being real. This function assumes that the true number of photons expected per pixel is << 1.
    
    Parameters
    ----------
    data_set : DataSet
        The proto-Lightspeed data set

    Returns
    -------
    array-like
        The image, crrected for flat and quantum efficiency
    """
    numer = np.zeros(data_set.image_shape)
    denom = np.zeros(data_set.image_shape)
    n_frames = np.zeros(data_set.image_shape, int)
    qe = get_qe()
    frame_duration = None
    for frame in data_set:
        good_mask = ~np.isnan(frame.image)
        masked_image = frame.image[good_mask]
        p0 = data_set.pixel_properties.get_prob(masked_image, 0, mask=good_mask) * qe(0)
        p1 = data_set.pixel_properties.get_prob(masked_image, 1, mask=good_mask) * qe(1)
        p2 = data_set.pixel_properties.get_prob(masked_image, 2, mask=good_mask) * qe(2)
        odds1 = p1/p0
        odds2 = p2/p0
        numer[good_mask] += odds1 - 1
        denom[good_mask] += odds1**2 - odds2
        n_frames[good_mask] += 1
        frame_duration = frame.duration

    image = numer / denom / frame_duration
    return Image(image, data_set, n_frames, correct_qe=False)
    