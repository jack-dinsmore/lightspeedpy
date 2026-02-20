import numpy as np
from scipy.special import factorial
from scipy.ndimage import rotate
from astropy.io import fits
from astropy.wcs import WCS
from ..qe import get_qe
from ..constants import PIXEL_SIZE, FORBIDDEN_KEYWORDS
from ..util import from_hms, from_dms

WEIGHTED_FLAT_P = 0.1
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
    def __init__(self, image, data_set, n_frames, offset=None):
        self.header0 = data_set.header0
        self.header1 = data_set.header1
        for frame in data_set:
            self.frame_duration = frame.duration
            break
        if type(n_frames) is int:
            self.n_frames = n_frames * np.ones(image.shape, int)
        else:
            self.n_frames = n_frames

        qe = get_qe()
        electrons_per_frame = image / self.n_frames
        photons_per_frame = electrons_per_frame / qe(electrons_per_frame)
        self.photons_per_second = photons_per_frame / self.frame_duration
        self.flat_corrected = False
        if data_set.flat is not None:
            self.photons_per_second /= data_set.flat
            self.photons_per_second[data_set.flat < FLAT_NAN_THRESHOLD] = np.nan
            self.flat_corrected = True

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

def load_image(image, assert_items):
    with fits.open(image) as hdul:
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
        The image, crrected for flat TODO and quantum efficiency
    """
    image = np.zeros(data_set.image_shape)
    duration = np.zeros(data_set.image_shape)
    n_frames = np.zeros(data_set.image_shape)
    for frame in data_set:
        good_mask = ~np.isnan(frame.image)
        image[good_mask] += np.round(frame.image[good_mask])
        duration[good_mask] += frame.duration
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
        The image, crrected for flat TODO and quantum efficiency
    """
    duration = np.zeros(data_set.image_shape)
    image = np.zeros(data_set.image_shape)
    n_frames = np.zeros(data_set.image_shape)
    for frame in data_set:
        good_mask = ~np.isnan(frame.image)
        image[good_mask] += frame.image[good_mask]
        duration[good_mask] += frame.duration
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
        The image, crrected for flat TODO and quantum efficiency
    """
    numer = np.zeros(data_set.image_shape)
    denom = np.zeros(data_set.image_shape)
    pixel_properties = data_set.runs[0].get_pixel_properties() # TODO assumes these all have the same properties
    w_denom = 1/(2*pixel_properties.widths**2)
    g_norm = np.sqrt(2*np.pi*pixel_properties.widths**2)
    n_frames = np.zeros(data_set.image_shape, int)
    for frame in data_set:
        frame_duration = frame.duration
        p0 = np.exp(-frame.image**2 * w_denom) + WEIGHTED_FLAT_P*g_norm
        p1 = np.exp(-(frame.image-1)**2 * w_denom) + WEIGHTED_FLAT_P*g_norm
        good_mask = ~np.isnan(frame.image)
        odds = (p1/p0)[good_mask]
        numer[good_mask] += odds - 1
        denom[good_mask] += odds**2
        n_frames[good_mask] += 1

    image = numer/denom
    image *= n_frames

    return Image(image, data_set, n_frames)
    
def get_weighted_image(data_set):
    """
    Get a bias, dark, flat corrected image from a :class:`DataSet` after weighting by the probability of each being real.
    
    Parameters
    ----------
    data_set : DataSet
        The proto-Lightspeed data set

    Returns
    -------
    array-like
        The image, crrected for flat TODO and quantum efficiency
    """
    raise NotImplementedError()
    # Get initial image
    image = get_summed_image(data_set)
    frame_duration = data_set.runs[0].header1["EXPOSURE TIME"]
    image *= frame_duration
    if data_set.flat is not None:
        image *= data_set.flat

    width_image = data_set.runs[0].get_pixel_properties().widths
    w_denom = 1 / (2*width_image**2)
    g_norm = 1 / np.sqrt(2*np.pi*width_image**2)
    max_n = int(np.nanmax(image)) + 2
    if max_n > 100:
        print(f"WARNING: The image has pixels with >100 electrons per frame (max: {max_n}). Are you sure you want to do weighted extraction? It will take a long time")

    import matplotlib.pyplot as plt
    plt.style.use("root")

    image_cpy =np.copy(image)
    image_cpy[~np.isfinite(image)] = 0
    plt.imsave(f"start.png", image_cpy, vmin=0, vmax=50)

    # Run a few Newton iterations
    for iteration in range(4):
        ts = np.zeros(data_set.image_shape)
        tsderiv = np.zeros(data_set.image_shape)

        poisson_pdfs = []
        for n in range(max_n):
            poisson_pdfs.append(image**n / factorial(n))
        poisson_pdfs = np.array(poisson_pdfs)

        # Count up the moments
        for frame in data_set:
            good_mask = ~np.isnan(frame.image)
            n_good_pixels = np.sum(good_mask)
            m0 = np.zeros(n_good_pixels)
            m1 = np.zeros(n_good_pixels)
            m2 = np.zeros(n_good_pixels)
            for n in range(max_n):
                delta = frame.image - n
                pdf = (np.exp(-delta*delta*w_denom) + WEIGHTED_FLAT_P*g_norm)[good_mask]

                m0 += pdf * poisson_pdfs[n][good_mask]
                if n >= 1:
                    m1 += pdf * poisson_pdfs[n-1][good_mask]
                if n >= 2:
                    m2 += pdf * poisson_pdfs[n-2][good_mask]
            m1/=m0
            m2/=m0

            ts[good_mask] += m1 - 1
            tsderiv[good_mask] += m2 - m1*m1

        # Perform the iterative step
        delta = ts / tsderiv

        image -= delta
        image = np.maximum(image, 0)
        p99 = np.nanpercentile(np.abs(delta), 99)
        print("Iteration", iteration+1, "Median shift", np.nanmedian(delta), "Max shift", np.nanmax(np.abs(delta)), "99th percentile shift", p99)

        image_cpy =np.copy(image)
        image_cpy[~np.isfinite(image)] = 0
        plt.imsave(f"image{iteration}.png", image_cpy, vmin=0, vmax=50)

        if p99 < 0.01:
            break
        
    image[~np.isfinite(image)] = 0
    image /= get_qe()(image)
    image /= frame_duration
    if data_set.flat is not None:
        image /= data_set.flat
        image[data_set.flat < FLAT_NAN_THRESHOLD] = np.nan

    return Image(image, data_set)