import numpy as np
from astropy.io import fits
from scipy.special import factorial
from ..regions import Region
from ..ephemeris import Ephemeris

MAX_N_SCALE = 2
SMEAR_FRAME = False # Set to True to smear each frame's flux over the phases for which it is valid. Set to False to give all the flux to the one bin at the middle of the frame.

def delta_phase(phase_start, phase_end):
    if phase_end > phase_start:
        return phase_end - phase_start
    else:
        return 1 - (phase_start - phase_end)

class Lightcurve:
    """
    Class to store light curves and save them

    Parameters
    ----------
    edges : array-like
        Edges of the phase bins of the light curve
    fluxes : array-like
        Flux in each light curve bin. If edges has length N+1, fluxes should have length N
    exposures : array-like
        Time in seconds spent in each bin
    duration : array-like
        Duration of each frame, in seconds
    eph : Ephemeris
        Target ephemeris
    """
    def __init__(self, edges, flux, exposures, eph, data_set):
        self.edges = edges
        self.flux = flux
        self.exposures = exposures

        for frame in data_set:
            self.duration = frame.duration
            break

        self.ephemeris = eph
        self.header0 = data_set.header0
        self.header1 = data_set.header1

    def save(self, filename, clobber=False, save_kwargs=None):
        """
        Save the light curve to a file
        
        Parameters
        ----------
        filename : str
            The file name to which the light curve should be saved
        clobber : bool, optional
            Set to True to allow overwriting
        save_kwargs : dict, optional
            Dictionary of keywords to write to the light curve header
        """
        cols = [
            fits.Column(name='PHASEHI', array=self.edges[1:], format='E'),
            fits.Column(name='PHASELO', array=self.edges[:-1], format='E'),
            fits.Column(name='FLUX', array=self.flux, format='E'),
            fits.Column(name='EXPOSURE', array=self.exposures, format='E'),
        ]
        hdu = fits.BinTableHDU.from_columns(cols)

        hdu.header["EXPTIME"] = np.sum(self.exposures)
        hdu.header["DURATION"] = self.duration
        hdu.header["NU"] = self.ephemeris.nu

        if "GPSSTART" in self.header0:
            hdu.header["GPSSTART"] = self.header0["GPSSTART"]

        for key, value in self.header1.items():
            if key.startswith("HIERARCH") or key.startswith("TEL") or key in ["FILTER", "SHUTTER", "SLIT", "HALPHA", "POLSTAGE", "AIRMASS", "DATEOBS", "TELUT"]:
                if len(key) > 8: key = f"HIERARCH {key}"
                hdu.header[key] = value

        if save_kwargs is not None:
            for key, value in save_kwargs.items():
                if type(value) is list:
                    for i, item in enumerate(value):
                        key = f"{key}{i}"
                        if len(key) > 8: key = f"HIERARCH {key}"
                        hdu.header[f"{key}{i}"] = item
                    continue
                if len(key) > 8: key = f"HIERARCH {key}"
                hdu.header[key] = value

        # Write to file, table in HDU 1
        hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
        hdul.writeto(filename, overwrite=clobber)

def get_bin_weights(phase_edges, start_phase, end_phase):
    """
    The weight is the frame that goes into this bin
    """
    weights = np.zeros(len(phase_edges)-1)
    bin_phase_duration = phase_edges[1] - phase_edges[0]
    phase_duration = delta_phase(start_phase, end_phase)

    # Get the weights at the start and end
    start_index = int(start_phase / bin_phase_duration)
    end_index = int(end_phase / bin_phase_duration)

    if start_index == end_index:
        weights[start_index] += 1
    else:
        weights[start_index] += delta_phase(start_phase, phase_edges[start_index+1]) / phase_duration
        weights[end_index] += delta_phase(phase_edges[end_index], end_phase) / phase_duration

    # Get the weights between the start and end
    if start_index > end_index:
        weights[start_index+1:] += bin_phase_duration / phase_duration
        weights[:end_index] += bin_phase_duration / phase_duration
    else:
        weights[start_index+1:end_index] += bin_phase_duration / phase_duration

    assert(np.abs(np.sum(weights) - 1) < 1e-5)
    
    return weights

def get_summed_lc(data_set, n_bins, reg_file, ephemeris):
    """
    Get the light curve of a source by summing all the detected photons per frame
    
    Parameters
    ----------
    data_set : DataSet
        The data set of the observation
    n_bins : int
        Number of light curve bins to use
    reg_file : str
        The ciao-format, physical coordinate region file containing the source
    ephemeris : Ephemeris
        The source ephemeris
    
    Returns
    -------
    Lightcurve
        The light curve object, corrected for quantum efficiency TODO
    """
    # Does not use the image
    electrons = np.zeros(n_bins)
    exposures = np.zeros(n_bins)
    roi = Region.load(reg_file)
    phase_edges = np.linspace(0, 1, n_bins+1)
    xs, ys = np.meshgrid(np.arange(data_set.image_shape[1]), np.arange(data_set.image_shape[0]))
    roi_mask = roi.check_inside_absolute(xs, ys)

    for frame in data_set:
        counts = np.nanmean(frame.image[roi_mask]) * np.sum(roi_mask)

        if SMEAR_FRAME:
            start_phase = ephemeris.get_phase(frame.timestamp-frame.duration/2)
            end_phase = ephemeris.get_phase(frame.timestamp+frame.duration/2)
            weights = get_bin_weights(phase_edges, start_phase, end_phase)
        else:
            phase = ephemeris.get_phase(frame.timestamp)
            weights = np.zeros(n_bins)
            weights[np.digitize(phase, phase_edges)-1] = 1

        electrons += counts*weights
        exposures += frame.duration*weights

    fluxes = electrons / exposures # Counts per second
    return Lightcurve(phase_edges, fluxes, exposures, ephemeris, data_set)

def get_clipped_lc(data_set, n_bins, reg_file, ephemeris):
    """
    Get the light curve of a source by summing all the detected photons per frame, clipped to zero or 1
    
    Parameters
    ----------
    data_set : DataSet
        The data set of the observation
    n_bins : int
        Number of light curve bins to use
    reg_file : str
        The ciao-format, physical coordinate region file containing the source
    ephemeris : Ephemeris
        The source ephemeris
    
    Returns
    -------
    Lightcurve
        The light curve object, corrected for quantum efficiency TODO
    """
    # Does not use the image
    electrons = np.zeros(n_bins)
    exposures = np.zeros(n_bins)
    roi = Region.load(reg_file)
    phase_edges = np.linspace(0, 1, n_bins+1)
    xs, ys = np.meshgrid(np.arange(data_set.image_shape[1]), np.arange(data_set.image_shape[0]))
    roi_mask = roi.check_inside_absolute(xs, ys)

    for frame in data_set:
        masked_image = np.round(frame.image[roi_mask])
        # masked_image[masked_image >= 2] = 0 # TODO
        counts = np.nanmean(masked_image) * np.sum(roi_mask)

        if SMEAR_FRAME:
            start_phase = ephemeris.get_phase(frame.timestamp-frame.duration/2)
            end_phase = ephemeris.get_phase(frame.timestamp+frame.duration/2)
            weights = get_bin_weights(phase_edges, start_phase, end_phase) # TODO
        else:
            phase = ephemeris.get_phase(frame.timestamp)
            weights = np.zeros(n_bins)
            weights[np.digitize(phase, phase_edges)-1] = 1

        electrons += counts*weights
        exposures += frame.duration*weights

    fluxes = electrons / exposures # Counts per bin
    return Lightcurve(phase_edges, fluxes, exposures, ephemeris, data_set)

def get_weighted_lc_linearized(data_set, image, n_bins, reg_file, ephemeris):
    """
    Get the light curve of a source by summing all the detected photons per frame after weighting by the probability of each being real. This function assumes that the true number of photons expected per pixel is << 1. This function can also do PSF-weighted photometry.
    
    Parameters
    ----------
    data_set : DataSet
        The data set of the observation
    image : array-like or None
        image of PSF weights to use. Set to None to not use PSF weights.
    n_bins : int
        Number of light curve bins to use
    reg_file : str
        The ciao-format, physical coordinate region file containing the source
    ephemeris : Ephemeris
        The source ephemeris
    
    Returns
    -------
    Lightcurve
        The light curve object, corrected for quantum efficiency TODO
    """
    # Does not use the image
    numer = np.zeros(n_bins)
    denom = np.zeros(n_bins)
    exposures = np.zeros(n_bins)
    roi = Region.load(reg_file)
    phase_edges = np.linspace(0, 1, n_bins+1)
    xs, ys = np.meshgrid(np.arange(data_set.image_shape[1]), np.arange(data_set.image_shape[0]))
    roi_mask = roi.check_inside_absolute(xs, ys)

    if image is None:
        image = np.ones(data_set.image_shape)

    for frame in data_set:
        if SMEAR_FRAME:
            start_phase = ephemeris.get_phase(frame.timestamp-frame.duration/2)
            end_phase = ephemeris.get_phase(frame.timestamp+frame.duration/2)
            weights = get_bin_weights(phase_edges, start_phase, end_phase)
        else:
            phase = ephemeris.get_phase(frame.timestamp)
            weights = np.zeros(n_bins)
            weights[np.digitize(phase, phase_edges)-1] = 1

        good_mask = roi_mask & (~np.isnan(frame.image))
        masked_image = frame.image[good_mask]

        p0 = data_set.pixel_properties.get_prob(masked_image, 0, mask=good_mask)
        p1 = data_set.pixel_properties.get_prob(masked_image, 1, mask=good_mask)
        odds = p1/p0
        
        numer += weights*np.sum(odds - 1)
        denom += weights*np.sum(odds**2)
        # TODO put PSF weighting back in, using the `image` variable`.

        exposures += frame.duration*weights

    fluxes = numer / denom # TODO normalization may be wrong
    return Lightcurve(phase_edges, fluxes, exposures, ephemeris, data_set)