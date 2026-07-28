import numpy as np
from astropy.io import fits
import copy
from multiprocessing import Pool
from ..cli import get_dataset
from ..regions import Region, CircleRegion, EllipseRegion
from ..ephemeris import Ephemeris
from ..constants import FORBIDDEN_KEYWORDS
from ..weight import Weighter

MAX_N_SCALE = 2
SMEAR_FRAME = False # Set to True to smear each frame's flux over the phases for which it is valid. Set to False to give all the flux to the one bin at the middle of the frame.

def make_psf_image(data_set, reg_file):
    """
    Returns an image which contains the PSF weights for each pixel given the PSF shape contained in reg_file.

    Parameters
    ----------
    data_set: DataSet
        Data set to be processed. This is used to get the shape of the full frame
    reg_file: str
        File that contains the PSF shape. The shape should be a ciao-format region file, either circular or elliptical, which should define the FWHM.
    """
    xs, ys = np.meshgrid(np.arange(data_set.image_shape[1]), np.arange(data_set.image_shape[0]))
    region = Region.load(reg_file)
    if type(region) is CircleRegion:
        fwhm_pixels = np.sqrt(region.radius2)
        sigma_x = fwhm_pixels / 2.34
        sigma_y = fwhm_pixels / 2.34
        theta = 0
    elif type(region) is EllipseRegion:
        sigma_x = region.a / 2.34
        sigma_y = region.b / 2.34
        theta = region.angle
    else:
        raise Exception("PSF weighting can only be performed with elliptical or circular regions")
    
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    inv_cov = rot @ np.diag([1/sigma_x**2, 1/sigma_y**2]) @ np.transpose(rot)
    vec = np.array([xs - region.x, ys - region.y])
    gauss_exp = np.einsum("iab,ij,jab->ab", vec, inv_cov, vec)
    image = np.exp(-gauss_exp / 2)

    return image

def delta_phase(phase_start, phase_end):
    """
    Get the difference in phase between start and end, accounting for phases that wrap around.
    """
    if phase_end > phase_start:
        return phase_end - phase_start
    else:
        return 1 - (phase_start - phase_end)

def get_bootstrap_instance(seed, data_set_orig, ephemeris, roi, psf_image, args):
    """
    Get a light curve from a randomly drawn boostrapped sample of the data set
    """
    data_set = copy.deepcopy(data_set_orig)
    data_set.bootstrap(seed)
    lc = make_lc(data_set, args.bins, roi, ephemeris, args.mode, psf_image)
    return lc

def get_lc(args):
    """
    Run the light curve extraction program
    """
    data_set = get_dataset(args)
    print("Load files")
    data_set.display_filenames()
    ephemeris = Ephemeris(args.eph, data_set, args.observatory)
    roi = Region.load(args.roi)
    psf_image = None
    if args.psf:
        psf_image = make_psf_image(data_set, args.roi)
        # Triple the size of the PSF to get the ROI
        if type(roi) is CircleRegion:
            roi.radius2 *= 3**2
        elif type(roi) is EllipseRegion:
            roi.a *= 3
            roi.b *= 3
        else:
            raise Exception("PSF weighting can only be performed with elliptical or circular regions")

    if args.errors is None:
        lc = make_lc(data_set, args.bins, roi, ephemeris, args.mode, psf_image)
    else:
        N_LCS = 8 # TODO
        
        params = []
        for _ in range(N_LCS):
            params.append([np.random.randint(2**32), data_set, ephemeris, roi, psf_image, args])
        
        with Pool() as pool:
            lcs = pool.starmap(get_bootstrap_instance, params)

        lc_m0 = np.zeros_like(lcs[0].flux)
        lc_m1 = np.zeros_like(lcs[0].flux)
        lc_m2 = np.zeros_like(lcs[0].flux)
        for lc in lcs:
            lc_m0 += lc.exposures
            lc_m1 += lc.flux * lc.exposures
            lc_m2 += lc.flux * lc.flux * lc.exposures
        lc_m1 /= lc_m0
        lc_m2 /= lc_m0
        lc_std = np.sqrt(lc_m2 - lc_m1**2) * np.sqrt(N_LCS / (N_LCS - 1))

        main_lc = lcs[0]
        main_lc.exposures = lc_m0 / N_LCS
        main_lc.flux = lc_m1 * main_lc.exposures
        main_lc.errors = lc_std * main_lc.exposures
        lc = main_lc

    save_kwargs = vars(args)
    if "func" in save_kwargs: del save_kwargs["func"]
    lc.save(args.output, args.clobber, save_kwargs)


def add_lc(args):
    """
    Run the light curve addition program
    """
    lc = None
    for arg in args.inputs:
        if lc is None:
            lc = Lightcurve.load(arg)
        else:
            lc += Lightcurve.load(arg)
    lc.save(args.output, clobber=args.clobber)

class Lightcurve:
    """
    Class to store light curves and save them
    """
    def __init__(self, edges, flux, exposures, nu, header0, header1, duration, errors=None):
        self.edges = edges
        self.flux = flux
        self.exposures = exposures
        if errors is None:
            self.errors = np.zeros(len(flux))
        else:
            self.errors = errors
        self.duration = duration
        self.nu = nu
        self.header0 = header0
        self.header1 = header1

    def from_data_set(data_set, edges, flux, exposures, eph):
        """
        Create a light curve object from a data set

        Parameters
        ----------
        data_set: DataSet
            Data set object
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
        for frame in data_set.iterator(bar_color=None):
            duration = frame.duration
            break
        return Lightcurve(edges, flux, exposures, eph.nu, data_set.header0, data_set.header1, duration)

    def load(filename):
        with fits.open(filename) as hdul:
            header0 = hdul[0].header
            header1 = hdul[1].header
            edges = np.array(hdul[1].data["PHASELO"])
            edges = np.append(edges, hdul[1].data["PHASEHI"][-1])
            flux = np.array(hdul[1].data["FLUX"])
            errors = np.array(hdul[1].data["ERROR"])
            exposures = np.array(hdul[1].data["EXPOSURE"])
            duration = hdul[1].header["DURATION"]
            nu = hdul[1].header["NU"]
        return Lightcurve(edges, flux, exposures, nu, header0, header1, duration, errors)
    
    def __iadd__(self, other):
        if len(self.edges) != len(other.edges) or np.any(self.edges != other.edges):
            raise Exception("Cannot add light curves with different edges")
        self_weight = self.exposures / (self.exposures + other.exposures)
        other_weight = other.exposures / (self.exposures + other.exposures)
        self.exposures += other.exposures
        self.flux = self.flux * self_weight + other.flux * other_weight
        self.flux = self.flux * self_weight + other.flux * other_weight
        self.errors = np.sqrt(self.errors**2 * self_weight**2 + other.errors**2 * other_weight**2)
        if self.duration != other.duration:
            self.duration = 0
        return self

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
            fits.Column(name='ERROR', array=self.errors, format='E'),
            fits.Column(name='EXPOSURE', array=self.exposures, format='E'),
        ]
        hdu = fits.BinTableHDU.from_columns(cols)

        hdu.header["EXPTIME"] = np.sum(self.exposures)
        hdu.header["DURATION"] = self.duration
        hdu.header["NU"] = self.nu

        if "GPSSTART" in self.header0:
            hdu.header["GPSSTART"] = self.header0["GPSSTART"]

        for key, value in self.header1.items():
            if key in FORBIDDEN_KEYWORDS: continue
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
    Gets an array of light curve bin weights. The weight is the frame fraction that goes into this bin

    Parameters
    ----------
    phase_edges : array
        The edges of the light curve phase bins
    start_phase : float
        The phase at the start of the frame
    end_phase : float
        The phase at the end of the frame
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

def make_lc(data_set, n_bins, roi, ephemeris, method, psf_image=None):
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
    method : str
        Either "sum", "clip", or "weight", specifying the method of LC generation
    psf_image : array (optional)
        PSF image used for weighting. The image is only used when doing pixel weighting.
    
    Returns
    -------
    Lightcurve
        The light curve object, corrected for quantum efficiency TODO
    """

    electrons = np.zeros(n_bins)
    exposures = np.zeros(n_bins)
    phase_edges = np.linspace(0, 1, n_bins+1)
    xs, ys = np.meshgrid(np.arange(data_set.image_shape[1]), np.arange(data_set.image_shape[0]))
    roi_mask = roi.check_inside_absolute(xs, ys)
    if psf_image is None:
        psf_image = np.ones(data_set.image_shape)

    if method == "weight":
        one_to_one = np.all(psf_image == 1) and not SMEAR_FRAME
        weighter = Weighter(data_set, one_to_one=one_to_one)

    for frame in data_set:
        masked_image = frame.image[roi_mask]
        
        if SMEAR_FRAME:
            start_phase = ephemeris.get_phase(frame.timestamp-frame.duration/2)
            end_phase = ephemeris.get_phase(frame.timestamp+frame.duration/2)
            weights = get_bin_weights(phase_edges, start_phase, end_phase)
        else:
            phase = ephemeris.get_phase(frame.timestamp)
            weights = np.zeros(n_bins)
            weights[np.digitize(phase, phase_edges)-1] = 1

        exposures += frame.duration*weights
        if method == "sum":
            electrons += np.nansum(masked_image) * weights
        elif method == "clip":
            electrons += np.nansum(np.round(masked_image)) * weights
        elif method == "weight":
            if one_to_one:
                weighter.add_pixels(masked_image, np.ones(len(masked_image), dtype=int) * np.argmax(weights), roi_mask)
            else:
                psf_weights = psf_image[roi_mask]
                psf_weights /= np.sum(psf_weights)
                weight_matrix = np.multiply.outer(psf_weights, weights)
                weighter.add_pixels(masked_image, weight_matrix, roi_mask)
        else:
            raise Exception(f"Unrecognized method {method}")

    if method == "sum" or method == "clip":
        fluxes = electrons / exposures # Counts per second
    else:
        fluxes = weighter.get_fluxes() / frame.duration
        # TODO the fluxes are the wrong size
    return Lightcurve.from_data_set(data_set, phase_edges, fluxes, exposures, ephemeris)