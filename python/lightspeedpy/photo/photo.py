import numpy as np
import os, copy
from multiprocessing import Pool
from astropy.io import fits
from astropy.time import Time
from ..cli import get_dataset
from ..regions import Region
from ..constants import FORBIDDEN_KEYWORDS
from ..weight import Weighter, PixelLayout

N_BOOTSTRAP = 8 # Number of LCs to use for boostrapped errors

def get_photometry(args):
    data_set = get_dataset(args)
    print("Load files")
    data_set.display_filenames()

    src_reg = Region.load(args.src)
    if args.bkg is None:
        raise NotImplementedError()
    else:
        bkg_reg = Region.load(args.bkg)

    if args.errors is None:
        photometry = make_photometry(data_set, src_reg, bkg_reg, args.mode, args.rebin, args.n_electrons, args.n_iterations)
        save_name = args.output
        is_slurm = False
    else:
        if "SLURM_ARRAY_TASK_ID" in os.environ and os.environ["SLURM_ARRAY_TASK_ID"] != "":
            photometry = make_photometry(data_set, src_reg, bkg_reg, args.mode, args.rebin, args.n_electrons, args.n_iterations, seed=np.random.randint(2**32))
            if not os.path.exists(args.output[:-5]):
                os.mkdir(args.output[:-5])
            save_name = f"{args.output[:-5]}/{os.environ["SLURM_ARRAY_TASK_ID"]}.fits"
            is_slurm = True
        else:
            params = []
            for _ in range(N_BOOTSTRAP):
                params.append([data_set, src_reg, bkg_reg, args.mode, args.rebin, args.n_electrons, args.n_iterations, np.random.randint(2**32)])
            with Pool() as pool:
                photos = pool.starmap(make_photometry, params)
            photometry = accumulate_bootstrap_photos(photos)
            save_name = args.output
            is_slurm = False

    save_kwargs = vars(args)
    if "func" in save_kwargs: del save_kwargs["func"]
    photometry.save(save_name, clobber=args.clobber, save_kwargs=save_kwargs)

    # Perform last step of SLURM light curve addition
    if is_slurm:
        photos = []
        for task_id in range(1, int(os.environ["SLURM_ARRAY_TASK_COUNT"])+1):
            filename = f"{args.output[:-5]}/{task_id}.fits"
            if not os.path.exists(filename):
                # Not all the threads have finished yet
                return
            photos.append(Photometry.load(filename))

        # Save the accumulated light curve
        photo = accumulate_bootstrap_photos(photos)
        photo.save(args.output, args.clobber, save_kwargs)
    
def accumulate_bootstrap_photos(photos):
    """
    Accumulate bootstrapped LCs into one LC with errors
    """
    lc_m1 = np.zeros_like(photos[0].flux)
    lc_normalized_m1 = np.zeros_like(photos[0].flux)
    lc_normalized_m2 = np.zeros_like(photos[0].flux)
    for lc in photos:
        lc_normalized = np.copy(lc.flux)
        lc_normalized -= np.min(lc_normalized)
        lc_normalized /= np.max(lc_normalized)
        lc_m1 += lc.flux
        lc_normalized_m1 += lc_normalized
        lc_normalized_m2 += lc_normalized * lc_normalized
    lc_m1 /= N_BOOTSTRAP
    lc_normalized_m1 /= N_BOOTSTRAP
    lc_normalized_m2 /= N_BOOTSTRAP
    lc_std = np.sqrt(lc_normalized_m2 - lc_normalized_m1**2) * np.sqrt(N_BOOTSTRAP / (N_BOOTSTRAP - 1))
    lc_std *= np.nanmax(lc_m1) - np.nanmin(lc_m1)# Convert the error back to normal light curve space

    main_lc = photos[0]
    main_lc.flux = lc_m1
    main_lc.errors = lc_std
    return main_lc

class Photometry:
    def __init__(self, times, flux, header0, header1, duration, mjdrefi, errors=None):
        self.times = times # seconds
        self.flux = flux
        if errors is None:
            self.errors = np.zeros_like(self.flux)
        else:
            self.errors = errors
        self.mjdrefi = mjdrefi
        self.duration = duration
        self.header0=header0
        self.header1=header1

    def from_data_set(data_set, times, flux, duration, mjdrefi):
        return Photometry(times, flux, data_set.header0, data_set.header1, duration, mjdrefi)

    def save(self, filename, clobber=False, save_kwargs=None):
        """
        Save the photometry to a file
        
        Parameters
        ----------
        filename : str
            The file name to which the photometry data should be saved
        clobber : bool, optional
            Set to True to allow overwriting
        save_kwargs : dict, optional
            Dictionary of keywords to write to the file header
        """
        cols = [
            fits.Column(name='TIMEHI', array=self.times[1:], format='D'),
            fits.Column(name='TIMELO', array=self.times[:-1], format='D'),
            fits.Column(name='FLUX', array=self.flux, format='D'),
            fits.Column(name='ERROR', array=self.errors, format='D'),
        ]
        hdu = fits.BinTableHDU.from_columns(cols)

        hdu.header["EXPTIME"] = np.sum(self.times[-1] - self.times[0])
        hdu.header["DURATION"] = self.duration
        hdu.header["MJDREFI"] = self.mjdrefi

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

    def load(self, filename):
        """
        Load a photometry analysis from a fits file
        
        Parameters
        ----------
        filename : str
            Name of the photometry file
        """
        with fits.open(filename) as hdul:
            header0 = hdul[0].header
            header1 = hdul[1].header
            edges = np.array(hdul[1].data["TIMELO"])
            edges = np.append(edges, hdul[1].data["TIMEHI"][-1])
            flux = np.array(hdul[1].data["FLUX"])
            errors = np.array(hdul[1].data["ERROR"])
            duration = hdul[1].header["DURATION"]
            mjdrefi = int(hdul[1].header["MJDREFI"])

        return Photometry(edges, flux, header0, header1, duration, mjdrefi, errors=errors)

def make_photometry(data_set, src_reg, bkg_reg, mode, rebin, n_electrons=3, n_iterations=25, seed=None):
    """
    Analyze the photometry of a data set by adding the target flux, with a comp star as a calibrator

    Parameters
    ----------
    data_set : DataSet
        The data set of the observation
    src_reg: Region
        The region to be used for the target
    bkg_reg: Region
        The region to be used for the target background. If set to None, PSF weighting will be used
    mode: str
        The mode to extract photometry with
    rebin: int
        How many bins to rebin the LC with

    Returns
    -------
    Photometry
        The photometry object, corrected for internal quantum efficiency
    """

    xs, ys = np.meshgrid(np.arange(data_set.image_shape[1]), np.arange(data_set.image_shape[0]))
    src_mask = src_reg.contains(xs, ys)
    bkg_mask = bkg_reg.contains(xs, ys)
    background_ratio = np.sum(src_mask) / np.sum(bkg_mask)
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = None

    if mode == "weight":
        # Figure out which pixels are in source and which are in background
        pixel_assignment = np.zeros(data_set.image_shape, int)
        pixel_assignment[src_mask] = 1
        pixel_assignment[bkg_mask] = 2
        mask = pixel_assignment != 0
        pixel_assignment = pixel_assignment[mask]

        # Make the weight matrix
        weight_matrix = np.zeros((len(pixel_assignment), 2))
        weight_matrix[pixel_assignment==1,0] = 1 / np.sum(src_mask)
        weight_matrix[pixel_assignment==2,1] = 1 / np.sum(bkg_mask)
        layout = PixelLayout.image(data_set, mask=mask)
        weighter = Weighter(layout, n_electrons, weight_matrix)

    times = []
    fluxes = []
    duration = None
    mjd_refi = None
    rebin_index = 0
    flux = 0
    for frame in data_set:
        if duration is None:
            mjd_refi = Time(int(frame.timestamp.mjd), format="mjd", scale=frame.timestamp.scale)
            duration = frame.duration
        frame_start = (frame.timestamp-mjd_refi).jd * 86_400
        frame_start -= frame.duration/2
        if duration is None:
            times.append(frame_start)

        rebin_index = (rebin_index + 1) % rebin

        masked_src = frame.image[src_mask & np.isfinite(frame.image)]
        masked_bkg = frame.image[bkg_mask & np.isfinite(frame.image)]

        if rng is not None:
            # Bootstrap
            masked_src = rng.choice(masked_src, len(masked_src))
            masked_bkg = rng.choice(masked_bkg, len(masked_bkg))

        if mode == "sum":
            flux += np.nansum(masked_src) - np.nansum(masked_bkg)*background_ratio
        elif mode == "clip":
            flux += np.nansum(np.round(masked_src)) - np.nansum(np.round(masked_bkg))*background_ratio
        elif mode == "weight":
            if rng is None:
                weighter.add_pixels(frame.image[mask])
            else:
                arg_selection = rng.integers(0, np.sum(mask), np.sum(mask))
                weighter.add_pixels(frame.image[mask][arg_selection], histogram_indices=arg_selection)
        else:
            raise Exception(f"Unrecognized method {mode}")
        if rebin_index == 0:
            if mode == "weight":
                (src_flux, bkg_flux) = weighter.get_fluxes(n_iterations)
                flux = src_flux - bkg_flux*background_ratio
                weighter.clear()
            times.append(frame_start + frame.duration)
            fluxes.append(flux / frame.duration)
            flux = 0

    fluxes = np.array(fluxes) # Source photons per second
    times = np.array(times)

    return Photometry.from_data_set(data_set, times, fluxes, duration*rebin, int(np.round(mjd_refi.mjd)))