import numpy as np
from ..cli import get_dataset
from ..regions import Region
from astropy.io import fits
from ..constants import FORBIDDEN_KEYWORDS
from ..weight import Weighter

def get_photometry(args):
    data_set = get_dataset(args)
    print("Load files")
    data_set.display_filenames()

    comp_reg = Region.load(args.comp)
    targ_reg = Region.load(args.targ)
    if args.comp_bg is None:
        raise NotImplementedError()
    else:
        comp_bg_reg = Region.load(args.comp_bg)

    if args.targ_bg is None:
        raise NotImplementedError()
    else:
        targ_bg_reg = Region.load(args.targ_bg)

    photometry = make_photometry(data_set, comp_reg, targ_reg, comp_bg_reg, targ_bg_reg, args.mode, args.rebin, args.n_electrons, args.n_iterations)
    photometry.save(args.output)

class Photometry:
    def __init__(self, times, flux, header0, header1, duration):
        self.times = times
        self.flux = flux
        self.duration = duration
        self.header0=header0
        self.header1=header1

    def from_data_set(data_set, times, flux, duration):
        return Photometry(times, flux, data_set.header0, data_set.header1, duration)

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
            fits.Column(name='TIMEHI', array=self.times[1:], format='E'),
            fits.Column(name='TIMELO', array=self.times[:-1], format='E'),
            fits.Column(name='FLUX', array=self.flux, format='E'),
        ]
        hdu = fits.BinTableHDU.from_columns(cols)

        hdu.header["EXPTIME"] = np.sum(self.times[-1] - self.times[0])
        hdu.header["DURATION"] = self.duration

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
            duration = hdul[1].header["DURATION"]
        return Photometry(edges, flux, header0, header1, duration)

def make_photometry(data_set, comp_reg, targ_reg, comp_bg_reg, targ_bg_reg, mode, rebin, n_electrons=3, n_iterations=25):
    """
    Analyze the photometry of a data set by adding the target flux, with a comp star as a calibrator

    Parameters
    ----------
    data_set : DataSet
        The data set of the observation
    comp_reg: Region
        The region to be used for the comp star
    targ_reg: Region
        The region to be used for the target
    comp_bg_reg: Region
        The region to be used for the comp star background. If set to None, PSF weighting will be used
    targ_bg_reg: Region
        The region to be used for the target background. If set to None, PSF weighting will be used

    Returns
    -------
    Photometry
        The photometry object, corrected for internal quantum efficiency (TODO)
    """

    xs, ys = np.meshgrid(np.arange(data_set.image_shape[1]), np.arange(data_set.image_shape[0]))
    comp_mask = comp_reg.contains(xs, ys)
    targ_mask = targ_reg.contains(xs, ys)
    comp_bg_mask = comp_bg_reg.contains(xs, ys)
    targ_bg_mask = targ_bg_reg.contains(xs, ys)
    comp_inv_area = 1 / comp_reg.area()
    targ_inv_area = 1 / targ_reg.area()
    comp_bg_inv_area = 1 / comp_bg_reg.area()
    targ_bg_inv_area = 1 / targ_bg_reg.area()

    if mode == "weight":
        comp_weighter = Weighter(data_set, 1, n_electrons, blur=False)
        targ_weighter = Weighter(data_set, 1, n_electrons, blur=False)

    times = []
    flux = []
    duration = None
    rebin_index = 0
    comp_flux = 0
    targ_flux = 0
    for frame in data_set:
        if duration is None:
            duration = frame.duration
            times.append(frame.timestamp)

        rebin_index = (rebin_index + 1) % rebin
        
        masked_comp = frame.image[comp_mask & np.isfinite(frame.image)]
        masked_targ = frame.image[targ_mask & np.isfinite(frame.image)]
        masked_comp_bg = frame.image[comp_bg_mask & np.isfinite(frame.image)]
        masked_targ_bg = frame.image[targ_bg_mask & np.isfinite(frame.image)]

        if mode == "sum":
            comp_flux += np.nansum(masked_comp) * comp_inv_area - np.nansum(masked_comp_bg) * comp_bg_inv_area
            targ_flux += np.nansum(masked_targ) * targ_inv_area - np.nansum(masked_targ_bg) * targ_bg_inv_area
        elif mode == "clip":
            comp_flux += np.nansum(np.round(masked_comp)) * comp_inv_area - np.nansum(np.round(masked_comp_bg)) * comp_bg_inv_area
            targ_flux += np.nansum(np.round(masked_targ)) * targ_inv_area - np.nansum(np.round(masked_targ_bg)) * targ_bg_inv_area
        elif mode == "weight":
            comp_weights = (
                np.arange(len(masked_comp) + len(masked_comp_bg)),
                np.concatenate([np.ones(len(masked_comp)) * comp_inv_area, np.ones(len(masked_comp_bg)) * -comp_bg_inv_area]),
            )
            targ_weights = (
                np.arange(len(masked_targ) + len(masked_targ_bg)),
                np.concatenate([np.ones(len(masked_targ)) * targ_inv_area, np.ones(len(masked_targ_bg)) * -targ_bg_inv_area]),
            )
            comp_weighter.add_pixels(np.concatenate([masked_comp, masked_comp_bg]), comp_weights, (comp_mask | comp_bg_mask) & np.isfinite(frame.image))
            targ_weighter.add_pixels(np.concatenate([masked_targ, masked_targ_bg]), targ_weights, (targ_mask | targ_bg_mask) & np.isfinite(frame.image))
        else:
            raise Exception(f"Unrecognized method {mode}")
        if rebin_index == 0:
            if mode == "weight":
                comp_flux += comp_weighter.get_fluxes(n_iterations)
                targ_flux += targ_weighter.get_fluxes(n_iterations)
                targ_weighter.clear()
                comp_weighter.clear()
            times.push(frame.timestamp + duration)
            flux.push(targ_flux / comp_flux)
            comp_flux = 0
            targ_flux = 0

    return Photometry.from_data_set(data_set, times, flux, duration*rebin)