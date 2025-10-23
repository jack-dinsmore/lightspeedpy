import numpy as np
from scipy.fft import fft, fftfreq
from astropy.io import fits
from .regions import Region
from .ephemeris import Ephemeris

class Fft:
    def __init__(self, counts, spacing, exposure):
        coeffs = fft(counts)
        self.powers = 2 * np.abs(coeffs)**2 / np.sum(counts)
        self.freqs = fftfreq(len(counts), spacing)
        self.powers = self.powers[self.freqs > 1]
        self.freqs = self.freqs[self.freqs > 1]
        self.exposure = exposure

    def save(self, data_set, args):
        cols = [
            fits.Column(name='FREQ', array=self.freqs, format='E'),
            fits.Column(name='POWER', array=self.powers, format='E'),
        ]
        hdu = fits.BinTableHDU.from_columns(cols)

        hdu.header["EXPOSURE"] = self.exposure

        for key, value in vars(args).items():
            if key == "func": continue
            hdu.header[key] = value
        if "GPSSTART" in data_set.header0:
            hdu.header["GPSSTART"] = data_set.header0["GPSSTART"]
        for key, value in data_set.header1.items():
            if key.startswith("HIERARCH") or key.startswith("TEL") or key in ["FILTER", "SHUTTER", "SLIT", "HALPHA", "POLSTAGE", "AIRMASS", "DATEOBS", "TELUT"]:
                hdu.header[key] = value

        # Write to file, table in HDU 1
        hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
        hdul.writeto(args.output, overwrite=True)

def get_linear_fft(data_set, args):
    roi = Region.load(args.roi)
    xs, ys = np.meshgrid(np.arange(data_set.image_shape[1]), np.arange(data_set.image_shape[0]))
    roi_mask = roi.check_inside_absolute(xs, ys)
    
    counts = []
    exposure = 0
    for frame in data_set:
        counts.append(np.nanmean(frame.image[roi_mask]) * np.sum(roi_mask))
        frame_duration = frame.duration
        exposure += frame.duration

    return Fft(counts, frame_duration, exposure)