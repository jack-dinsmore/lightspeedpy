import numpy as np
from astropy.io import fits
from scipy.special import factorial
from .regions import Region
from .ephemeris import Ephemeris

MAX_N_SCALE = 2

def delta_phase(phase_start, phase_end):
    if phase_end > phase_start:
        return phase_end - phase_start
    else:
        return 1 - (phase_start - phase_end)

class Lightcurve:
    def __init__(self, edges, flux, exposures, duration, eph):
        self.edges = edges
        self.flux = flux
        self.exposures = exposures
        self.duration = duration
        self.ephemeris = eph

    def save(self, data_set, args):
        cols = [
            fits.Column(name='PHASEHI', array=self.edges[1:], format='E'),
            fits.Column(name='PHASELO', array=self.edges[:-1], format='E'),
            fits.Column(name='FLUX', array=self.flux, format='E'),
            fits.Column(name='EXPOSURE', array=self.exposures, format='E'),
        ]
        hdu = fits.BinTableHDU.from_columns(cols)

        hdu.header["EXPOSURE"] = np.sum(self.exposures)
        hdu.header["DURATION"] = self.duration
        hdu.header["NU"] = self.ephemeris.nu
        hdu.header["NUDOT"] = self.ephemeris.nudot
        hdu.header["NUDDOT"] = self.ephemeris.nuddot
        hdu.header["PEPOCH"] = self.ephemeris.pepoch / (3600*24)

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

def get_bin_weights(phase_edges, start_phase, end_phase):
    """
    The weight is the fraction of the bin that goes into this frame.
    """
    weights = np.zeros(len(phase_edges)-1)
    bin_phase_duration = phase_edges[1] - phase_edges[0]

    # Get the weights at the start and end
    start_index = int(start_phase / bin_phase_duration)
    end_index = int(end_phase / bin_phase_duration)

    if start_index == end_index:
        weights[start_index] += delta_phase(start_phase, end_phase) / bin_phase_duration
    else:
        weights[start_index] += delta_phase(start_phase, phase_edges[start_index+1]) / bin_phase_duration
        weights[end_index] += delta_phase(phase_edges[end_index], end_phase) / bin_phase_duration

    # Get the weights between the start and end
    weights[start_index+1:end_index] += 1
    if start_index > end_index:
        weights[start_index+1:] += 1
        weights[:end_index] += 1
    
    return weights

def get_linear_lc(data_set, args):
    electrons = np.zeros(args.bins)
    exposures = np.zeros(args.bins)
    roi = Region.load(args.roi)
    ephemeris = Ephemeris.from_file(args.eph)
    phase_edges = np.linspace(0, 1, args.bins+1)
    bin_time_duration = (phase_edges[1] - phase_edges[0]) / ephemeris.nu
    xs, ys = np.meshgrid(np.arange(data_set.image_shape[1]), np.arange(data_set.image_shape[0]))
    roi_mask = roi.check_inside_absolute(xs, ys)
    
    for frame in data_set:
        bins_per_frame = frame.duration / bin_time_duration
        counts = np.nanmean(frame.image[roi_mask]) * np.sum(roi_mask)

        start_phase = ephemeris.get_phase(frame.timestamp)
        end_phase = ephemeris.get_phase(frame.timestamp+frame.duration)
        weights = get_bin_weights(phase_edges, start_phase, end_phase)

        count_per_bin = counts / bins_per_frame
        electrons += count_per_bin*weights
        exposures += bin_time_duration*weights
        frame_duration = frame.duration

    fluxes = electrons / exposures # Counts per bin
    return Lightcurve(phase_edges, fluxes, exposures, frame_duration, ephemeris)

def get_weighted_lc(data_set, image, args):
    # Set up lightcurve
    lightcurve = np.zeros(args.bins) # Multiplier to the image
    exposures = np.zeros(args.bins)
    roi = Region.load(args.roi)
    ephemeris = Ephemeris.from_file(args.eph)
    phase_edges = np.linspace(0, 1, args.bins+1)
    bin_time_duration = (phase_edges[1] - phase_edges[0]) / ephemeris.nu
    xs, ys = np.meshgrid(np.arange(data_set.image_shape[1]), np.arange(data_set.image_shape[0]))
    roi_mask = roi.check_inside_absolute(xs, ys)

    # Get pixel data
    bad_pixels = roi_mask & np.isnan(image)
    roi_mask &= ~bad_pixels
    pixel_image = image[roi_mask]
    width_image = data_set.get_pixel_properties().widths[roi_mask]
    gauss_denominator = 1 / (2*width_image**2)
    gauss_norm = 1 / np.sqrt(2*np.pi*width_image**2)

    # Get the initial lightcurve
    frame_weights = []
    for frame in data_set:
        bins_per_frame = frame.duration / bin_time_duration

        start_phase = ephemeris.get_phase(frame.timestamp)
        end_phase = ephemeris.get_phase(frame.timestamp+frame.duration)
        weights = get_bin_weights(phase_edges, start_phase, end_phase)
        frame_weights.append(np.copy(weights))
        frame_duration = frame.duration

        # Add to lightcurve
        counts = np.nanmean(frame.image[roi_mask]) * np.sum(roi_mask)
        count_per_bin = counts / bins_per_frame
        lightcurve += count_per_bin*weights
        exposures += bin_time_duration*weights

    lightcurve /= np.mean(lightcurve)
    lightcurve /= bins_per_frame

    pixel_image *= frame_duration # Convert back to electrons per frame

    # Perform the fit

    # Pre-initialize some arrays
    ts = np.zeros(len(lightcurve))
    ts_hessian = np.zeros((len(lightcurve), len(lightcurve)))
    max_n = MAX_N_SCALE*int(np.ceil(np.max(pixel_image))) # The "MAX_N_SCALE" prefactor indicates that the LC maximum is at most double the LC mean. If this is untrue, increase the number (at the cost of performance)

    if max_n > 100:
        print(f"WARNING: The image has pixels with >100 electrons per frame (max: {max_n}). Are you sure you want to do weighted extraction? It will take a long time")

    n_factorial = [factorial(n) for n in range(max_n)]

    # Run a few Newton iterations
    for iteration in range(8):
        ts *= 0
        ts_hessian *= 0

        # Count up the moments
        for frame_num, frame in enumerate(data_set):
            weights = frame_weights[frame_num]
            good_mask = ~np.isnan(frame.image)[roi_mask]
            lambdas = pixel_image[good_mask] * np.sum(weights*lightcurve)

            poisson_pdfs = []
            for n in range(max_n):
                poisson_pdfs.append(lambdas**n / n_factorial[n])
            n_good_pixels = np.sum(good_mask)
            m0 = np.zeros(n_good_pixels)
            m1 = np.zeros(n_good_pixels)
            m2 = np.zeros(n_good_pixels)
            for n in range(max_n):
                delta = frame.image[roi_mask] - n
                pdf = (np.exp(-delta*delta*gauss_denominator)*gauss_norm)[good_mask]

                m0 += pdf * poisson_pdfs[n]
                if n >= 1:
                    m1 += pdf * poisson_pdfs[n-1]
                if n >= 2:
                    m2 += pdf * poisson_pdfs[n-2]
            m1/=m0
            m2/=m0

            ts += np.sum((m1 - 1)*pixel_image[good_mask])*weights
            ts_hessian += np.multiply.outer(weights, weights) * np.sum((m2 - m1*m1)*pixel_image[good_mask]**2)

        ts_inv_hessian = np.linalg.pinv(ts_hessian)

        # import matplotlib.pyplot as plt # TODO
        # fig, axs = plt.subplots(ncols=2)
        # axs[0].imshow(ts_hessian, vmax=0)
        # vmax = np.max(np.abs(ts_inv_hessian))
        # axs[1].imshow(ts_inv_hessian, vmin=-vmax, vmax=vmax, cmap="RdBu")
        # fig.savefig("dbg.png")
        
        # Perform the iterative step
        delta = ts_inv_hessian @ ts

        lightcurve -= delta
        lightcurve = np.maximum(lightcurve, 0.1)
        p99 = np.nanpercentile(np.abs(delta), 99)
        print("Iteration", iteration, "Median shift", np.nanmedian(delta), "Max shift", np.nanmax(np.abs(delta)), "99th percentile shift", p99)

        import matplotlib.pyplot as plt # TODO
        fig, ax = plt.subplots()
        ax.plot((phase_edges[1:] + phase_edges[:-1]) / 2, lightcurve)
        fig.savefig(f"dbg-{iteration}.png")

        if p99 < 0.001:
            break

    lightcurve[lightcurve / np.nanmean(lightcurve) > MAX_N_SCALE] = np.nan

    # Convert lightcurve from a fraction to counts
    lightcurve *= np.sum(pixel_image) # Now counts per frame
    lightcurve /= frame_duration # Now counts per second

    return Lightcurve(phase_edges, lightcurve, exposures, frame_duration, ephemeris)