import numpy as np
from scipy.special import factorial
from astropy.io import fits
from ..qe import get_qe

WEIGHTED_FLAT_P = 0.1
FLAT_NAN_THRESHOLD = 0.1
FORBIDDEN_KEYWORDS = "XTENSION BITPIX NAXIS NAXIS1 NAXIS2 NAXIS3 PCOUNT GCOUNT BSCALE BZERO EXTNAME".split()

def get_clipped_image(data_set):
    image = np.zeros(data_set.image_shape)
    duration = np.zeros(data_set.image_shape)
    n_frames = np.zeros(data_set.image_shape)
    for frame in data_set:
        good_mask = ~np.isnan(frame.image)
        image[good_mask] += np.round(frame.image[good_mask])
        duration[good_mask] += frame.duration
        n_frames[good_mask] += 1

    image /= get_qe()(image/n_frames)
    image /= duration
    image[np.isnan(image)] = np.nanmedian(image)
    if data_set.flat is not None:
        image /= data_set.flat
        image[data_set.flat < FLAT_NAN_THRESHOLD] = np.nan

    return image

def get_summed_image(data_set):
    duration = np.zeros(data_set.image_shape)
    image = np.zeros(data_set.image_shape)
    n_frames = np.zeros(data_set.image_shape)
    for frame in data_set:
        good_mask = ~np.isnan(frame.image)
        image[good_mask] += frame.image[good_mask]
        duration[good_mask] += frame.duration
        n_frames[good_mask] += 1

    image /= get_qe()(image/n_frames)
    image /= duration
    image[np.isnan(image)] = np.nanmedian(image)
    if data_set.flat is not None:
        image /= data_set.flat
        image[data_set.flat < FLAT_NAN_THRESHOLD] = np.nan

    return image

def save_image(image, data_set, args):
    hdu = fits.PrimaryHDU(image)

    for frame in data_set:
        frame_duration = frame.duration
        break
    exposure = data_set.num_frames() * frame_duration
    
    # Write header
    for key, value in vars(args).items():
        if key == "func": continue
        if type(value) == list:
            for v in value:
                key = f"{key}i"
                if len(key) > 8: key = f"HIERARCH {key}"
                hdu.header[f"{key}i"] = v
            continue
        if len(key) > 8: key = f"HIERARCH {key}"
        hdu.header[key] = value
    if "GPSSTART" in data_set.header0:
        hdu.header["GPSSTART"] = data_set.header0["GPSSTART"]
    for key, value in data_set.header1.items():
        if key not in FORBIDDEN_KEYWORDS:
            if len(key) > 8: key = f"HIERARCH {key}"
            hdu.header[key] = value
    hdu.header["EXPTIME"] = exposure
            
    hdu.writeto(args.output, overwrite=True)

def load_image(image, assert_items):
    with fits.open(image) as hdul:
        for key in assert_items:
            assert(key in hdul[0].header)
            assert(hdul[0].header[key] == assert_items[key])
        return np.array(hdul[0].data)
    
def get_weighted_image(data_set):
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

    return image

def get_weighted_image_linearized(data_set):
    numer = np.zeros(data_set.image_shape)
    denom = np.zeros(data_set.image_shape)
    pixel_properties = data_set.runs[0].get_pixel_properties() # TODO assumes these all have the same properties
    w_denom = 1/(2*pixel_properties.widths**2)
    g_norm = np.sqrt(2*np.pi*pixel_properties.widths**2)
    for frame in data_set:
        frame_duration = frame.duration
        p0 = np.exp(-frame.image**2 * w_denom) + WEIGHTED_FLAT_P*g_norm
        p1 = np.exp(-(frame.image-1)**2 * w_denom) + WEIGHTED_FLAT_P*g_norm
        good_mask = ~np.isnan(frame.image)
        odds = (p1/p0)[good_mask]
        numer[good_mask] += odds - 1
        denom[good_mask] += odds**2

    image = numer/denom

    # Divide by flat
    image /= get_qe()(image)
    image /= frame_duration
    image[np.isnan(image)] = np.nanmedian(image)
    if data_set.flat is not None:
        image /= data_set.flat
        image[data_set.flat < FLAT_NAN_THRESHOLD] = np.nan

    return image