import numpy as np
from scipy.special import factorial
from astropy.io import fits

WEIGHTED_FLAT_P = 0.1

def get_clipped_image(data_set):
    duration = np.zeros(data_set.image_shape)
    image = np.zeros(data_set.image_shape)
    for frame in data_set:
        good_mask = ~np.isnan(frame.image)
        image[good_mask] += np.round(frame.image[good_mask])
        duration[good_mask] += frame.duration
    image /= duration

    # Divide by flat
    if data_set.flat is not None:
        image /= data_set.flat

    # Replace nans with medians
    image[np.isnan(image)] = np.nanmedian(image)

    return image

def get_summed_image(data_set):
    duration = np.zeros(data_set.image_shape)
    image = np.zeros(data_set.image_shape)
    for frame in data_set:
        good_mask = ~np.isnan(frame.image)
        image[good_mask] += frame.image[good_mask]
        duration[good_mask] += frame.duration
    image /= duration

    # Divide by flat
    if data_set.flat is not None:
        image /= data_set.flat

    # Replace nans with medians
    image[np.isnan(image)] = np.nanmedian(image)

    return image

def save_image(image, data_set, args):
    hdu = fits.PrimaryHDU(image)
    
    for key, value in vars(args).items():
        if key == "func": continue
        hdu.header[key] = value
    if "GPSSTART" in data_set.header0:
        hdu.header["GPSSTART"] = data_set.header0["GPSSTART"]
    for key, value in data_set.header1.items():
        if key.startswith("HIERARCH") or key.startswith("TEL") or key in ["FILTER", "SHUTTER", "SLIT", "HALPHA", "POLSTAGE", "AIRMASS", "DATEOBS", "TELUT"]:
            if len(key) > 8:
                key = key[-8:]
            hdu.header[key] = value
            
    hdu.writeto(args.output, overwrite=True)

def load_image(image, assert_items):
    with fits.open(image) as hdul:
        for key in assert_items:
            assert(key in hdul[0].header)
            assert(hdul[0].header[key] == assert_items[key])
        return np.array(hdul[0].data)
    
def get_weighted_image(data_set):
    # Get initial image
    image = np.zeros(data_set.image_shape)
    n_frames = np.zeros(data_set.image_shape)
    for frame in data_set:
        frame_duration = frame.duration
        good_mask = ~np.isnan(frame.image)
        image[good_mask] += frame.image[good_mask]
        n_frames[good_mask] += 1
    image /= n_frames
    image = np.maximum(image, 0)

    # Get some simple variables useful for integration
    width_image = data_set.get_pixel_properties().widths
    gauss_denominator = 1 / (2*width_image**2)
    gauss_norm = 1 / np.sqrt(2*np.pi*width_image**2)
    max_n = int(np.nanmax(image)) + 2
    if max_n > 100:
        print(f"WARNING: The image has pixels with >100 electrons per frame (max: {max_n}). Are you sure you want to do weighted extraction? It will take a long time")

    # Pre-initialize some arrays
    ts = np.zeros(data_set.image_shape)
    tsderiv = np.zeros(data_set.image_shape)

    # Run a few Newton iterations
    for iteration in range(4):
        ts *= 0
        tsderiv *= 0

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
                pdf = (np.exp(-delta*delta*gauss_denominator)*gauss_norm)[good_mask]

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

        if p99 < 0.01:
            break
        
    # The image is now an accurate estimate of the number of electrons per frame.

    # Rescale to electrons per second
    # image[image < 0] = np.nan
    # image[image > max_n] = np.nan
    image /= frame_duration

    image[np.isnan(image)] = 0
    
    # Divide by flat
    if data_set.flat is not None:
        image /= data_set.flat

    # Replace nans with medians
    # image[np.isnan(image)] = np.nanmedian(image)

    return image

def get_weighted_image_linearized(data_set):
    numer = np.zeros(data_set.image_shape)
    denom = np.zeros(data_set.image_shape)
    w_denom = 1/(2*data_set.get_pixel_properties().widths**2)
    g_norm = np.sqrt(2*np.pi*data_set.get_pixel_properties().widths**2)
    for frame in data_set:
        frame_duration = frame.duration
        p0 = np.exp(-frame.image**2 * w_denom) + WEIGHTED_FLAT_P*g_norm
        p1 = np.exp(-(frame.image-1)**2 * w_denom) + WEIGHTED_FLAT_P*g_norm
        good_mask = ~np.isnan(frame.image)
        odds = (p1/p0)[good_mask]
        numer[good_mask] += odds - 1
        denom[good_mask] += odds**2

    image = numer/denom
    image /= frame_duration

    # Divide by flat
    if data_set.flat is not None:
        image /= data_set.flat

    # Replace nans with medians
    image[np.isnan(image)] = np.nanmedian(image)

    return image