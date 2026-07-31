import numpy as np
import copy
from ..image.image import Image
from ..weight import Weighter

def get_range(s):
    on, off = s.split(',')
    on_low, on_high = on.split(':')
    off_low, off_high = off.split(":")
    return (float(on_low), float(on_high)), (float(off_low), float(off_high))

def contains_phase(rang, phase):
    if rang[0] < rang[1]:
        return (rang[0] < phase) and (phase < rang[1])
    else:
        return (rang[0] < phase) or (phase < rang[1])

def make_on_off(data_set, ephemeris, phase_string, method):
    """
    Get a bias, dark, flat corrected image from a :class:`DataSet` by summing all the detected photons per frame.
    
    Parameters
    ----------
    data_set : DataSet
        The proto-Lightspeed data set
    ephemeris : Ephemeris
        The ephemeris for which to load 
    phase_string : str
        The string which encodes the phase range. Remember it's formatted as low:high,low_high, where the first section is the on range and the second is the off range.
    method : str
        Either "sum", "clip", or "weight", specifying the method of image generation

    Returns
    -------
    Image
        The image, crrected for flat and quantum efficiency
    """
    on_range, off_range = get_range(phase_string)

    n_pixels = np.prod(data_set.image_shape)
    on_image = np.zeros(data_set.image_shape)
    on_n_frames = np.zeros(data_set.image_shape)
    off_image = np.zeros(data_set.image_shape)
    off_n_frames = np.zeros(data_set.image_shape)
    if method == "weight":
        on_weighter = Weighter(data_set, n_pixels, blur=False)
        off_weighter = Weighter(data_set, n_pixels, blur=False)

    for frame in data_set:
        good_mask = ~np.isnan(frame.image)
        masked_image = frame.image[good_mask]
        phase = ephemeris.get_phase(frame.timestamp-frame.duration/2)
        if contains_phase(on_range, phase):
            if method == "sum":
                on_image[good_mask] += masked_image
            elif method == "clip":
                on_image[good_mask] += np.round(masked_image)
            elif method == "weight":
                indices = np.arange(n_pixels)
                weight_array = np.transpose([indices, np.ones(n_pixels)])
                on_weighter.add_pixels(masked_image, weight_array, good_mask)
            else:
                raise Exception(f"Unrecognized method {method}")
            on_n_frames[good_mask] += 1
                
        if contains_phase(off_range, phase):
            if method == "sum":
                off_image[good_mask] += masked_image
            elif method == "clip":
                off_image[good_mask] += np.round(masked_image)
            elif method == "weight":
                indices = np.arange(n_pixels)
                weight_array = np.transpose([indices, np.ones(n_pixels)])
                off_weighter.add_pixels(masked_image, weight_array, good_mask)
            else:
                raise Exception(f"Unrecognized method {method}")
            off_n_frames[good_mask] += 1

    if method == "weight":
        # One needs to multiply by the number of frames since weighting calculates the mean value and the other methods compute the sum.
        on_image = on_weighter.get_fluxes().reshape(data_set.image_shape) * on_n_frames
        off_image = off_weighter.get_fluxes().reshape(data_set.image_shape) * off_n_frames


    on = Image(on_image, data_set, on_n_frames)
    off = Image(off_image, data_set, off_n_frames)
    image = copy.deepcopy(on)
    image.photons_per_second -= off.photons_per_second

    return image
