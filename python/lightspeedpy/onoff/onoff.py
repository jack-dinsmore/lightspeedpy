import numpy as np
from ..image.image import Image
import copy

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

def get_clipped_on_off(data_set, ephemeris, phase_string):
    """
    Get a bias, dark, flat corrected image from a :class:`DataSet` by summing all the detected photons per frame, clipped to zero or 1.
    
    Parameters
    ----------
    data_set : DataSet
        The proto-Lightspeed data set

    Returns
    -------
    array-like
        The image, crrected for flat and quantum efficiency
    """
    on_range, off_range = get_range(phase_string)

    on_image = np.zeros(data_set.image_shape)
    on_n_frames = np.zeros(data_set.image_shape)
    off_image = np.zeros(data_set.image_shape)
    off_n_frames = np.zeros(data_set.image_shape)
    for frame in data_set:
        good_mask = ~np.isnan(frame.image)
        phase = ephemeris.get_phase(frame.timestamp-frame.duration/2)
        if contains_phase(on_range, phase):
            on_image[good_mask] += np.round(frame.image[good_mask])
            on_n_frames[good_mask] += 1
        if contains_phase(off_range, phase):
            off_image[good_mask] += np.round(frame.image[good_mask])
            off_n_frames[good_mask] += 1

    on = Image(on_image, data_set, on_n_frames)
    off = Image(off_image, data_set, off_n_frames)
    image = copy.deepcopy(on)
    image.photons_per_second -= off.photons_per_second

    return image

def get_summed_on_off(data_set, ephemeris, phase_string):
    """
    Get a bias, dark, flat corrected image from a :class:`DataSet` by summing all the detected photons per frame.
    
    Parameters
    ----------
    data_set : DataSet
        The proto-Lightspeed data set

    Returns
    -------
    array-like
        The image, crrected for flat and quantum efficiency
    """
    on_range, off_range = get_range(phase_string)

    on_image = np.zeros(data_set.image_shape)
    on_n_frames = np.zeros(data_set.image_shape)
    off_image = np.zeros(data_set.image_shape)
    off_n_frames = np.zeros(data_set.image_shape)
    for frame in data_set:
        good_mask = ~np.isnan(frame.image)
        phase = ephemeris.get_phase(frame.timestamp-frame.duration/2)
        if contains_phase(on_range, phase):
            on_image[good_mask] += frame.image[good_mask]
            on_n_frames[good_mask] += 1
        if contains_phase(off_range, phase):
            off_image[good_mask] += frame.image[good_mask]
            off_n_frames[good_mask] += 1

    on = Image(on_image, data_set, on_n_frames)
    off = Image(off_image, data_set, off_n_frames)
    image = copy.deepcopy(on)
    image.photons_per_second -= off.photons_per_second

    return image

def get_weighted_on_off_linearized(data_set, ephemeris, phase_string):
    """
    Get a bias, dark, flat corrected image from a :class:`DataSet` after weighting by the probability of each being real. This function assumes that the true number of photons expected per pixel is << 1.
    
    Parameters
    ----------
    data_set : DataSet
        The proto-Lightspeed data set

    Returns
    -------
    array-like
        The image, crrected for flat and quantum efficiency
    """
    on_range, off_range = get_range(phase_string)

    on_numer = np.zeros(data_set.image_shape)
    on_denom = np.zeros(data_set.image_shape)
    on_n_frames = np.zeros(data_set.image_shape, int)
    off_numer = np.zeros(data_set.image_shape)
    off_denom = np.zeros(data_set.image_shape)
    off_n_frames = np.zeros(data_set.image_shape, int)
    for frame in data_set:
        good_mask = ~np.isnan(frame.image)
        masked_image = frame.image[good_mask]
        p0 = data_set.pixel_properties.get_prob(masked_image, 0, mask=good_mask)
        p1 = data_set.pixel_properties.get_prob(masked_image, 1, mask=good_mask)
        odds = p1/p0
        phase = ephemeris.get_phase(frame.timestamp-frame.duration/2)
        if contains_phase(on_range, phase):
            on_numer[good_mask] += odds - 1
            on_denom[good_mask] += odds**2
            on_n_frames[good_mask] += 1
        if contains_phase(off_range, phase):
            off_numer[good_mask] += odds - 1
            off_denom[good_mask] += odds**2
            off_n_frames[good_mask] += 1

    on_image = on_numer/on_denom
    on_image *= on_n_frames
    off_image = off_numer/off_denom
    off_image *= off_n_frames

    on = Image(on_image, data_set, on_n_frames)
    off = Image(off_image, data_set, off_n_frames)
    image = copy.deepcopy(on)
    image.photons_per_second -= off.photons_per_second

    return image