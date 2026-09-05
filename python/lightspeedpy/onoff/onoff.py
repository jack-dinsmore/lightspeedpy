import numpy as np
import copy
from ..image.image import Image
from ..weight import Weighter, PixelLayout
from ..util import Matrix

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

class PhaseRange:
    def __init__(self, s):
        self.ranges = []
        self.reversed = []
        for r in s.split("&"):
            chunks = r.split(":")
            if len(chunks) != 2:
                raise Exception("Each range must be colon-separated")
            low, high = float(chunks[0]), float(chunks[1])
            self.ranges.append((low, high))
            if low < high:
                self.reversed.append(False)
            else:
                self.reversed.append(True)
                
    def contains(self, phase):
        for reverse, (low, high) in zip(self.reversed, self.ranges):
            if (not reverse) and (low < phase and phase < high):
                return True
            if reverse and (low < phase or phase < high):
                return True
        return False

def make_on_off(data_set, ephemeris, phase_string, mode, n_electrons=3, n_iterations=25):
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
    mode : str
        Either "sum", "clip", or "weight", specifying the mode of image generation

    Returns
    -------
    Image
        The image, crrected for flat and quantum efficiency
    """
    strings = phase_string.split(',')
    if len(strings) != 2:
        raise Exception("The phase string must have two comma-separated ranges")
    on_range = PhaseRange(strings[0])
    off_range = PhaseRange(strings[1])

    on_image = np.zeros(data_set.image_shape)
    on_n_frames = np.zeros(data_set.image_shape)
    off_image = np.zeros(data_set.image_shape)
    off_n_frames = np.zeros(data_set.image_shape)
    if mode == "weight":
        layout = PixelLayout.image(data_set)
        weight_matrix = Matrix.identity(layout.n_pixels)
        histogram_max_electrons = data_set.iter_kwargs["cr_ceil"] if "cr_ceil" in data_set.iter_kwargs else None
        on_weighter = Weighter(layout, n_electrons, weight_matrix, histogram_max_electrons=histogram_max_electrons)
        off_weighter = Weighter(layout, n_electrons, weight_matrix, histogram_max_electrons=histogram_max_electrons)

    for frame in data_set:
        phase = ephemeris.get_phase(frame.timestamp)
        if on_range.contains(phase):
            good_mask = ~np.isnan(frame.image)
            if mode == "sum":
                on_image[good_mask] += frame.image[good_mask]
            elif mode == "clip":
                on_image[good_mask] += np.round(frame.image[good_mask])
            elif mode == "weight":
                on_weighter.add_pixels(frame.image)
            else:
                raise Exception(f"Unrecognized mode {mode}")
            on_n_frames[good_mask] += 1
                
        if off_range.contains(phase):
            good_mask = ~np.isnan(frame.image)
            if mode == "sum":
                off_image[good_mask] += frame.image[good_mask]
            elif mode == "clip":
                off_image[good_mask] += np.round(frame.image[good_mask])
            elif mode == "weight":
                off_weighter.add_pixels(frame.image)
            else:
                raise Exception(f"Unrecognized mode {mode}")
            off_n_frames[good_mask] += 1

    if mode == "weight":
        # One needs to multiply by the number of frames since weighting calculates the mean value and the other methods compute the sum.
        on_image = on_weighter.get_fluxes(n_iterations).reshape(data_set.image_shape) * on_n_frames
        off_image = off_weighter.get_fluxes(n_iterations).reshape(data_set.image_shape) * off_n_frames

    on = Image(on_image, data_set, on_n_frames)
    off = Image(off_image, data_set, off_n_frames)
    image = copy.deepcopy(on)
    image.photons_per_second -= off.photons_per_second

    return image
