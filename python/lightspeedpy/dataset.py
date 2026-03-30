import numpy as np
import os, copy
from astropy.io import fits
from astropy.time import Time
from .frame import DataSetIteratorRet
from .pixel_properties import PixelProperties
from .qe import get_qe

DEFAULT_TIME = "2025-09-13 06:00:00.00429"

def is_header_equal(h1, h2):
    """
    This function tests whether the headers h1 and h2 correspond to the same observation. Some header keywords change between cubes, and these are not tested."""
    for key in h1:
        if key in ["CUBEIDX", "TELRA", "TELDEC", "TELROT", "TELPA", "ROTENC", "TELEL", "TELAZ", "TELHA", "TELST", "TELUT", "AIRMASS", "NAXIS3"]: continue
        if h1[key] != h2[key]:
            return False
    return True

class DataSet:
    """
    Contains metadata relating to one contiguous LightSpeed observation. The data set is never loaded into memory, so many-GB observations can be treated.

    There are three ways to initialize a :class:`DataSet`:
     
    1. Use :meth:`DataSet.from_first` to create a data set of all files in the capture
    2. Use :meth:`DataSet.from_dir` to create a data set out of all the files in a directory
    3. Use the constructor and pass all the files you want to be in the data set as a parameter.

    Parameters
    ----------
    filenames : list of str
        List of files to include
    cut_cr : bool, optional
        Set to False to no longer cut CRs.
    """
    def __init__(self, filenames, **kwargs):
        if len(filenames) == 0:
            raise Exception("You must provide at least one filename")
        
        # Check the headers to make sure they are all the same
        self.filenames = np.sort(filenames)
        self.frames = []
        with fits.open(self.filenames[0]) as hdul:
            self.header0 = hdul[0].header
            self.header1 = hdul[1].header
            self._get_image_data(hdul)
            self._get_timing_data(hdul)
            self.frames.append(hdul[1].shape[0] * self.frames_per_bundle)

        for filename in self.filenames[1:]:
            with fits.open(filename) as hdul:
                self.frames.append(hdul[1].shape[0] * self.frames_per_bundle)
                if not is_header_equal(self.header0, hdul[0].header) or not is_header_equal(self.header1, hdul[1].header):
                    raise Exception(f"File {filename} had a different header. Lightspeedpy does not currently support this.")
        self.frames = np.array(self.frames)
                
        self.dark = None
        self.flat = None
        self.noises = None
        self.iter_kwargs = kwargs
        self.auto_bias = False

    def from_first(filename, min_index=None, max_index=None, **kwargs):
        """
        Create a :class:`DataSet` for all files in a capture
        
        Parameters
        ----------
        filename : str
            name of a file in the capture
        min_index : int, optional
            Minimum cube index to use. Default: the lowest that was downloaded
        max_index : int, optional
            Maximum cube index to use. Default: the highest that was downloaded
        
        Parameters
        -----------------
        See :class:`DataSet` for keyword parameters
        """
        directory = os.path.dirname(filename)
        filename = os.path.basename(filename)
        prefix = filename[:-8]

        filenames = []
        indices = []
        for f in os.listdir(directory):
            if not f.startswith(prefix): continue
            index = int(f[len(prefix):-5])
            if min_index is not None and index < min_index: continue
            if max_index is not None and index > max_index: continue
            filenames.append(f"{directory}/{f}")
            indices.append(index)
        if len(filenames) == 0:
            raise Exception(f"No filenames with these criteria exist")
        
        filenames = np.array(filenames)[np.argsort(indices)]
        return DataSet(filenames, **kwargs)
    
    def from_dir(directory, **kwargs):
        """
        Create a :class:`DataSet` for all files in a directory
        
        Parameters
        ----------
        directory : str
            name of the directory
        
        Keyword Arguments
        -----------------
        See :class:`DataSet`
        """
        if not os.path.exists(directory):
            raise Exception(f"The directory {directory} does not exist")
        
        filenames = []
        for f in os.listdir(directory):
            if not f.endswith(".fits"): continue
            filenames.append(f"{directory}/{f}")

        if len(filenames) == 0:
            raise Exception(f"The directory contained no FITS files")
        
        return DataSet(filenames, **kwargs)
    
    def iterator(self, **kwargs):
        """
        Create an iterator for iterating through all frames in a :class:`DataSet`. 

        Keyword Arguments
        -----------------
        The default keyword arguments are the ones you passed when you created the :class:`DataSet`.       
        You can override the defaults by passing new arguments here. You can also pass the following

        max_frames : int, optional
            Maximum number of frames to iterate through. Default: all the frames
        use_bar : bool, optional
            Show a progress bar. Default: True.

        Notes
        -----
        ``for frame in data_set.iterator()`` is equivalent to ``for frame in data_set``.
        """
        use_kwargs = {} if self.iter_kwargs is None else copy.copy(self.iter_kwargs)
        for k, v in kwargs.items():
            use_kwargs[k] = v
        return DataSetIteratorRet(self, **use_kwargs)

    def __iter__(self):
        return self.iterator().__iter__()

    def display_filenames(self):
        """
        Print all the filenames in the data set
        """
        indices = [int(f[-8:-5]) for f in self.filenames]
        sorted_frames = np.array(self.frames)[np.argsort(indices)]
        sorted_filenames = np.array(self.filenames)[np.argsort(indices)]
        indices = np.array(indices)[np.argsort(indices)]

        # Print all contiguous units
        breaks = [0]
        for i in range(1, len(indices)):
            if indices[i] != indices[i-1]+1:
                breaks.append(i)
        breaks.append(len(indices))

        for i in range(len(breaks)-1):
            start = breaks[i]
            stop = breaks[i+1]-1
            if stop - start > 2:
                print(sorted_filenames[start], f"({sorted_frames[start]}) frames")
                print("...")
                print(sorted_filenames[stop], f"({sorted_frames[stop]}) frames")
            else:
                for j in range(start, stop+1):
                    print(sorted_filenames[j], f"({sorted_frames[j]}) frames")

    def bootstrap(self, seed):
        """
        Resamples the file names for use in bootstrapping. If you want to estimate uncertainties, redo your analysis with many :meth:`DataSet.bootstrap`'ed data sets and calculate the standard deviation of your results.

        Notes
        -----
        You should never bootstrap a data set twice. Instead, you should always create a copy of the original data set with copy.deepcopy and then bootstrap the copy.
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(np.arange(len(self.filenames)), len(self.filenames), replace=True)
        self.filenames = self.filenames[indices]
        self.frames = self.frames[indices]

    def num_frames(self):
        """
        Counts the number of frames in the data set

        Returns
        -------
        int
            The number of frames in the data set
        """
        return np.sum(self.frames)
    
    def _get_image_data(self, hdul):
        if hdul[1].header["HIERARCH FRAMEBUNDLE MODE"] == "OFF":
            self.frames_per_bundle = 1
        else:
            self.frames_per_bundle = int(hdul[1].header["HIERARCH FRAMEBUNDLE NUMBER"])

        self.image_shape = (hdul[1].data.shape[1]//self.frames_per_bundle, hdul[1].data.shape[2])

    def _get_timing_data(self, hdul):
        # Get timing data for this fileset. Note: this function should only be called on the first cube. _get_image_data has to be run first
        self.seconds_per_frame = hdul[1].header["HIERARCH EXPOSURE TIME"]
        try:
            self.start_time = Time(hdul[0].header["GPSSTART"], format="isot")
        except:
            self.start_time = Time(0, format="mjd")
        self.start_time = self.start_time.mjd * 3600 * 24 # start_time in units of seconds
        self.start_time -= hdul[2].data["TIMESTAMP"][0] # start_time is now time that the last frame in the first bundle was read out. This line doesn't do anything since December, since the first timestamp is already subtracted.
        self.start_time += hdul[1].header["HIERARCH TIMING READOUT TIME"] / 2# start_time is now time that the frame before the first frame was halfway through readout
        self.start_time += hdul[1].header["HIERARCH EXPOSURE TIME"] / 2# start_time is offset by half the exposure

    def apply_timing_offset(self, timing_offset=0):
        """
        Apply a timing offset to the data set. Now, a frame read at `timing_offset` will be treated as having time 0.

        Parameters
        ----------
        timing_offset : float
            Time offset, in seconds
        """
        self.start_time -= timing_offset

    def stack(self, **kwargs):
        """
        Obtain the average image of the data cube.

        Returns
        -------
        array-like
            The average image of the cube, in electrons.

        Notes
        ------
        Same keyword arguments as :class:`DataSet`. For keyword arguments you do not provide, the values you passed to the data set constructor will be used.
        """
        frame_total = np.zeros(self.image_shape)
        n_frames = np.zeros(self.image_shape, int)
        for frame in self.iterator(**kwargs):
            goodmask = np.isfinite(frame.image)
            frame_total[goodmask] += frame.image[goodmask]
            n_frames[goodmask] += 1
        return frame_total / n_frames

    def _get_timestamps(self):
        timestamps = []
        for filename in self.filenames:
            with fits.open(filename) as hdul:
                these_timestamps = []
                for timestamp in hdul[2].data["TIMESTAMP"]:
                    for frame_index in range(self.frames_per_bundle):
                        these_timestamps.append(np.float64(timestamp) + frame_index*self.seconds_per_frame + self.start_time)
            timestamps = np.concatenate([timestamps, these_timestamps])
        return timestamps
    
    @property
    def pixel_properties(self):
        """
        PixelProperties
            Pixel properties of the data set
        """
        if not hasattr(self, "_pixel_properties"):
            self._pixel_properties = PixelProperties.default(self)
        return self._pixel_properties
    
    def set_bias(self, bias):
        """
        Set the bias

        Parameters
        ----------
        bias : DataSet or PixelProperties
            A :class:`DataSet` containing the bias observation, or the :class:`PixelProperties` of the observation
        """
        if type(bias) is DataSet:
            if hasattr(self, "_pixel_properties"):
                raise Exception("You set a bias frame after calling a function that calculates the pixel properties (e.g. set_self_bias, get_pixel_properties, etc.). You should do this in the reverse order since get_pixel_properties needs a good bias to function.")
            if not hasattr(bias, "_pixel_properties"):
                bias._pixel_properties = PixelProperties.from_bias(bias, self)
            self._pixel_properties = copy.deepcopy(bias.pixel_properties)
        else:
            self._pixel_properties = bias

    def set_self_bias(self):
        """
        Estimate an observation from these data. This can only be done accurately for very fast readout of a faint source, where most of the pixels detect zero photons.
        """
        self._pixel_properties = PixelProperties.from_data(self, self)

    def set_dark(self, dark_data_set):
        """
        Set the dark

        Parameters
        ----------
        dark_data_set : DataSet
            A :class:`DataSet` containing the dark observation
        """
        self.dark = dark_data_set.stack() / dark_data_set.seconds_per_frame

    def set_flat(self, flat_data_set):
        """
        Set the flat

        Parameters
        ----------
        flat_data_set : DataSet
            A :class:`DataSet` containing the flat observation
        """
        self.flat = flat_data_set.stack()
        self.flat /= get_qe()(self.flat)
        self.flat /= np.nanmax(self.flat)