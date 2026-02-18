import numpy as np
import os, tqdm
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from .pixel_properties import PixelProperties
from .qe import get_qe

DEFAULT_TIME = "2025-09-13 06:00:00.00429"
ADU_PER_ELECTRON = 8.9
DEFAULT_BIAS = 199.5

class DataSet:
    def __init__(self, runs, frames=None):
        if len(runs) == 0:
            raise Exception("You must provide at least one filename")
        self.runs = runs
        self.image_shape = runs[0].image_shape
        self.header0 = runs[0].header0
        self.header1 = runs[0].header1
        self.flat = self.runs[0].flat
        self.frames = frames
        for run in runs:
            if run.image_shape != self.image_shape:
                raise Exception("You cannot form one data set out of runs with different subarrays")

    def from_files(filenames, cr_cut=True, auto_bias=True, timing_offset=0, frames=None):
        runs = [SingleRun(filename, cr_cut, auto_bias, timing_offset) for filename in filenames]
        return DataSet(runs, frames)

    def iterator(self, use_bar=True, max_frames=None):
        return DataSetIteratorRet(self, use_bar, max_frames)

    def display_filenames(self):
        for run in self.runs:
            run.display_filenames()

    def bootstrap(self):
        print("BOOTSTRAP")
        for run in self.runs:
            run.filenames = np.random.choice(run.filenames, len(run.filenames), replace=True)
        self.display_filenames()

    def __iter__(self):
        return self.iterator().__iter__()
    
    def __iadd__(self, data_set):
        # TODO check for compatibility between header0 and header1 (in particular, the subarray selection. I don't believe I need to require the exposure to be the same.)
        if self.image_shape != data_set.image_shape:
            raise Exception("You cannot form one data set out of runs with different subarrays")
        for r in data_set.runs:
            self.runs.append(r)

    def num_frames(self):
        n_frames = 0
        if self.frames is None:
            for run in self.runs:
                n_frames += run.num_frames()
        else:
            if len(self.frames) == 1:
                n_frames = 1
            else:
                n_frames = self.frames[1] - self.frames[0]
        return n_frames

    def get_timestamps(self):
        timestamps = []
        for run in self.runs:
            timestamps = np.concatenate([timestamps, run.get_timestamps()])
        return timestamps
    
    def set_bias(self, bias):
        for run in self.runs:
            run.set_bias(bias)

    def set_self_bias(self):
        for run in self.runs:
            run.set_self_bias()

    def set_dark(self, dark):
        for run in self.runs:
            run.set_dark(dark)

    def set_flat(self, flat):
        for run in self.runs:
            run.set_flat(flat)
        self.flat = self.runs[0].flat # TODO this is a stopgap to divide by flat
    
class DataSetIteratorRet:
    def __init__(self, *args):
        self.args = args

    def __iter__(self):
        return DataSetIterator(*self.args)


class SingleRun:
    def __init__(self, filename, cr_cut, auto_bias, timing_offset):
        """
        Create a new data set
        # Arguments
        - filename: name of the first cube in the data set
        - cr_cut: set to True to automatically remove cosmic rays
        - auto_bias: set to True to subtract 199.5 ADC counts from the image as a very rough bias when not using a bias frame. If you are using a bias frame, this flag has no effect.
        - timing offset: set to offset the TIMESTAMPs by a given number [seconds]
        """
        directory = os.path.dirname(filename)
        filename = os.path.basename(filename)
        prefix = filename[:-8]

        self.filenames = []
        for f in os.listdir(directory):
            if not f.startswith(prefix): continue
            self.filenames.append(f"{directory}/{f}")
        if len(self.filenames) == 0:
            raise Exception(f"The filename {filename} does not exist")
        self.filenames = np.sort(self.filenames)

        with fits.open(self.filenames[0]) as hdul:
            self.header0 = hdul[0].header
            self.header1 = hdul[1].header
            self.get_image_data(hdul)
            self.get_timing_data(hdul, timing_offset)

        self.bias = None
        self.bias_data_set = None
        self.dark = None
        self.flat = None
        self.pixel_properties=None
        self.cr_cut = cr_cut
        self.auto_bias = auto_bias

    def display_filenames(self):
        if len(self.filenames) <= 3:
            for filename in self.filenames:
                print(filename)
        else:
            print(self.filenames[0])
            print("...")
            print(self.filenames[-1])

    def get_image_data(self, hdul):
        if hdul[1].header["HIERARCH FRAMEBUNDLE MODE"] == "OFF":
            self.frames_per_bundle = 1
        else:
            self.frames_per_bundle = int(hdul[1].header["HIERARCH FRAMEBUNDLE NUMBER"])

        self.image_shape = (hdul[1].data.shape[1]//self.frames_per_bundle, hdul[1].data.shape[2])

    def num_frames(self):
        n_frames = 0
        for filename in self.filenames:
            with fits.open(filename) as hdul:
                n_frames += hdul[1].shape[0] * self.frames_per_bundle
        return n_frames

    def get_timing_data(self, hdul, timing_offset=0):
        # Get timing data for this fileset. Note: this function should only be called on the first cube. get_image_data has to be run first
        self.time_per_frame = hdul[1].header["HIERARCH EXPOSURE TIME"]
        try:
            self.start_time = Time(hdul[0].header["GPSSTART"], format="isot")
        except:
            self.start_time = Time(0, format="mjd")
        self.start_time = self.start_time.mjd * 3600 * 24 # start_time in units of seconds
        self.start_time -= hdul[2].data["TIMESTAMP"][0] # start_time is now time that the last frame in the first bundle was read out. This line doesn't do anything since December, since the first timestamp is already subtracted.
        self.start_time += hdul[1].header["HIERARCH TIMING READOUT TIME"] / 2# start_time is now time that the frame before the first frame was halfway through readout
        self.start_time += hdul[1].header["HIERARCH EXPOSURE TIME"] / 2# start_time is offset by half the exposure
        self.start_time -= timing_offset

    def get_timestamps(self):
        timestamps = []
        for filename in self.filenames:
            with fits.open(filename, memmap=True) as hdul:
                these_timestamps = []
                for timestamp in hdul[2].data["TIMESTAMP"]:
                    for frame_index in range(self.frames_per_bundle):
                        these_timestamps.append(np.float64(timestamp) + frame_index*self.time_per_frame + self.start_time)
            timestamps = np.concatenate([timestamps, these_timestamps])
        return timestamps

    def get_pixel_properties(self):
        if self.pixel_properties is None:
            if self.bias_data_set is None:
                self.pixel_properties = PixelProperties(DataSet([self]), False)
            else:
                my_vpos = int(self.header1["HIERARCH SUBARRAY VPOS"]) if self.header1["HIERARCH SUBARRAY MODE"] == "ON" else 0
                my_hpos = int(self.header1["HIERARCH SUBARRAY HPOS"]) if self.header1["HIERARCH SUBARRAY MODE"] == "ON" else 0
                bias_vpos = int(self.bias_data_set.header1["HIERARCH SUBARRAY VPOS"]) if self.bias_data_set.header1["HIERARCH SUBARRAY MODE"] == "ON" else 0
                bias_hpos = int(self.bias_data_set.header1["HIERARCH SUBARRAY HPOS"]) if self.bias_data_set.header1["HIERARCH SUBARRAY MODE"] == "ON" else 0
                start_x = my_vpos - bias_vpos
                start_y = my_hpos - bias_hpos
                print(start_x, start_y, self.image_shape)

                self.pixel_properties = PixelProperties(self.bias_data_set, True, window=(start_x, start_x+self.image_shape[0], start_y, start_y+self.image_shape[1]))

                self.pixel_properties.bias *= 0 # Turn off the bias when using the bias data set to determine pixel properties, as the bias has already been subtracted.

        return self.pixel_properties

    def set_bias(self, bias):
        if self.pixel_properties is not None:
            raise Exception("You set a bias frame after calling a function that calculates the pixel properties (e.g. set_self_bias, get_pixel_properties, etc.). You cannot do this because get_pixel_properties needs a good bias to function.")
        
        bias_set = DataSet.from_files(bias, cr_cut=False, auto_bias=False)
        self.bias_data_set = bias_set

        frame_total = np.zeros(bias_set.image_shape)
        n_frames = 0
        for frame in bias_set.iterator(max_frames=10_000):
            frame_total += frame.image
            n_frames += 1
        bias_image = frame_total / n_frames

        # Trim down the bias image to be the correct size
        my_vpos = int(self.header1["HIERARCH SUBARRAY VPOS"]) if self.header1["HIERARCH SUBARRAY MODE"] == "ON" else 0
        my_hpos = int(self.header1["HIERARCH SUBARRAY HPOS"]) if self.header1["HIERARCH SUBARRAY MODE"] == "ON" else 0
        bias_vpos = int(bias_set.header1["HIERARCH SUBARRAY VPOS"]) if bias_set.header1["HIERARCH SUBARRAY MODE"] == "ON" else 0
        bias_hpos = int(bias_set.header1["HIERARCH SUBARRAY HPOS"]) if bias_set.header1["HIERARCH SUBARRAY MODE"] == "ON" else 0
        start_x = my_vpos - bias_vpos
        start_y = my_hpos - bias_hpos
        self.bias = bias_image[start_x:start_x+self.image_shape[0], start_y:start_y+self.image_shape[1]]

    def set_self_bias(self):
        if self.bias_data_set is None:
            self.bias = self.get_pixel_properties().bias + DEFAULT_BIAS / ADU_PER_ELECTRON
            self.pixel_properties = None
            self.get_pixel_properties()

        else:
            self.bias += self.get_pixel_properties().bias
            self.pixel_properties = None
            self.get_pixel_properties()

    def set_dark(self, dark):
        data_set = DataSet.from_files(dark)
        data_set.bias = self.bias
        frame_total = np.zeros(data_set.image_shape)
        n_frames = np.zeros(data_set.image_shape, int)
        for frame in data_set:
            good_mask = ~np.isnan(frame.image)
            frame_total[good_mask] += frame.image[good_mask]
            n_frames[good_mask] +=1
        self.dark = frame_total / n_frames # The dark is now electrons per frame

    def set_flat(self, flat):
        data_set = DataSet.from_files(flat)
        data_set.bias = self.bias
        data_set.dark = self.dark
        frame_total = np.zeros(data_set.image_shape)
        n_frames = np.zeros(data_set.image_shape, int)
        for frame in data_set:
            good_mask = ~np.isnan(frame.image)
            frame_total[good_mask] += frame.image[good_mask]
            n_frames[good_mask] +=1
        self.flat = frame_total / n_frames
        self.flat /= get_qe()(self.flat)
        self.flat /= np.nanmax(self.flat)

class DataSetIterator:
    def __init__(self, data_set, use_bar, max_frames):
        self.data_set = data_set
        self.open_file = None
        self.run_index = 0
        self.file_index = 0
        self.first_run = True
        self.total_frame_index = 0
        self.max_frames = max_frames

        self.renew_file()
        n_frames = data_set.num_frames()

        if use_bar:
            if max_frames is not None and max_frames < n_frames:
                self.bar = tqdm.tqdm(total=max_frames, colour="green")
            else:
                self.bar = tqdm.tqdm(total=n_frames)
        else:
            self.bar = None

    def renew_file(self):
        if self.open_file is not None:
            self.open_file.close()
        self.open_file = fits.open(self.data_set.runs[self.run_index].filenames[self.file_index])
        self.bundle_index = 0
        self.frame_index = 0
        self.n_bundles = self.open_file[1].data.shape[0]

    def stop(self):
        self.open_file.close()
        self.bar.close()
        raise StopIteration

    def __next__(self):
        self.total_frame_index += 1
        if self.max_frames is not None and self.total_frame_index > self.max_frames:
            self.stop()
        
        if not self.first_run:
            # Increment the counters
            self.frame_index += 1
            if self.frame_index >= self.data_set.runs[self.run_index].frames_per_bundle:
                self.frame_index = 0
                self.bundle_index += 1
            if self.bundle_index >= self.n_bundles:
                self.bundle_index = 0
                self.file_index += 1
                if self.file_index >= len(self.data_set.runs[self.run_index].filenames):
                    self.file_index = 0
                    self.run_index += 1
                    if self.run_index >= len(self.data_set.runs):
                        self.stop()
                self.renew_file()
        self.first_run = False

        if self.data_set.frames is not None:
            if len(self.data_set.frames) == 1:
                if self.total_frame_index-1 != self.data_set.frames[0]: self.__next__()
            else:
                if self.total_frame_index-1 < self.data_set.frames[0] or self.total_frame_index-1 >= self.data_set.frames[1]: self.__next__()

        # Get the frame
        start_pixel = self.data_set.runs[self.run_index].image_shape[0] * self.frame_index
        raw_image = self.open_file[1].data[self.bundle_index, start_pixel:(start_pixel+self.data_set.runs[self.run_index].image_shape[0]), :]
        timestamp = np.float64(self.open_file[2].data["TIMESTAMP"][self.bundle_index])
        timestamp += self.data_set.runs[self.run_index].time_per_frame * self.frame_index
        timestamp += self.data_set.runs[self.run_index].start_time

        if self.bar is not None:
            self.bar.update(1)

        return Frame(raw_image, timestamp, self.data_set.runs[self.run_index])
    
class Frame:
    """
    # Attributes
    * image contains the frame image (float) in units of electrons. If darks, biases, and cr cuts were set, they are already subtracted
    * timestamp contains the time in seconds after camera start
    """
    def __init__(self, raw_image, timestamp, data_set):
        self.image = raw_image.astype(float) / ADU_PER_ELECTRON
        self.duration = data_set.time_per_frame
        self.timestamp = timestamp # Seconds


        if data_set.bias is not None:
            self.image -= data_set.bias
        elif data_set.auto_bias:
            self.image -= DEFAULT_BIAS / ADU_PER_ELECTRON

        if data_set.dark is not None:
            self.image -= data_set.dark

        if data_set.cr_cut:
            cosmic_ray_filter(self.image)

def cosmic_ray_filter(image):
    # Compute laplacian
    step=1 # Pixels
    img = np.pad(image, step, mode='constant')
    laplacian = (
        img[step:-step, 2*step:] +
        img[2*step:, step:-step] +
        img[:-2*step, step:-step] +
        img[step:-step, :-2*step] -
        4 * img[step:-step, step:-step]
    )/step**2
    cr_sensor = -laplacian/image

    # Mask CRs TODO is this working well for pulsars? I made it for LH
    image[(cr_sensor > 2) & (image > 3)] = np.nan