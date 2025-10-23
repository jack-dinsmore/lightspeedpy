import numpy as np
import os, tqdm
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from .pixel_properties import PixelProperties

DEFAULT_TIME = "2025-09-13 06:00:00.00429"
ADU_PER_ELECTRON = 8.9
DEFAULT_BIAS = 199.5

class DataSet:
    def __init__(self, input, cr_cut=True, auto_bias=True):
        directory = os.path.dirname(input)
        filename = os.path.basename(input)
        prefix = filename[:-8]

        self.filenames = []
        for f in os.listdir(directory):
            if not f.startswith(prefix): continue
            self.filenames.append(f"{directory}/{f}")
        if len(self.filenames) == 0:
            raise Exception(f"The filename {input} does not exist")
        self.filenames = np.sort(self.filenames)
        self.display_filenames()

        with fits.open(self.filenames[0]) as hdul:
            self.header0 = dict(hdul[0].header)
            self.header1 = dict(hdul[1].header)
            self.get_image_data(hdul)
            self.get_timing_data(hdul)

        self.bias = None
        self.bias_data_set = None
        self.dark = None
        self.flat = None
        self.pixel_properties=None
        self.cr_cut = cr_cut
        self.auto_bias = auto_bias

    def display_filenames(self):
        print("Loading files")
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

    def get_timing_data(self, hdul):
        try:
            self.start_time = Time(hdul[0].header["GPSSTART"], scale="utc")
            self.start_time += 10 * u.s # Kevin's offset
        except:
            print(f"No GPS time detected. Using {DEFAULT_TIME}")
            default_mjd = Time(DEFAULT_TIME, scale="utc").mjd
            self.start_time = Time(default_mjd, format="mjd", scale="utc")
        self.start_time = self.start_time.tt # Switch to TT
        self.time_per_frame = (hdul[2].data["TIMESTAMP"][1] - hdul[2].data["TIMESTAMP"][0]) / float(self.frames_per_bundle)

        psr_coord = SkyCoord(
            ra=hdul[1].header["TELRA"],
            dec=hdul[1].header["TELDEC"],
            unit=(u.hourangle, u.deg),
            frame="icrs"
        )
        loc = EarthLocation.of_site('LCO')
        start_ltt_bary = self.start_time.light_travel_time(psr_coord, location=loc)
        start_plus_one_ltt_bary = (self.start_time+1*u.s).light_travel_time(psr_coord, location=loc)
        ltt_derivative = (start_plus_one_ltt_bary - start_ltt_bary).to(u.s).value # Seconds per second
        self.start_time = self.start_time.mjd *(3600*24) + start_ltt_bary.to(u.s).value
        self.time_dilation_factor = 1 + ltt_derivative

    def get_pixel_properties(self):
        if self.pixel_properties is None:
            if self.bias_data_set is None:
                self.pixel_properties = PixelProperties(self)
            else:
                # Take the larger of the two data_sets
                if self.num_frames() > self.bias_data_set.num_frames():
                    print("Using the pixel properties of the data set")
                    self.pixel_properties = PixelProperties(self)
                else:
                    print("Using the pixel properties of the bias")
                    self.pixel_properties = PixelProperties(self.bias_data_set)
        return self.pixel_properties


    def set_bias(self, bias):
        if bias == "self":
            self.bias = self.get_pixel_properties().bias
        else:
            data_set = DataSet(bias, cr_cut=False, auto_bias=False)
            self.bias_data_set = data_set

            frame_total = np.zeros(data_set.image_shape)
            n_frames = 0
            for frame in data_set:
                frame_total += frame.image
                n_frames += 1
            self.bias = frame_total / n_frames

    def set_dark(self, dark):
        data_set = DataSet(dark)
        data_set.bias = self.bias
        frame_total = np.zeros(data_set.image_shape)
        n_frames = np.zeros(data_set.image_shape, int)
        for frame in data_set:
            good_mask = ~np.isnan(frame.image)
            frame_total[good_mask] += frame.image[good_mask]
            n_frames[good_mask] +=1
        self.dark = frame_total / n_frames

    def set_flat(self, flat):
        data_set = DataSet(flat)
        data_set.bias = self.bias
        data_set.dark = self.dark
        frame_total = np.zeros(data_set.image_shape)
        n_frames = np.zeros(data_set.image_shape, int)
        for frame in data_set:
            good_mask = ~np.isnan(frame.image)
            frame_total[good_mask] += frame.image[good_mask]
            n_frames[good_mask] +=1
        self.flat = frame_total / n_frames

    def __iter__(self):
        return DataSetIterator(self)

class DataSetIterator:
    def __init__(self, data_set):
        self.data_set = data_set
        self.open_file = None
        self.file_index = 0
        self.first_run = True

        self.renew_file()
        n_frames = data_set.frames_per_bundle*self.n_bundles*len(data_set.filenames)
        self.bar = tqdm.tqdm(total=n_frames)

    def renew_file(self):
        if self.open_file is not None:
            self.open_file.close()
        self.open_file = fits.open(self.data_set.filenames[self.file_index])
        self.bundle_index = 0
        self.frame_index = 0
        self.n_bundles = self.open_file[1].data.shape[0]

    def __next__(self):
        if not self.first_run:
            # Increment the counters
            self.frame_index += 1
            if self.frame_index >= self.data_set.frames_per_bundle:
                self.frame_index = 0
                self.bundle_index += 1
            if self.bundle_index >= self.n_bundles:
                self.bundle_index = 0
                self.file_index += 1
                if self.file_index >= len(self.data_set.filenames):
                    self.open_file.close()
                    self.bar.close()
                    raise StopIteration
                self.renew_file()

        # Get the frame
        start_pixel = self.data_set.image_shape[0] * self.frame_index
        raw_image = self.open_file[1].data[self.bundle_index, start_pixel:(start_pixel+self.data_set.image_shape[0]), :]
        timestamp = self.open_file[2].data["TIMESTAMP"][self.bundle_index]
        timestamp += self.data_set.time_per_frame * self.frame_index
        timestamp *= self.data_set.time_dilation_factor
        timestamp += self.data_set.start_time
        self.bar.update(1)
        self.first_run = False

        return Frame(raw_image, timestamp, self.data_set)
    
class Frame:
    def __init__(self, raw_image, timestamp, data_set):
        self.image = raw_image.astype(float) / ADU_PER_ELECTRON
        self.duration = data_set.time_per_frame
        self.timestamp = timestamp

        if data_set.bias is not None:
            self.image -= data_set.bias
        elif data_set.auto_bias:
            self.image -= DEFAULT_BIAS / ADU_PER_ELECTRON

        if data_set.dark is not None:
            self.image -= data_set.dark * data_set.time_per_frame

        if data_set.cr_cut:
            cosmic_ray_filter(self.image)

def cosmic_ray_filter(image):
    # Compute laplacian
    step=3 # Pixels
    img = np.pad(image, step, mode='constant')
    laplacian = (
        img[step:-step, 2*step:] +
        img[2*step:, step:-step] +
        img[:-2*step, step:-step] +
        img[step:-step, :-2*step] -
        4 * img[step:-step, step:-step]
    )/step**2

    # Mask CRs
    image[laplacian < -0.8] = np.nan