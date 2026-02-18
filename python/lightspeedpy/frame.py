import numpy as np
import tqdm
from astropy.io import fits
from .constants import ADU_PER_ELECTRON

class DataSetIteratorRet:
    def __init__(self, data_set, **kwargs):
        self.data_set
        self.kwargs = kwargs

    def __iter__(self):
        return DataSetIterator(self.data_set, **self.kwargs)

class DataSetIterator:
    def __init__(self, data_set, cut_cr=True, use_bar=True, max_frames=None):
        self.data_set = data_set
        self.cut_cr = cut_cr
        self.max_frames = max_frames

        self.open_file = None
        self.file_index = 0
        self.first_run = True
        self.total_frame_index = 0

        self._renew_file()
        n_frames = data_set.num_frames()

        if use_bar:
            if max_frames is not None and max_frames < n_frames:
                self.bar = tqdm.tqdm(total=max_frames, colour="green")
            else:
                self.bar = tqdm.tqdm(total=n_frames)
        else:
            self.bar = None

    def _renew_file(self):
        if self.open_file is not None:
            self.open_file.close()
        self.open_file = fits.open(self.data_set.runs[self.run_index].filenames[self.file_index])
        self.bundle_index = 0
        self.frame_index = 0
        self.n_bundles = self.open_file[1].data.shape[0]

    def _stop(self):
        self.open_file.close()
        self.bar.close()
        raise StopIteration

    def __next__(self):
        self.total_frame_index += 1
        if self.max_frames is not None and self.total_frame_index > self.max_frames:
            self._stop()
        
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
                    self._stop()
                self._renew_file()
        self.first_run = False

        # Get the frame
        start_pixel = self.data_set.runs[self.run_index].image_shape[0] * self.frame_index
        image = self.open_file[1].data[self.bundle_index, start_pixel:(start_pixel+self.data_set.runs[self.run_index].image_shape[0]), :]
        if self.cut_cr:
            cosmic_ray_filter(image)
        image -= self.data_set.pixel_properties.bias()
        self.image -= self.data_set.dark * self.data_set.seconds_per_frame
        # Don't flat or QE correct; that should be post-processing

        timestamp = np.float64(self.open_file[2].data["TIMESTAMP"][self.bundle_index])
        timestamp += self.data_set.runs[self.run_index].seconds_per_frame * self.frame_index
        timestamp += self.data_set.runs[self.run_index].start_time

        if self.bar is not None:
            self.bar.update(1)

        return Frame(image, timestamp, self.data_set.seconds_per_frame)
    
class Frame:
    """
    # Attributes
    * image contains the frame image (float) in units of electrons. If darks, biases, and cr cuts were set, they are already subtracted
    * timestamp contains the time in seconds after camera start
    """
    def __init__(self, raw_image, timestamp, duration):
        self.image = raw_image.astype(float) / ADU_PER_ELECTRON
        self.duration = duration
        self.timestamp = timestamp # Seconds

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