import numpy as np
import tqdm
from astropy.io import fits
from scipy.ndimage import median_filter
from .constants import ADU_PER_ELECTRON

class DataSetIteratorRet:
    def __init__(self, data_set, **kwargs):
        self.data_set = data_set
        self.kwargs = kwargs

    def __iter__(self):
        return DataSetIterator(self.data_set, **self.kwargs)

class DataSetIterator:
    def __init__(self, data_set, cut_cr=True, use_bar=True, max_frames=None, cr_thresh=20):
        self.data_set = data_set
        self.cut_cr = cut_cr
        self.max_frames = max_frames

        self.open_file = None
        self.file_index = 0
        self.first_run = True
        self.total_frame_index = 0
        self.cr_thresh = cr_thresh

        self._renew_file()
        n_frames = data_set.num_frames()

        if max_frames is not None:
            use_bar = use_bar and max_frames > 1 # Don't use the bar if there's > 1 frame

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
        self.open_file = fits.open(self.data_set.filenames[self.file_index])
        self.bundle_index = 0
        self.frame_index = 0
        self.n_bundles = self.open_file[1].data.shape[0]

    def _stop(self):
        self.open_file.close()
        if self.bar is not None:
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
        start_pixel = self.data_set.image_shape[0] * self.frame_index
        image = self.open_file[1].data[self.bundle_index, start_pixel:(start_pixel+self.data_set.image_shape[0]), :]
        is_saturated = np.max(image) >= 65_535
        image = image.astype(float)
        image -= 199.5
        image /= ADU_PER_ELECTRON
        image -= self.data_set.pixel_properties.bias
        if self.data_set.dark is not None:
            image -= self.data_set.dark * self.data_set.seconds_per_frame
        if self.cut_cr:
            cosmic_ray_filter(image, self.cr_thresh)

        # Don't flat or QE correct; that should be post-processing
        timestamp = np.float64(self.open_file[2].data["TIMESTAMP"][self.bundle_index])
        timestamp += self.data_set.seconds_per_frame * self.frame_index
        timestamp += self.data_set.start_time

        if self.bar is not None:
            self.bar.update(1)

        return Frame(image, timestamp, self.data_set.seconds_per_frame, is_saturated)
    
class Frame:
    """
    Attributes
    ----------
    image : array-like
        Contains the frame image in units of electrons. If darks and biases were set, they are already subtracted. If a dark was not set, it is assumed to be zero. If the bias is not set, it is assumed to be 199.5 ADU. The flat is not included, nor the QE.
    timestamp : float
        Time in seconds after camera start
    duration : float
        Frame time in seconds
    """
    def __init__(self, image, timestamp, duration, is_saturated):
        self.image = image
        self.duration = duration
        self.timestamp = timestamp # Seconds
        self.is_saturated = is_saturated

def cosmic_ray_filter(image, cr_thresh):
    # Compute laplacian
    rolling_median = median_filter(image, size=3)
    cr_sensor = (image - rolling_median) / np.sqrt(np.maximum(rolling_median, 0) + 1)
    cr_sensor = np.abs(cr_sensor)
    cr_sensor[image < -2] = cr_thresh + 1 # Remove very negative flux pixels

    # Mask CRs TODO is this working well for pulsars? I made it for LH
    cr_mask = cr_sensor > cr_thresh
    if np.mean(cr_mask) > 0.03:
        print(f"Warning: Masking more than 3% of the image as CRs ({np.mean(cr_mask)*100:.2f}%)")
    image[cr_mask] = np.nan
