import numpy as np
from scipy.interpolate import LinearNDInterpolator
import os

GRID_LOCATION = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "moments.npy"))

def make_m1m2_grid():
    if not os.path.exists(GRID_LOCATION):
        sigmas = np.linspace(0, 1, 100)[1:]
        means = np.linspace(-0.5, 0.5, 100)
        line = np.linspace(-3, 3, 100)
        xs = line - np.floor(line + 0.5)
        x2s = xs**2
        moments = np.zeros((2, len(sigmas), len(means)))
        for i, sigma in enumerate(sigmas):
            for j, mean in enumerate(means):
                gauss = np.exp(-(line - mean)**2 / (2*sigma**2))
                gauss /= np.sum(gauss)
                moments[0,i,j] = np.sum(gauss*xs)
                moments[1,i,j] = np.sum(gauss*x2s)

        sigma_grid, mean_grid = np.meshgrid(sigmas, means, indexing="ij")
        sigma_grid = sigma_grid.reshape(-1)
        mean_grid = mean_grid.reshape(-1)
        moments = moments.reshape(2, -1)
        np.save(GRID_LOCATION, np.concatenate([moments, [sigma_grid], [mean_grid]]))
    m1, m2, sigmas, means = np.load(GRID_LOCATION)

    return LinearNDInterpolator((m1,m2), np.transpose([means, sigmas]))

GRID_INTERPOLATOR = make_m1m2_grid()

class PixelProperties:
    def __init__(self, data_set):
        m1_image = np.zeros(data_set.image_shape)
        m2_image = np.zeros(data_set.image_shape)
        n_frames = np.zeros(data_set.image_shape)

        n_total_frames = 0
        
        for frame in data_set:
            clipped_image = frame.image - np.floor(frame.image + 0.5)
            good_mask = ~np.isnan(frame.image)
            m1_image[good_mask] += clipped_image[good_mask]
            m2_image[good_mask] += clipped_image[good_mask]**2
            n_frames[good_mask] += 1
            n_total_frames += 1
            if n_total_frames >= 10_000: break # The mean and bias are pretty well measured at this point

        m1_image /= n_frames
        m2_image /= n_frames

        output = GRID_INTERPOLATOR((m1_image, m2_image))
        self.bias = output[:,:,0]
        self.widths = output[:,:,1]