import numpy as np
import os
from scipy.optimize import least_squares
from astropy.io import fits
from ..regions import CircleRegion
from ..constants import PIXEL_SIZE
from ..psf_utils import moffat_min_fn

def fit_moffat(args):
    if not os.path.exists(args.roi):
        raise Exception(f"The region file {args.roi} does not exist")
    try:
        reg = CircleRegion.load(args.roi)
    except:
        raise Exception("Please provide a circular region")
    x0, y0 = reg.x, reg.y
    radius = np.sqrt(reg.radius2)

    with fits.open(args.input) as hdul:
        image = hdul[0].data
    xmin = int(np.clip(x0-radius, 0, image.shape[1]))
    xmax = int(np.clip(x0+radius, 0, image.shape[1]))
    ymin = int(np.clip(y0-radius, 0, image.shape[0]))
    ymax = int(np.clip(y0+radius, 0, image.shape[0]))
    image = image[ymin:ymax, xmin:xmax]

    xs, ys = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    initial_params = (image.shape[1]/2, image.shape[0]/2, 5, 5, 0, 1, 1, 0,)
    result = least_squares(moffat_min_fn,
        x0=initial_params,
        bounds=[(0, 0, 0, 0, -np.inf, 0.2, 0, 0), (image.shape[1], image.shape[0], np.inf, np.inf, np.inf, 10, np.inf, np.inf)],
        args=(xs, ys, image),
    )

    px, py, gx, gy, theta, alpha, amp, bg = result.x
    ratio = amp/bg
    if gx < gy:
        gx, gy = gy, gx
        theta += np.pi/2
    theta *= 180 / np.pi # Convert angle to degrees
    theta = (theta + 360) % 180

    # Convert sigmas to fwhm
    major = gx * 2 * np.sqrt(2**(1/alpha) - 1) * PIXEL_SIZE
    minor = gy * 2 * np.sqrt(2**(1/alpha) - 1) * PIXEL_SIZE

    print(f"The PSF was {major:.2f}\" x {minor:.2f}\" @ {theta:.0f} deg, with an amplitude of {ratio:.1f}x background")

    return major, minor, theta