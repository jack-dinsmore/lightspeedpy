import numpy as np
import os
from scipy.optimize import least_squares
from ..regions import CircleRegion
from astropy.io import fits
from ..constants import PIXEL_SIZE

def gaussian_min_fn(params, xs, ys, image):
    positions = np.array([xs - params[0], ys - params[1]])
    matrix = np.array([[params[2], params[4]], [params[4], params[3]]])
    model = np.exp(-np.einsum("iab,ij,jab->ab", positions, matrix, positions) / 2)
    model /= np.mean(model)
    model *= params[5]
    model += params[6] # Background

    return (model - image).reshape(-1)

def fit_gaussian(args):
    if not os.path.exists(args.roi):
        raise Exception(f"The region file {args.roi} does not exist")
    try:
        reg = CircleRegion.load(args.roi)
    except:
        raise Exception("Please provide a circular region")
    x0, y0 = reg.x, reg.y
    radius = np.sqrt(reg.radius2)

    with fits.open(args.input) as hdul:
        image = np.transpose(hdul[0].data)
    xmin = int(np.clip(x0-radius, 0, image.shape[0]))
    xmax = int(np.clip(x0+radius, 0, image.shape[0]))
    ymin = int(np.clip(y0-radius, 0, image.shape[1]))
    ymax = int(np.clip(y0+radius, 0, image.shape[1]))
    image = image[xmin:xmax, ymin:ymax]


    xs, ys = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing="ij")
    initial_params = (image.shape[0]/2, image.shape[1]/2, 1/5**2, 1/5**2, 0, 1, 0)

    max_diags = 1/3**2
    result = least_squares(gaussian_min_fn,
        x0=initial_params,
        bounds=[(0, 0, 0, 0, -0.9, 0, 0), (image.shape[0], image.shape[1], max_diags, max_diags, 0.9, np.inf, np.inf)],
        args=(xs, ys, image),
    )

    positions = np.array([xs - result.x[0], ys - result.x[1]])
    matrix = np.array([[result.x[2], result.x[4]], [result.x[4], result.x[3]]])
    
    # model = np.exp(-np.einsum("iab,ij,jab->ab", positions, matrix, positions) / 2)
    # model /= np.mean(model)
    # model *= result.x[5]
    # model += result.x[6] # Background
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(ncols=2)
    # axs[0].imshow(image, vmin=np.nanmin(image), vmax=np.nanmax(image))
    # axs[1].imshow(model, vmin=np.nanmin(image), vmax=np.nanmax(image))
    # fig.savefig("psf.png")

    cov = np.linalg.inv(matrix)
    evals, evecs = np.linalg.eigh(cov)
    major, minor = np.sqrt(evals)
    if major >= minor:
        theta = np.arctan2(evecs[0][0], evecs[0][1])
    else:
        theta = np.arctan2(evecs[1][0], evecs[1][1])
        major, minor = minor, major

    theta *= 180 / np.pi # Convert angle to degrees
    theta = (-theta + 360) % 180

    # Convert sigmas to fwhm
    major *= 2.355 * PIXEL_SIZE
    minor *= 2.355 * PIXEL_SIZE

    print(f"The PSF was {major:.2f}\" x {minor:.2f}\" @ {theta:.0f} deg")

    return major, minor, theta