import numpy as np
import os
from scipy.optimize import minimize
from ..regions import CircleRegion
from astropy.io import fits
from ..constants import PIXEL_SIZE

def gaussian_chisq(params, xs, ys, image):
    positions = np.array([xs - params[0], ys - params[1]])
    matrix = np.array([[params[2], params[4]], [params[4], params[3]]])
    if np.linalg.det(matrix) < 0:
        return (1-np.linalg.det(matrix)) * 1e8
    model = np.exp(-np.einsum("iab,ij,jab->ab", positions, matrix, positions) / 2)
    model += params[5] # Background
    model *= params[6]

    return np.sum((model - image)**2)

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
    x0 -= xmin
    y0 -= ymin

    xs, ys = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing="ij")
    bg = np.nanpercentile(image, 50)
    mean = np.mean(image - bg)
    mxx = np.mean((xs - x0)**2 * (image - bg)) / mean
    myy = np.mean((ys - y0)**2 * (image - bg)) / mean
    mxy = np.mean((xs - x0)*(ys - y0) * (image - bg)) / mean
    det = mxx*myy - mxy**2
    initial_params = (x0, y0, myy/det, mxx/det, -mxy/det, bg / np.max(image), np.max(image))

    result = minimize(gaussian_chisq,
        x0=initial_params,
        bounds=[(0, image.shape[0]), (0, image.shape[1]), (0, None), (0, None), (None,None), (0, 1), (0, None)],
        args=(xs, ys, image),
        method="nelder-mead"
    )

    positions = np.array([xs - result.x[0], ys - result.x[1]])
    matrix = np.array([[result.x[2], result.x[4]], [result.x[4], result.x[3]]])
    model = np.exp(-np.einsum("iab,ij,jab->ab", positions, matrix, positions) / 2)
    model += result.x[5] # Background
    model *= result.x[6]

    matrix = np.array([[result.x[2], result.x[4]], [result.x[4], result.x[3]]])
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