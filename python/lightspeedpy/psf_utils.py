import numpy as np
from .regions import EllipseRegion, CircleRegion

def moffat_min_fn(params, xs, ys, image):
    # Params are x, y, gx, gy, theta, alpha, amp, bg
    model = make_psf_image(xs - params[0], ys - params[1], params[2], params[3], params[4], params[5])
    model /= np.max(model)
    model *= params[6]
    model += params[7] # Background
    return (model - image).reshape(-1)

def make_psf_from_region(data_set, region):
    """
    Returns an image which contains the PSF weights for each pixel given the PSF shape (half max contour) contained in reg_file.

    Parameters
    ----------
    data_set: DataSet
        Data set to generate a PSF image for
    region: Region
        Region describing the half max contour of the PSF. Must be either circular or elliptical.

    Returns
    -------
        array-like
    An image the size of the typical data set frame, with the PSF drawn. The image is normalized to sum to 1.
    """

    xs, ys = np.meshgrid(np.arange(data_set.image_shape[1]), np.arange(data_set.image_shape[0]))
    alpha = 1

    if type(region) is CircleRegion:
        fwhm_pixels = np.sqrt(region.radius2)
        gamma_x = fwhm_pixels / (2 * np.sqrt(2**(1/alpha) - 1))
        gamma_y = fwhm_pixels / (2 * np.sqrt(2**(1/alpha) - 1))
        theta = 0
    elif type(region) is EllipseRegion:
        gamma_x = region.a / (2 * np.sqrt(2**(1/alpha) - 1))
        gamma_y = region.b / (2 * np.sqrt(2**(1/alpha) - 1))
        theta = region.angle
    else:
        raise Exception("PSF weighting can only be performed with elliptical or circular regions")
    
    return make_psf_image(xs - region.x, ys - region.y, gamma_x, gamma_y, theta)

def make_psf_image(xs, ys, gamma_x, gamma_y, theta, alpha=1):
    """
    Returns an image which contains the PSF weights for each pixel given the PSF shape (half max contour) contained in reg_file.

    Parameters
    ----------
    xs: array-like
        Image x coordinates measured from PSF center (from np.meshgrid)
    ys: array-like
        Image y coordinates measured from PSF center (from np.meshgrid)
    gamma_x: float
        Moffat gamma parameter for the major axis
    gamma_x: float
        Moffat gamma parameter for the minor axis
    theta: float
        Major axis position angle
    alpha: float
        Moffat slope parameter.

    Returns
    -------
        array-like
    An image the size of the typical data set frame, with the PSF drawn. The image is normalized to sum to 1.
    """
    rot = np.array([[np.sin(theta), np.cos(theta)], [-np.cos(theta), np.sin(theta)]])
    inv_cov = rot @ np.diag([1/gamma_x**2, 1/gamma_y**2]) @ np.transpose(rot)
    vec = np.array([xs, ys])
    arg = np.einsum("iab,ij,jab->ab", vec, inv_cov, vec)
    image = 1 / (1 + arg)**alpha

    return image