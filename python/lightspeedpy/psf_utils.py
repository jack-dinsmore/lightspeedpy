import numpy as np
from .regions import EllipseRegion, CircleRegion

def make_psf_image(data_set, region, extend_region=False):
    """
    Returns an image which contains the PSF weights for each pixel given the PSF shape contained in reg_file.

    Parameters
    ----------
    data_set: DataSet
        Data set to be processed. This is used to get the shape of the full frame
    region: Region
        Region describing the half max contour of the PSF. Must be either circular or elliptical.
    extend_region: bool
        Set to True to triple the size of the source region, so that it can now be used as an ROI. Default: False.

    Returns
    -------
        array-like
    An image the size of the typical data set frame, with the PSF drawn. The image is normalized to sum to 1.

    """
    xs, ys = np.meshgrid(np.arange(data_set.image_shape[1]), np.arange(data_set.image_shape[0]))
    if type(region) is CircleRegion:
        fwhm_pixels = np.sqrt(region.radius2)
        sigma_x = fwhm_pixels / 1.178
        sigma_y = fwhm_pixels / 1.178
        theta = 0
    elif type(region) is EllipseRegion:
        sigma_x = region.a / 1.178
        sigma_y = region.b / 1.178
        theta = region.angle
    else:
        raise Exception("PSF weighting can only be performed with elliptical or circular regions")

    rot = np.array([[np.sin(theta), np.cos(theta)], [-np.cos(theta), np.sin(theta)]])
    inv_cov = rot @ np.diag([1/sigma_x**2, 1/sigma_y**2]) @ np.transpose(rot)
    vec = np.array([xs - region.x, ys - region.y])
    gauss_exp = np.einsum("iab,ij,jab->ab", vec, inv_cov, vec)
    image = np.exp(-gauss_exp / 2)
    image /= np.sum(image)

    if extend_region:
        expansion_ratio = 5
        if type(region) is CircleRegion:
            region.radius2 *= expansion_ratio**2
        elif type(region) is EllipseRegion:
            region.a *= expansion_ratio
            region.b *= expansion_ratio
        else:
            raise Exception("PSF weighting can only be performed with elliptical or circular regions")
        
    return image