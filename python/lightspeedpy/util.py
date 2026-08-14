import numpy as np
import os
TMP_LOCATION = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "tmp"))
MAX_ENORMOUS_ARRAY_SIZE = 100e6 # 100 megs

def trim_image(image, source_data_set, dest_data_set):
    my_vpos = int(dest_data_set.header1["HIERARCH SUBARRAY VPOS"]) if dest_data_set.header1["HIERARCH SUBARRAY MODE"] == "ON" else 0
    my_hpos = int(dest_data_set.header1["HIERARCH SUBARRAY HPOS"]) if dest_data_set.header1["HIERARCH SUBARRAY MODE"] == "ON" else 0
    bias_vpos = int(source_data_set.header1["HIERARCH SUBARRAY VPOS"]) if source_data_set.header1["HIERARCH SUBARRAY MODE"] == "ON" else 0
    bias_hpos = int(source_data_set.header1["HIERARCH SUBARRAY HPOS"]) if source_data_set.header1["HIERARCH SUBARRAY MODE"] == "ON" else 0
    start_x = my_vpos - bias_vpos
    start_y = my_hpos - bias_hpos
    return image[start_x:start_x + dest_data_set.image_shape[0], start_y:start_y + dest_data_set.image_shape[1]]

def from_hms(s):
    # Convert the passed string from hms to degrees
    h, m, s = s.split(':')
    return (float(h) + float(m) / 60 + float(s)/3600) * 360 / 24

def to_hms(d, arcsec_precision=None):
    # Convert the passed string from dms to degrees
    x = d * 24 / 360
    h = int(x)
    x = (x- int(x)) * 60
    m = int(x)
    x = (x- int(x)) * 60
    s = x
    if arcsec_precision is not None:
        return f"{h:02d}:{m:02d}:{s:.{arcsec_precision}f}"
    else:
        return f"{h:02d}:{m:02d}:{s}"

def from_dms(s):
    # Convert the passed string to hms from degrees
    d, m, s = s.split(':')
    x = np.abs(float(d)) + float(m) / 60 + float(s)/3600
    return np.sign(float(d)) * x

def to_dms(d, arcsec_precision=None):
    # Convert the passed string to dms from degrees
    sign = "-" if d < 0 else ""
    x = np.abs(d)
    d = int(x)
    x = (x- int(x)) * 60
    m = int(x)
    x = (x- int(x)) * 60
    s = x
    if arcsec_precision is not None:
        return f"{sign}{d:02d}:{m:02d}:{s:.{arcsec_precision}f}"
    else:
        return f"{sign}{d:02d}:{m:02d}:{s}"

class Matrix:
    __array_ufunc__ = None
    """
    A matrix class that supports large, diagonal matrices

    Parameters
    ----------
    m : array-like
        A matrix to be stored
    is_diagonal : bool
        If true, m is assumed to be a vector which is the diagonal of the matrix. If false, m is a non-diagonal matrix.

    Notes 
    -----
    Use of the raw constructor is not recommended. Use from_matrix(m) or from_diagonal(d) instead.
    """
    def __init__(self, m, is_diagonal):
        self.m = m
        self.is_diagonal = is_diagonal
        if is_diagonal:
            self.shape = (len(m), len(m))
        else:
            self.shape = m.shape

    def from_matrix(m):
        """
        Create a matrix from a 2d array
        
        Parameters
        ----------
        m : array-like
            The matrix entries
        """
        if m.ndim == 1:
            m = m.reshape(-1, 1)
        if (m.shape[0] == m.shape[1]) and (np.count_nonzero(m) == np.count_nonzero(np.diagonal(m))):
            is_diagonal = True
            m = np.diagonal(m)
        else:
            is_diagonal = False
            m = m
        return Matrix(m, is_diagonal)

    def from_diagonal(d):
        """
        Create a diagonal matrix from a 1d array of diagonal entries
        
        Parameters
        ----------
        d : array-like
            The diagonal entries
        """
        return Matrix(d, True)

    def identity(ndim):
        """
        Create an identity matrix
        
        Parameters
        ----------
        ndim : int
            The number of diagonal entries
        """
        return Matrix.from_diagonal(np.ones(ndim))

    def __matmul__(self, other):
        if self.is_diagonal:
            return np.transpose(self.m * np.transpose(other))
        else:
            return self.m @ other
        
    def __rmatmul__(self, other):
        if self.is_diagonal:
            return other * self.m
        else:
            return other @ self.m

    def inv(self):
        """
        Compute the matrix inverse
        """
        if self.is_diagonal:
            return Matrix.from_diagonal(1/self.m)
        else:
            return Matrix.from_matrix(np.linalg.inv(self.m))
        
    def pinv(self):
        """
        Compute the matrix Moore-Penrose inverse
        """
        if self.is_diagonal:
            return Matrix.from_diagonal(1/self.m)
        else:
            return Matrix.from_matrix(np.linalg.pinv(self.m))

    def hess_product(self, v):
        """
        Compute the operation encoded by np.einsum("ai,aj,a->ij", self, self, v)

        Parameters
        ----------
        v : array-like
            The array to substitute for v. Should be 1 dimensional.
        """
        if self.is_diagonal:
            return Matrix.from_diagonal(self.m**2 * v)
        else:
            return Matrix.from_matrix(np.einsum("ai,aj,a->ij", self.m, self.m, v))

