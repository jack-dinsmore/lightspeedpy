import numpy as np
import os
from scipy.interpolate import interp1d

QE_LOCATION = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "qe.csv"))
def get_qe():
    """
    Returns an interpolator which gives the QE as a function of the number of electrons detected per pixel
    """
    data = np.loadtxt(QE_LOCATION, delimiter=',')
    return interp1d(data[:,0], data[:,1], fill_value=(data[0,1], data[-1,1]), bounds_error=False)