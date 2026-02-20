import numpy as np

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