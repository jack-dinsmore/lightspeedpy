import numpy as np
from astropy.io import fits
from scipy.optimize import least_squares
import os
from ..cli import get_dataset
from ..regions import CircleRegion
from ..psf_utils import moffat_min_fn

def shift_image(img, dx=0, dy=0, fill=np.nan):
    out = np.full_like(img, fill)

    y_src_start = max(0, -dy)
    y_src_end   = img.shape[-2] - max(0, dy)
    x_src_start = max(0, -dx)
    x_src_end   = img.shape[-1] - max(0, dx)

    y_dst_start = max(0, dy)
    y_dst_end   = y_dst_start + (y_src_end - y_src_start)
    x_dst_start = max(0, dx)
    x_dst_end   = x_dst_start + (x_src_end - x_src_start)

    out[..., y_dst_start:y_dst_end, x_dst_start:x_dst_end] = \
        img[..., y_src_start:y_src_end, x_src_start:x_src_end]

    return out

def shift(args):
    data_set = get_dataset(args)
    xs, ys = np.meshgrid(np.arange(data_set.image_shape[1]), np.arange(data_set.image_shape[0]))

    # Get the fitting region
    if not os.path.exists(args.roi):
        raise Exception(f"The region file {args.roi} does not exist")
    try:
        reg = CircleRegion.load(args.roi)
    except:
        raise Exception("Please provide a circular region")
    x0, y0 = reg.x, reg.y
    radius = np.sqrt(reg.radius2)
    xmin = int(np.clip(x0-radius, 0, data_set.image_shape[1]))
    xmax = int(np.clip(x0+radius, 0, data_set.image_shape[1]))
    ymin = int(np.clip(y0-radius, 0, data_set.image_shape[0]))
    ymax = int(np.clip(y0+radius, 0, data_set.image_shape[0]))

    # Make the output directory
    if os.path.exists(args.output):
        for f in os.listdir(args.output):
            if not args.clobber:
                raise Exception("Output directory is not empty. You must set clobber to overwrite it")
            os.remove(f"{args.output}/{f}")
    else:
        os.mkdir(args.output)

    # Get the shifts
    results = []
    for frame in data_set:
        image = frame.image[ymin:ymax,xmin:xmax]
        xs, ys = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        initial_params = (image.shape[1]/2, image.shape[0]/2, 5, 5, 0, 1, 1, 0,)
        result = least_squares(moffat_min_fn,
            x0=initial_params,
            bounds=[(0, 0, 0, 0, -np.inf, 0.2, 0, 0), (image.shape[1], image.shape[0], np.inf, np.inf, np.inf, 10, np.inf, np.inf)],
            args=(xs, ys, image),
        )
        result.x[0] -= image.shape[1]/2
        result.x[1] -= image.shape[0]/2
        results.append(result.x)

    i = 0
    for filename in data_set.filenames:
        print(filename)
        out_name = f"{args.output}/{filename.split('/')[-1]}"
        print(out_name)

        psf_table = {"X": [], "Y": [], "MAJOR": [], "MINOR": [], "THETA": [], "ALPHA": []}

        with fits.open(filename) as hdul:
            out_cube = np.zeros_like(hdul[1].data)
            for j, image in enumerate(hdul[1].data):
                result = results[i]

                if not args.noshift:
                    shift_x = -int(np.round(result[0]))
                    shift_y = -int(np.round(result[1]))
                    result[0] -= shift_x
                    result[1] -= shift_y  
                    out_cube[j] = shift_image(image, shift_x, shift_y)

                psf_table["X"].append(result[0])
                psf_table["Y"].append(result[1])
                psf_table["MAJOR"].append(result[2])
                psf_table["MINOR"].append(result[3])
                psf_table["THETA"].append(result[4])
                psf_table["ALPHA"].append(result[5])
                i += 1

            if args.noshift:
                out_cube = hdul[1].data

            cols = []
            for keys, values in psf_table.items():
                cols.append(fits.Column(name=keys, format="D", array=np.array(values)))
            psf_hdu = fits.BinTableHDU.from_columns(cols, name="PSF")

            header = hdul[1].header.copy()
            if args.noshift:
                header["SHIFTED"] = "F"
            else:
                header["SHIFTED"] = "T"
            cube_hdu = fits.ImageHDU(data=out_cube, name="DATA_CUBE", header=header)
            hdul = fits.HDUList([hdul[0], cube_hdu, hdul[1], psf_hdu])
            hdul.writeto(out_name, overwrite=args.clobber)