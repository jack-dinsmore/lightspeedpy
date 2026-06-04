import numpy as np
from astropy.io import fits
import numpy as np
from ..cli import get_dataset

def cube(args):
    data_set = get_dataset(args)
    cube = np.zeros((data_set.num_frames(),data_set.image_shape[0], data_set.image_shape[1]))
    print(cube.shape)
    for i, frame in enumerate(data_set):
        cube[i] = frame.image
    
    primary_hdu = fits.PrimaryHDU(cube, header=data_set.header0)
    cube_hdu = fits.ImageHDU(data=cube, name="CUBE", header=data_set.header1)
    hdul = fits.HDUList([primary_hdu, cube_hdu])
    hdul.writeto(args.output, overwrite=args.clobber)
