from astropy.io import fits
import os
from ..constants import FORBIDDEN_KEYWORDS

def split(args):
    components = args.input.split('/')[-1][:-5].split('_')
    cube = components[-1]
    time = components[-2]
    date = components[-3]
    name = '_'.join(components[:-3])
    stem = f"{name}_{cube}_{date}_{time}"

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    with fits.open(args.input) as hdul:
        for i in range(hdul[1].data.shape[0]):
            h0 = fits.PrimaryHDU()
            h1 = fits.ImageHDU(data=[hdul[1].data[i]])
            h2 = fits.BinTableHDU(data=hdul[2].data[i:i+1])

            for key, value in hdul[0].header.items():
                if key not in FORBIDDEN_KEYWORDS:
                    if len(key) > 8: key = f"HIERARCH {key}"
                    h0.header[key] = value

            for key, value in hdul[1].header.items():
                if key not in FORBIDDEN_KEYWORDS:
                    if len(key) > 8: key = f"HIERARCH {key}"
                    h1.header[key] = value
                    h2.header[key] = value

            filename = f"{args.output}/{stem}_cube{i+1:03d}.fits"
            fits.HDUList([h0, h1, h2]).writeto(filename, overwrite=args.clobber)