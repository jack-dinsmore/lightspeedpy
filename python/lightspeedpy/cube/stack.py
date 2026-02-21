import os
from astropy.io import fits
from ..dataset import DataSet
from ..pixel_properties import PixelProperties

def get_dataset(args):
    if os.path.exists(args.output) and not args.clobber:
        raise Exception(f"Cannot save to {args.output}: file already exists and clobber is False")
    
    min_index = None if args.min_index is None else int(args.min_index)
    max_index = None if args.max_index is None else int(args.max_index)
    data_set = DataSet.from_first(args.input, min_index=min_index, max_index=max_index)

    print("Loaded files")
    data_set.display_filenames()

    def set_bias(ds):
        is_pix_prop = False
        if os.path.exists(args.bias):
            with fits.open(args.bias) as hdul:
                if "PIXPROP" in hdul[0].header and hdul[0].header["PIXPROP"] == "T":
                    is_pix_prop = True
        if is_pix_prop:
            ds.set_bias(PixelProperties.load(args.bias))
        else:
            ds.set_bias(DataSet.from_first(args.bias))

    if hasattr(args, "bias") and args.bias is not None:
        bias = DataSet.from_first(args.bias)
        set_bias(data_set)
    if hasattr(args, "dark") and args.dark is not None:
        try:
            dark = DataSet.from_first(args.dark)
        except:
            dark = DataSet([args.dark])
        set_bias(dark)
        data_set.set_dark(dark)
    if hasattr(args, "flat") and args.flat is not None:
        try:
            flat = DataSet.from_first(args.flat)
        except:
            flat = DataSet([args.flat])
        flat.set_bias(bias)
        set_bias(flat)
        data_set.set_flat(flat)
        
    return data_set

def stack_bias(args):
    data_set = get_dataset(args)
    pp = PixelProperties.from_bias(data_set, data_set)
    pp.save(args.output, args.clobber)