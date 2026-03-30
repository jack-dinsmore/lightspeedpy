# This is where all the functions that take "args" as an argument go
import os, argparse
from astropy.io import fits
from .dataset import DataSet
from .pixel_properties import PixelProperties

def get_dataset(args):
    """
    Get the :class:`DataSet` from an `argparse` parser, assuming the arguments correspond to those created by :func:`cli.add_dataset_args`.

    Parameters
    ----------
    args : argparse args
        Command line arguments
    """
    if os.path.exists(args.output) and not args.clobber:
        raise Exception(f"Cannot save to {args.output}: file already exists and clobber is False")
    
    min_index = None if args.min_index is None else int(args.min_index)
    max_index = None if args.max_index is None else int(args.max_index)
    data_set = DataSet.from_first(args.input, min_index=min_index, max_index=max_index)
    data_set.apply_timing_offset(args.timing_offset)

    print("Loaded files")
    data_set.display_filenames()
    if not args.self_bias and not args.bias:
        print("WARNING: No bias provided")

    def set_bias(ds):
        if args.bias is not None:
            is_pix_prop = False
            if os.path.exists(args.bias):
                with fits.open(args.bias) as hdul:
                    if "PIXPROP" in hdul[0].header and hdul[0].header["PIXPROP"] == "T":
                        is_pix_prop = True
            if is_pix_prop:
                ds.set_bias(PixelProperties.load(args.bias))
            else:
                ds.set_bias(DataSet.from_first(args.bias))
        if args.self_bias:
            ds.set_self_bias()

    set_bias(data_set)

    if args.dark is not None:
        try:
            dark = DataSet.from_first(args.dark, cut_cr=False)
        except:
            dark = DataSet([args.dark])
        set_bias(dark)
        data_set.set_dark(dark)
    else:
        print("WARNING: No dark provided")
    if args.flat is not None:
        try:
            flat = DataSet.from_first(args.flat)
        except:
            flat = DataSet([args.flat])
        set_bias(flat)
        data_set.set_flat(flat)
    else:
        print("WARNING: No flat provided")
    return data_set

def add_dataset_args(parser):
    """
    Add the standard lightspeedpy arguments to an `argparse` parser. These include input, output, bias, self-bias, flat, flat, frames, timing-offset, and clobber.

    Parameters
    ----------
    parser : argparse parser
        Parser to which to add arguments
    """
    parser.add_argument("--input", required=True, help="File name of dataset")
    parser.add_argument("--output", required=True, help="File name of output image")
    parser.add_argument("--bias", help="File name of bias")
    parser.add_argument("--self-bias", help="Measure bias from self", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dark", help="File name of dark")
    parser.add_argument("--flat", help="File name of flat")
    parser.add_argument("--min-index", help="Minimum cube index")
    parser.add_argument("--max-index", help="Maximum cube index")
    parser.add_argument("--timing-offset", help="Optional offset to apply to the start time (seconds)", type=float, default=0)
    parser.add_argument("--clobber", help="Set to allow overwrite", action=argparse.BooleanOptionalAction)