# This is where all the functions that take "args" as an argument go
import os, argparse
from .dataset import DataSet
from .image import *
from .lc import *
from .ephemeris import Ephemeris
from .pulse_search import get_linear_fft

def get_dataset(args):
    if os.path.exists(args.output) and not args.clobber:
        raise Exception(f"Cannot save to {args.output}: file already exists and clobber is False")
    
    data_set = DataSet(args.input, timing_offset=args.timing_offset)

    print("Loaded files")
    data_set.display_filenames()

    if args.bias is not None:
        data_set.set_bias(args.bias)
    if args.self_bias:
        data_set.set_self_bias()
    if not args.self_bias and not args.bias:
        print("WARNING: No bias provided")

    if args.dark is not None:
        data_set.set_dark(args.dark)
    else:
        print("WARNING: No dark provided")
    if args.flat is not None:
        data_set.set_dark(args.flat)
    else:
        print("WARNING: display_filenamesNo flat provided")
    return data_set

def get_image(args):
    data_set = get_dataset(args)

    if args.mode == "sum":
        image = get_summed_image(data_set)
    elif args.mode == "clip":
        image = get_clipped_image(data_set)
    elif args.mode == "weight":
        image = get_weighted_image_linearized(data_set)
    else:
        raise Exception("Not reachable")

    save_image(image, data_set, args)

def get_lc(args):
    data_set = get_dataset(args)
    ephemeris = Ephemeris(args.eph, data_set.get_timestamps())

    if args.mode == "sum":
        lc = get_summed_lc(data_set, args.bins, args.roi, ephemeris)
    elif args.mode == "clip":
        lc = get_clipped_lc(data_set, args.bins, args.roi, ephemeris)
    elif args.mode == "weight":
        if args.image is None:
            image = None
        else:
            image = load_image(args.image, assert_items=dict(flat=None))
        lc = get_weighted_lc_linearized(data_set, image, args.bins, args.roi, ephemeris)

    lc.save(data_set, args)

def get_fft(args):
    data_set = get_dataset(args)

    fft = get_linear_fft(data_set, args)

    fft.save(data_set, args)

def add_dataset_args(parser):
    parser.add_argument("--input", nargs='+', required=True, help="File name of dataset")
    parser.add_argument("--output", required=True, help="File name of output image")
    parser.add_argument("--bias", nargs='+', help="File name of bias")
    parser.add_argument("--self-bias", help="Measure bias from self", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dark", nargs='+', help="File name of dark")
    parser.add_argument("--flat", nargs='+', help="File name of flat")
    parser.add_argument("--timing-offset", help="Optional offset to apply to the start time (seconds)", type=float, default=0)
    parser.add_argument("--clobber", help="Set to allow overwrite", action=argparse.BooleanOptionalAction)