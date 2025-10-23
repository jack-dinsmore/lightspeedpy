from . import _rust

import argparse, os
from .dataset import DataSet
from .image import get_linear_image, get_weighted_image, save_image, load_image
from .lc import get_linear_lc, get_weighted_lc
from .pulse_search import get_linear_fft

def get_dataset(args):
    if os.path.exists(args.output) and not args.clobber:
        raise Exception(f"Cannot save to {args.output}: file already exists and clobber is False")
    
    data_set = DataSet(args.input)
    if args.bias is not None:
        data_set.set_bias(args.bias)
    else:
        print("WARNING: No bias provided")
    if args.dark is not None:
        data_set.set_dark(args.dark)
    else:
        print("WARNING: No dark provided")
    if args.flat is not None:
        data_set.set_dark(args.flat)
    else:
        print("WARNING: No flat provided")
    return data_set

def get_image(args):
    data_set = get_dataset(args)

    if args.weight:
        image = get_weighted_image(data_set)
    else:
        image = get_linear_image(data_set)

    save_image(image, data_set, args)

def get_lc(args):
    data_set = get_dataset(args)

    if args.weight:
        if args.image is not None:
            image = load_image(args.image, assert_items=dict(weight=True, flat=None))
        else:
            image = get_weighted_image(data_set)
        lc = get_weighted_lc(data_set, image, args)
    else:
        lc = get_linear_lc(data_set, args)

    lc.save(data_set, args)

def get_fft(args):
    data_set = get_dataset(args)

    fft = get_linear_fft(data_set, args)

    fft.save(data_set, args)

def main():
    parser = argparse.ArgumentParser(prog="process", description="Lightspeed processing CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_image = subparsers.add_parser("image", help="Process a dataset, creating a merged image")
    parser_image.add_argument("--input", required=True, help="File name of dataset")
    parser_image.add_argument("--output", required=True, help="File name of output image")
    parser_image.add_argument("--bias", help="File name of bias")
    parser_image.add_argument("--dark", help="File name of dark")
    parser_image.add_argument("--flat", help="File name of flat")
    parser_image.add_argument("--weight", help="Set to use weights", action=argparse.BooleanOptionalAction)
    parser_image.add_argument("--clobber", help="Set to allow overwrite", action=argparse.BooleanOptionalAction)
    parser_image.set_defaults(func=get_image)

    parser_lc = subparsers.add_parser("lc", help="Process a dataset, creating a lightcurve")
    parser_lc.add_argument("--input", required=True, help="File name of dataset")
    parser_lc.add_argument("--output", required=True, help="File name of output image")
    parser_lc.add_argument("--roi", required=True, help="Region file")
    parser_lc.add_argument("--eph", required=True, help="Ephemeris file")
    parser_lc.add_argument("--bins", required=True, help="Number of bins", type=int)
    parser_lc.add_argument("--image", help="Name of the image, if it already exists")
    parser_lc.add_argument("--bias", help="File name of bias")
    parser_lc.add_argument("--dark", help="File name of dark")
    parser_lc.add_argument("--flat", help="File name of flat")
    parser_lc.add_argument("--weight", help="Set to use weights", action=argparse.BooleanOptionalAction)
    parser_lc.add_argument("--clobber", help="Set to allow overwrite", action=argparse.BooleanOptionalAction)
    parser_lc.set_defaults(func=get_lc)

    parser_fft = subparsers.add_parser("fft", help="Compute the fft of a data set")
    parser_fft.add_argument("--input", required=True, help="File name of dataset")
    parser_fft.add_argument("--output", required=True, help="File name of output fft")
    parser_fft.add_argument("--roi", required=True, help="Region file")
    parser_fft.add_argument("--bias", help="File name of bias")
    parser_fft.add_argument("--dark", help="File name of dark")
    parser_fft.add_argument("--flat", help="File name of flat")
    parser_fft.add_argument("--clobber", help="Set to allow overwrite", action=argparse.BooleanOptionalAction)
    parser_fft.set_defaults(func=get_fft)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
