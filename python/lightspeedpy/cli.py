# This is where all the functions that take "args" as an argument go
import os, argparse
from .dataset import DataSet

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
    
    frames = None
    if args.frames is not None:
        if len(args.input) > 1:
            raise Exception("You cannot provide the frames argument if using more than one run.")
        frames = [int(f) for f in args.frames]
    data_set = DataSet.from_files(args.input, timing_offset=args.timing_offset, frames=frames)

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
        data_set.set_flat(args.flat)
    else:
        print("WARNING: No flat provided")
    return data_set

def add_dataset_args(parser):
    """
    Add the standard lightspeedpy arguments to an `argparse` parser. These include input, output, bias, self-bias, dark, flat, frames, timing-offset, and clobber.

    Parameters
    ----------
    parser : argparse parser
        Parser to which to add arguments
    """
    parser.add_argument("--input", nargs='+', required=True, help="File name of dataset")
    parser.add_argument("--output", required=True, help="File name of output image")
    parser.add_argument("--bias", nargs='+', help="File name of bias")
    parser.add_argument("--self-bias", help="Measure bias from self", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dark", nargs='+', help="File name of dark")
    parser.add_argument("--flat", nargs='+', help="File name of flat")
    parser.add_argument("--frames", nargs='+', help="Frame range to use. Zero indexed, inclusive on the start of the range and exclusive on the end, like python. If you wish to use only one frame, provide only one index. Default: use all frames.")
    parser.add_argument("--timing-offset", help="Optional offset to apply to the start time (seconds)", type=float, default=0)
    parser.add_argument("--clobber", help="Set to allow overwrite", action=argparse.BooleanOptionalAction)