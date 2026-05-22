import argparse
from ..cli import add_dataset_args
from .lc import get_lc_no_errors, get_lc_errors, add_lc

def get_lc(args):
    if args.errors:
        get_lc_errors(args)
    else:
        get_lc_no_errors(args)

def main():
    parser = argparse.ArgumentParser(prog="lightspeedpy.lc", description="Lightspeed processing CLI for light curve extraction")
    subparsers = parser.add_subparsers(dest="command")

    add_dataset_args(parser)
    parser.add_argument("--roi", required=True, help="Region file")
    parser.add_argument("--eph", required=True, help="Ephemeris file")
    parser.add_argument("--observatory", help="Observatory (Default: Las Campanas Observatory)", default="Las Campanas Observatory")
    parser.add_argument("--bins", required=True, help="Number of bins", type=int)
    parser.add_argument("--psf", type=float, help="Radius of the PSF, in arcseconds")
    parser.add_argument("--errors", help="Set to estimate bootstrapped errors", action=argparse.BooleanOptionalAction)
    parser.add_argument('--mode',
                    default='sum',
                    const='sum',
                    nargs='?',
                    choices=['sum', 'clip', 'weight'],
                    help='Analysis mode (sum, clip, or weight. Default: sum)'
    )
    parser.set_defaults(func=get_lc)

    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("--inputs", nargs="+", required=True, help="Light curve files to combine")
    add_parser.add_argument("--output", required=True, help="Combined light curve file name")
    add_parser.add_argument("--clobber", help="Set to allow overwrite", action=argparse.BooleanOptionalAction)
    add_parser.set_defaults(func=add_lc)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()