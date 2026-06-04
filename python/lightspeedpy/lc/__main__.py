import argparse, sys
from ..cli import add_dataset_args
from .lc import get_lc, add_lc


def main():
    parser = argparse.ArgumentParser(prog="lightspeedpy.lc", description="Lightspeed processing CLI for light curve extraction")

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
    get_lc(parser.parse_args())

def add_main():
    parser = argparse.ArgumentParser(prog="lightspeedpy.lc", description="Lightspeed processing CLI for light curve extraction")

    parser.add_argument("--inputs", nargs="+", required=True, help="Light curve files to combine")
    parser.add_argument("--output", required=True, help="Combined light curve file name")
    parser.add_argument("--clobber", help="Set to allow overwrite", action=argparse.BooleanOptionalAction)
    parser.set_defaults(func=add_lc)

    add_lc(parser.parse_args(sys.argv[2:]))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "add":
        add_main()
    else:
        main()