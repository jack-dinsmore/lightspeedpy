import argparse
from ..cli import add_dataset_args
from .cli import get_lc

def main():
    parser = argparse.ArgumentParser(prog="lightspeedpy.lc", description="Lightspeed processing CLI for light curve extraction")
    add_dataset_args(parser)
    parser.add_argument("--roi", required=True, help="Region file")
    parser.add_argument("--eph", required=True, help="Ephemeris file")
    parser.add_argument("--observatory", help="Observatory (Default: Las Campanas Observatory)", default="Las Campanas Observatory")
    parser.add_argument("--bins", required=True, help="Number of bins", type=int)
    parser.add_argument("--image", help="Name of the image, if it already exists")
    parser.add_argument('--mode',
                    default='sum',
                    const='sum',
                    nargs='?',
                    choices=['sum', 'clip', 'weight'],
                    help='Analysis mode (sum, clip, or weight. Default: sum)'
    )
    parser.set_defaults(func=get_lc)

if __name__ == "__main__":
    main()