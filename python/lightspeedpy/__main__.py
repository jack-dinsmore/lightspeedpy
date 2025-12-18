import argparse
from . import utils

def main():
    parser = argparse.ArgumentParser(prog="process", description="Lightspeed processing CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_image = subparsers.add_parser("image", help="Process a dataset, creating a merged image")
    utils.add_dataset_args(parser_image)
    parser_image.add_argument('--mode',
                    default='sum',
                    const='sum',
                    nargs='?',
                    choices=['sum', 'clip', 'weight'],
                    help='Analysis mode (sum, clip, or weight. Default: sum)'
    )
    parser_image.set_defaults(func=utils.get_image)

    parser_lc = subparsers.add_parser("lc", help="Process a dataset, creating a lightcurve")
    utils.add_dataset_args(parser_lc)
    parser_lc.add_argument("--roi", required=True, help="Region file")
    parser_lc.add_argument("--eph", required=True, help="Ephemeris file")
    parser_lc.add_argument("--observatory", help="Observatory (Default: Las Campanas Observatory)", default="Las Campanas Observatory")
    parser_lc.add_argument("--bins", required=True, help="Number of bins", type=int)
    parser_lc.add_argument("--image", help="Name of the image, if it already exists")
    parser_lc.add_argument('--mode',
                    default='sum',
                    const='sum',
                    nargs='?',
                    choices=['sum', 'clip', 'weight'],
                    help='Analysis mode (sum, clip, or weight. Default: sum)'
    )
    parser_lc.set_defaults(func=utils.get_lc)

    parser_fft = subparsers.add_parser("fft", help="Compute the fft of a data set")
    utils.add_dataset_args(parser_fft)
    parser_fft.add_argument("--roi", required=True, help="Region file")
    parser_fft.set_defaults(func=utils.get_fft)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
