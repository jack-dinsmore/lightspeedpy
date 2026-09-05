import argparse
from ..cli import add_dataset_args
from .split import split
from .cube import cube
from .shift import shift

def main():
    parser = argparse.ArgumentParser(prog="lightspeedpy.cube", description="Manipulate lightspeed cubes")
    subparsers = parser.add_subparsers(required=True)

    parser_split = subparsers.add_parser('split', help='Split a cube')
    parser_split.add_argument("--input", required=True, help="File name of cube")
    parser_split.add_argument("--output", required=True, help="Output directory")
    parser_split.add_argument("--clobber", help="Set to allow overwrite", action=argparse.BooleanOptionalAction)
    parser_split.set_defaults(func=split)

    parser_cube = subparsers.add_parser('stack', help='Create a data cube')
    add_dataset_args(parser_cube)
    parser_cube.set_defaults(func=cube)

    parser_shift = subparsers.add_parser('shift', help='Take out PSF variation by shifting to align the frames')
    add_dataset_args(parser_shift)
    parser_shift.add_argument("--roi", required=True, help="ROI to measure the PSF from")
    parser_shift.add_argument("--noshift", help="Set if you want to keep the cube untouched and just write the PSFs to the file", action=argparse.BooleanOptionalAction)
    parser_shift.set_defaults(func=shift)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()