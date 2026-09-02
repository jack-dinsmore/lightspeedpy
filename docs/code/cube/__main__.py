import argparse
from ..cli import add_dataset_args
from .split import split
from ..bias.stack_bias import stack_bias
from .cube import cube

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

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()