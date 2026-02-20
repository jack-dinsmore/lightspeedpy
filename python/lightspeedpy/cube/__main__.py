import argparse
from .split import split
from .stack import stack_bias

def main():
    parser = argparse.ArgumentParser(prog="lightspeedpy.cube", description="Manipulate lightspeed cubes")
    subparsers = parser.add_subparsers(required=True)

    parser_split = subparsers.add_parser('split', help='Split a cube')
    parser_split.add_argument("--input", required=True, help="File name of cube")
    parser_split.add_argument("--output", required=True, help="Output directory")
    parser_split.add_argument("--clobber", help="Set to allow overwrite", action=argparse.BooleanOptionalAction)
    parser_split.set_defaults(func=split)

    parser_stack_bias = subparsers.add_parser('stack-bias', help='Stack up bias frames')
    parser_stack_bias.add_argument("--input", required=True, help="File name of dataset")
    parser_stack_bias.add_argument("--output", required=True, help="File name of output image")
    parser_stack_bias.add_argument("--min-index", help="Minimum cube index")
    parser_stack_bias.add_argument("--max-index", help="Maximum cube index")
    parser_stack_bias.add_argument("--clobber", help="Set to allow overwrite", action=argparse.BooleanOptionalAction)
    parser_stack_bias.set_defaults(func=stack_bias)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()