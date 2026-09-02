import argparse
from .stack_bias import stack_bias

def main():
    parser = argparse.ArgumentParser(prog="lightspeedpy.bias", description="Create a bias file from bias frames")
    parser.add_argument("--input", required=True, help="File name of dataset")
    parser.add_argument("--output", required=True, help="File name of output image")
    parser.add_argument("--map-noise", help="Set to additionally map the noise distribution of each pixel", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dbg-noise", help="Create a debug figure of noise histograms and the best-fit model", action=argparse.BooleanOptionalAction)
    parser.add_argument("--min-index", help="Minimum cube index")
    parser.add_argument("--max-index", help="Maximum cube index")
    parser.add_argument("--clobber", help="Set to allow overwrite", action=argparse.BooleanOptionalAction)

    stack_bias(parser.parse_args())

if __name__ == "__main__":
    main()