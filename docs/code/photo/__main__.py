import argparse
from ..cli import add_dataset_args, add_method_args
from .photo import get_photometry

def main():
    parser = argparse.ArgumentParser(prog="lightspeedpy.photo", description="Perform aperture photometry on a source")
    add_dataset_args(parser)
    parser.add_argument("--src", required=True, help="Target star region")
    parser.add_argument("--bkg", help="Target star background region. If not provided, PSF weighting will be used, and the source region will be assumed to be a half-light contour.")
    parser.add_argument("--rebin", type=int, default=1, help="Number of frames to use per bin")
    parser.add_argument("--errors", help="Set to estimate bootstrapped errors", action=argparse.BooleanOptionalAction)
    add_method_args(parser)

    get_photometry(parser.parse_args())

if __name__ == "__main__":
    main()