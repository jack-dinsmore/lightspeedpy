import argparse
from ..cli import add_dataset_args, add_method_args
from .photo import get_photometry

def main():
    parser = argparse.ArgumentParser(prog="lightspeedpy.photo", description="Perform aperture photometry on a source")
    add_dataset_args(parser)
    parser.add_argument("--output", required=True, help="Output photometry file")
    parser.add_argument("--comp", required=True, help="Comparison star region")
    parser.add_argument("--target", required=True, help="Target star region")
    parser.add_argument("--comp-bg", help="Comparison star background region. If not provided, PSF weighting will be used, and the source region will be assumed to be a half-light contour.")
    parser.add_argument("--target-bg", help="Target star background region. If not provided, PSF weighting will be used, and the source region will be assumed to be a half-light contour.")
    parser.add_argument("--rebin", type=int, default=1, help="Number of frames to use per bin")
    parser.add_argument("--n-iterations", type=int, default=25, help="(Weight method) Number of iterations to use")
    parser.add_argument("--n-electrons", type=int, default=3, help="(Weight method) Number of electrons to simulate. Should be larger than the expected electrons per pixel.")
    add_method_args(parser)

    get_photometry(parser.parse())

if __name__ == "__main__":
    main()