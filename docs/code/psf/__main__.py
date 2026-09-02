import argparse
from .psf import fit_gaussian

def main():
    parser = argparse.ArgumentParser(prog="lightspeedpy.TEMPLATE_NAME", description="DESCRIPTION")
    parser.add_argument("--input", required=True, help="Input image")
    parser.add_argument("--roi", required=True, help="Region in which to fit a PSF. The center will be taken as the initial position and the radius as the ROI")
    fit_gaussian(parser.parse_args())

if __name__ == "__main__":
    main()