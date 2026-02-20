import argparse
from ..cli import add_dataset_args
from .cli import get_image

def main():
    parser = argparse.ArgumentParser(prog="lightspeedpy.image", description="Lightspeed processing CLI for image extraction")
    add_dataset_args(parser)
    parser.add_argument("--wcs", help="Apply wcs to final image", action=argparse.BooleanOptionalAction)
    parser.add_argument('--mode',
                    default='sum',
                    const='sum',
                    nargs='?',
                    choices=['sum', 'clip', 'weight'],
                    help='Analysis mode (sum, clip, or weight. Default: sum)'
    )
    get_image(parser.parse_args())

if __name__ == "__main__":
    main()