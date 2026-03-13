import argparse
from ..cli import add_dataset_args, get_dataset
from .image import *

def get_image(args):
    data_set = get_dataset(args)

    if args.mode == "sum":
        image = get_summed_image(data_set)
    elif args.mode == "clip":
        image = get_clipped_image(data_set)
    elif args.mode == "weight":
        image = get_weighted_image_linearized(data_set)
    else:
        raise Exception("Not reachable")

    save_kwargs = vars(args)
    if "func" in save_kwargs: del save_kwargs["func"]
    image.nan_remove()
    image.save(args.output, args.wcs, args.clobber, save_kwargs)


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