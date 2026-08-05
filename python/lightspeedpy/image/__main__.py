import argparse
from ..cli import add_dataset_args, get_dataset, add_method_args
from .image import make_image

def get_image(args):
    data_set = get_dataset(args)

    image = make_image(data_set, args.mode, args.n_electrons, args.n_iterations)

    save_kwargs = vars(args)
    if "func" in save_kwargs: del save_kwargs["func"]
    image.nan_remove()
    if args.smooth is not None:
        image.smooth(float(args.smooth))
    image.save(args.output, args.wcs, args.clobber, save_kwargs)

def main():
    parser = argparse.ArgumentParser(prog="lightspeedpy.image", description="Lightspeed processing CLI for image extraction")
    add_dataset_args(parser)
    parser.add_argument("--wcs", help="Apply wcs to final image", action=argparse.BooleanOptionalAction)
    parser.add_argument("--smooth",  required=False, help="Gaussian smoothing sigma (pixels)")
    add_method_args(parser)
    get_image(parser.parse_args())

if __name__ == "__main__":
    main()