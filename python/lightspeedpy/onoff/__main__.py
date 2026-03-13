import argparse
from ..cli import add_dataset_args, get_dataset
from ..image.image import *
from ..ephemeris import Ephemeris
from .onoff import *

def get_on_off(args):
    data_set = get_dataset(args)
    data_set.display_filenames()
    ephemeris = Ephemeris(args.eph, data_set, args.observatory)

    if args.mode == "sum":
        image = get_summed_on_off(data_set, ephemeris, args.range)
    elif args.mode == "clip":
        image = get_clipped_on_off(data_set, ephemeris, args.range)
    elif args.mode == "weight":
        image = get_weighted_on_off_linearized(data_set, ephemeris, args.range)

    save_kwargs = vars(args)
    if "func" in save_kwargs: del save_kwargs["func"]
    image.nan_remove()
    image.save(args.output, args.wcs, args.clobber, save_kwargs)

def main():
    parser = argparse.ArgumentParser(prog="lightspeedpy.onoff", description="Plot on minus off image")
    add_dataset_args(parser)
    parser.add_argument("--eph", required=True, help="Ephemeris file")
    parser.add_argument("--range", required=True, help="Phase range. Format should be on_low:on_high,off_low:off_high")
    parser.add_argument("--observatory", help="Observatory (Default: Las Campanas Observatory)", default="Las Campanas Observatory")
    parser.add_argument("--wcs", help="Apply wcs to final image", action=argparse.BooleanOptionalAction)
    parser.add_argument('--mode',
                    default='sum',
                    const='sum',
                    nargs='?',
                    choices=['sum', 'clip', 'weight'],
                    help='Analysis mode (sum, clip, or weight. Default: sum)'
    )
    get_on_off(parser.parse_args())

if __name__ == "__main__":
    main()