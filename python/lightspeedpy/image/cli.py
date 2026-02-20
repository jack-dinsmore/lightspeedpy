from ..cli import get_dataset
from .image import *

def get_image(args):
    data_set = get_dataset(args)

    if args.mode == "sum":
        image = get_summed_image(data_set)
    elif args.mode == "clip":
        image = get_clipped_image(data_set)
    elif args.mode == "weight":
        image = get_weighted_image(data_set)
    else:
        raise Exception("Not reachable")

    save_kwargs = vars(args)
    if "func" in save_kwargs: del save_kwargs["func"]
    image.save(args.output, args.wcs, args.clobber, save_kwargs)
