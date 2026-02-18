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

    save_image(image, data_set, args)
