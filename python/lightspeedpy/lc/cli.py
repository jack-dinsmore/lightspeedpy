# This is where all the functions that take "args" as an argument go
from ..image.image import *
from ..cli import get_dataset
from .lc import *

def get_lc(args):
    data_set = get_dataset(args)

    # data_set.bootstrap()

    ephemeris = Ephemeris(args.eph, data_set._get_timestamps())

    if args.mode == "sum":
        lc = get_summed_lc(data_set, args.bins, args.roi, ephemeris)
    elif args.mode == "clip":
        lc = get_clipped_lc(data_set, args.bins, args.roi, ephemeris)
    elif args.mode == "weight":
        if args.image is None:
            image = None
        else:
            image = load_image(args.image, assert_items=dict(flat=None))
        lc = get_weighted_lc_linearized(data_set, image, args.bins, args.roi, ephemeris)

    lc.save(data_set, args)
