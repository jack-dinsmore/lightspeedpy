import os
from ..dataset import DataSet

def get_dataset(args):
    if os.path.exists(args.output) and not args.clobber:
        raise Exception(f"Cannot save to {args.output}: file already exists and clobber is False")
    
    min_index = None if args.min_index is None else int(args.min_index)
    max_index = None if args.max_index is None else int(args.max_index)
    data_set = DataSet.from_first(args.input, min_index=min_index, max_index=max_index)

    print("Loaded files")
    data_set.display_filenames()

    if hasattr(args, "bias") and args.bias is not None:
        bias = DataSet.from_first(args.bias)
        data_set.set_bias(bias)
    if hasattr(args, "dark") and args.dark is not None:
        try:
            dark = DataSet.from_first(args.dark)
        except:
            dark = DataSet([args.dark])
        data_set.set_dark(dark)
    if hasattr(args, "flat") and args.flat is not None:
        try:
            flat = DataSet.from_first(args.flat)
        except:
            flat = DataSet([args.flat])
        data_set.set_flat(flat)
        
    return data_set

def stack_bias(args):
    data_set = get_dataset(args)
    data_set.pixel_properties.save(args.output, args.clobber)