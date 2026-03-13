import argparse
from ..cli import add_dataset_args, get_dataset

def main():
    parser = argparse.ArgumentParser(prog="lightspeedpy.TEMPLATE_NAME", description="DESCRIPTION")
    add_dataset_args(parser) # 

    # Add any additional command line arguments here

    args = args.parse()

    # You can load the data set like this:
    dataset = get_dataset(args)

    # Now execute your reduction code here.

    # You can create any additional files you wish in this directory to hold the code. Import them with 
    # from . import FILE_NAME

if __name__ == "__main__":
    main()