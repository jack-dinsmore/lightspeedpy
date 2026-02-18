import argparse
from ..cli import add_dataset_args
from .cli import get_image

def main():
    parser = argparse.ArgumentParser(prog="lightspeedpy.image", description="Lightspeed processing CLI for image extraction")
    parser.add_argument('--mode',
                    default='sum',
                    const='sum',
                    nargs='?',
                    choices=['sum', 'clip', 'weight'],
                    help='Analysis mode (sum, clip, or weight. Default: sum)'
    )
    parser.set_defaults(func=get_image)

if __name__ == "__main__":
    main()