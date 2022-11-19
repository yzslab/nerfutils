import sys
import argparse


def get_parameters(parser_configure=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--transforms-npy", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--down-sample", type=int, default=1)
    parser.add_argument("--image-dir", type=str, default="images", help="path to down sampled images directory")

    if parser_configure is not None:
        parser_configure(parser)

    args = parser.parse_args()

    # if args.image_dir is None:
    #     args.image_dir = args.path

    if (args.image_dir != "images" and args.down_sample == 1) or (args.image_dir == "images" and args.down_sample != 1):
        print("WARNING: --image-dir and --down-sample not changed at the same time", file=sys.stderr)

    return args
