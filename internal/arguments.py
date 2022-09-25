import argparse


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to directory storing images taken by DJI Drone")
    parser.add_argument("--out")
    parser.add_argument("--down_sample", type=int, default=1)
    parser.add_argument("--images", type=str, default="images")
    return parser.parse_args()
