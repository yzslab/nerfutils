import argparse


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to directory storing images taken by DJI Drone")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--down_sample", type=int, default=1)
    parser.add_argument("--images", type=str, default="images")
    parser.add_argument("--aabb_scale", type=int, default=16)
    parser.add_argument("--scene_scale", type=float, default=1)
    return parser.parse_args()
