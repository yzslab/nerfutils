import argparse


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to directory storing images taken by DJI Drone")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--down_sample", type=int, default=1)
    parser.add_argument("--image_dir", type=str, default=None, help="path to down sampled images directory")
    parser.add_argument("--aabb_scale", type=int, default=16, help="only work for instant-ngp")
    parser.add_argument("--scene_scale", type=float, default=1)

    args = parser.parse_args()

    if args.image_dir is None:
        args.image_dir = args.path

    return args
