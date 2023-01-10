import argparse
import internal.colmap.colmap

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", required=True, type=str)
parser.add_argument("--out-dir", type=str)
parser.add_argument("--out-name", default="transforms.npy")
parser.add_argument("--depth-percentile-min", type=float, default=.1)
parser.add_argument("--depth-percentile-max", type=float, default=99.9)
parser.add_argument("--max-scene-depth", type=float, default=5.0)
args = parser.parse_args()

internal.colmap.colmap.colmap2c2w(
    args.model_dir,
    args.out_dir,
    args.out_name,
    depth_percentile_min=args.depth_percentile_min,
    depth_percentile_max=args.depth_percentile_max,
    max_scene_depth=args.max_scene_depth,
)
