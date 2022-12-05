import argparse
import internal.colmap.colmap

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", required=True, type=str)
parser.add_argument("--out-dir", required=True, type=str)
parser.add_argument("--out-name", default="transforms.npy")
args = parser.parse_args()

internal.colmap.colmap.colmap2c2w(args.model_dir, args.out_dir, args.out_name)
