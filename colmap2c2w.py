import argparse
import internal.colmap

parser = argparse.ArgumentParser()
parser.add_argument("--text", required=True, type=str)
parser.add_argument("--out-dir", required=True, type=str)
args = parser.parse_args()

internal.colmap.colmap2c2w(args.text, args.out_dir)
