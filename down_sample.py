import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src")
parser.add_argument("--dst")
parser.add_argument("--factor", type=int)
args = parser.parse_args()

for root, dirs, files in os.walk(args.src):
    print(root)
    relative_path = root.replace(args.src, "").strip("/")
    dst_path = os.path.join(args.dst, relative_path)
    os.makedirs(dst_path, exist_ok=True)
    os.system(f"find {root} -maxdepth 1 -name '*' -type f -print | xargs -n 1 -P $(nproc) mogrify -quality 100 -resize {1. / args.factor * 100}% -path {dst_path}")
    # os.system(f"mogrify -quality 100 -resize {1. / args.factor * 100}% -path {dst_path} {root}/*")
