import os
import argparse
import yaml
from tqdm import tqdm
import internal.colmap

parser = argparse.ArgumentParser()
parser.add_argument("--block-dir", required=True, type=str)
args = parser.parse_args()

with open(os.path.join(args.block_dir, "block-list.yaml"), "r") as f:
    block_list = yaml.safe_load(f)

pbar = tqdm(block_list)
for block in pbar:
    pbar.set_description(f"Processing ({block})")
    base_dir = os.path.join(args.block_dir, block)
    internal.colmap.colmap2c2w(os.path.join(base_dir, "sparse_text"), base_dir)
