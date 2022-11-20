import os
import argparse
from tqdm import tqdm

from internal.block_nerf import load_block_list, load_block_information
from convert2ngp import convert2ngp, parser_configure

parser = argparse.ArgumentParser()
parser.add_argument("--block-dir", required=True, type=str)
parser_configure(parser)
args = parser.parse_args()

block_list = load_block_list(args.block_dir)

pbar = tqdm(block_list)
for block_id in pbar:
    pbar.set_description(f"Processing ({block_id})")
    block_dir = os.path.join(args.block_dir, block_id)
    block_information = load_block_information(args.block_dir, block_id)

    scale = None
    offset = None
    if args.recenter is False:
        offset = [0, 0, 0]
        scale = 0.01
        offset[0] = scale * -block_information["c"][0]
        offset[1] = scale * block_information["c"][1]

    convert2ngp(
        transforms_npy_path=os.path.join(block_dir, "aligned_transforms.npy"),
        out_path=os.path.join(block_dir, "aligned_ngp.json"),
        image_dir="images",
        down_sample=1,
        reorient_scene=args.reorient,
        recenter_scene=args.recenter,
        aabb_scale=args.aabb_scale,
        scene_scale=args.scene_scale,
        scale=scale,
        offset=offset
    )