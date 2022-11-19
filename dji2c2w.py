import numpy as np
import math

import argparse

import yaml

import internal.transform_matrix

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument("--exif-yaml", required=True)
parser.add_argument("--xyz-yaml", required=True)
parser.add_argument("--out-npy", help="c2w in opengl convention: camera looking -z", type=str, required=True)
args = parser.parse_args()

# load exif and xyz yaml
with open(args.exif_yaml, "r") as f:
    img_exif_dict = yaml.safe_load(f)
with open(args.xyz_yaml, "r") as f:
    img_xyz_dict = yaml.safe_load(f)

# generate camera-to-world transform matrix for each image
c2w_dict = {}
for i in img_exif_dict:
    exif = img_exif_dict[i]
    gimbal = exif["xmp"]["gimbal"]
    xyz = img_xyz_dict[i]

    c2w = internal.transform_matrix.generate_transform_matrix(
        [xyz[0], -xyz[1], xyz[2]],
        [
            math.radians(-gimbal["yaw"]),
            math.radians(-gimbal["pitch"]),
            math.radians(-gimbal["roll"])
        ],
    )
    c2w_dict[i] = c2w

np.save(args.out_npy, c2w_dict, allow_pickle=True)

# save a friendly version
for i in c2w_dict:
    c2w_dict[i] = c2w_dict[i].tolist()
with open(f"{args.out_npy}.yaml", "w") as f:
    yaml.dump(c2w_dict, f, allow_unicode=True)
