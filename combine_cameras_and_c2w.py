import argparse
import numpy as np
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--camera-npy")
parser.add_argument("--c2w-npy")
parser.add_argument("--out-npy", help="transforms.npy format")
args = parser.parse_args()

print("loading cameras")
cameras = np.load(args.camera_npy, allow_pickle=True).item()
print("loading camera poses")
c2w = np.load(args.c2w_npy, allow_pickle=True).item()

transforms = {
    "cameras": [],
    "intrinsics_matrix": [],
    "image_c2w": {},
    "image_camera_id": {},
}

for i in cameras["cameras"]:
    camera = {}
    for key in ["angle_x", "angle_y", "cx", "cy", "fl_x", "fl_y", "w", "h"]:
        camera[key] = i[key]
    transforms["cameras"].append(camera)
    transforms["intrinsics_matrix"].append(i["intrinsics_matrix"])

for image_name in cameras["image_cameras"]:
    transforms["image_c2w"][image_name] = c2w[image_name]
    transforms["image_camera_id"][image_name] = cameras["image_cameras"][image_name]

print("saving result")
np.save(args.out_npy, transforms, allow_pickle=True)

for id, i in enumerate(transforms["intrinsics_matrix"]):
    transforms["intrinsics_matrix"][id] = transforms["intrinsics_matrix"][id].tolist()
for i in transforms["image_c2w"]:
    transforms["image_c2w"][i] = transforms["image_c2w"][i].tolist()

with open(f"{args.out_npy}.yaml", "w") as f:
    yaml.dump(transforms, f, allow_unicode=True)
