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
    "cameras": {},
    "images": {},
}

for idx, i in enumerate(cameras["cameras"]):
    camera = {}
    for key in ["cx", "cy", "fl_x", "fl_y", "w", "h"]:
        camera[key] = i[key]
    transforms["cameras"][idx] = camera

for image_name in cameras["image_cameras"]:
    transforms["images"][image_name] = {
        "c2w": c2w[image_name],
        "camera_id": cameras["image_cameras"][image_name],
    }

print("saving result")
np.save(args.out_npy, transforms, allow_pickle=True)

for i in transforms["images"]:
    transforms["images"][i]["c2w"] = transforms["images"][i]["c2w"].tolist()

with open(f"{args.out_npy}.yaml", "w") as f:
    yaml.dump(transforms, f, allow_unicode=True)
