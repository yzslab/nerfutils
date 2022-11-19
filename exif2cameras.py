import argparse

import numpy as np

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--exif-yaml", required=True)
parser.add_argument("--out-npy", required=True)
args = parser.parse_args()

with open(args.exif_yaml, "r") as f:
    img_exif = yaml.safe_load(f)

camera_count = -1
camera_index = {}  # camera model name to list id
cameras = []  # camera list
image_cameras = {}  # image name to camera list id

for i in img_exif:
    exif = img_exif[i]
    camera_model = exif["Model"]

    # if camera model not in camera_index, create it
    if camera_model not in camera_index:
        width = exif["ImageWidth"]
        height = exif["ImageLength"]

        # focal length
        fl_in_35mm = exif["FocalLengthIn35mmFilm"]
        ## 35 mm movie film dimensions: 36mm x 24mm
        camera_angle_x = float(2 * np.arctan((36 / 2) / fl_in_35mm))
        camera_angle_y = float(2 * np.arctan((24 / 2) / fl_in_35mm))

        fl_in_pixel = fl_in_35mm * width / 36

        # principle point
        cx = width / 2
        cy = height / 2

        intrinsics_matrix = np.asarray([
            [fl_in_pixel, 0, cx, 0],
            [0, fl_in_pixel, cy, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        camera_count += 1
        camera_index[camera_model] = camera_count
        cameras.append({
            "w": width,
            "h": height,
            "angle_x": camera_angle_x,
            "angle_y": camera_angle_y,
            "fl_35mm": fl_in_35mm,
            "fl_x": fl_in_pixel,
            "fl_y": fl_in_pixel,
            "cx": cx,
            "cy": cy,
            "intrinsics_matrix": intrinsics_matrix,
        })

    image_cameras[i] = camera_count

np.save(args.out_npy, {
    "cameras": cameras,
    "image_cameras": image_cameras,
}, allow_pickle=True)

for i in cameras:
    i["intrinsics_matrix"] = i["intrinsics_matrix"].tolist()
with open(f"{args.out_npy}.yaml", "w") as f:
    yaml.dump({
        "cameras": cameras,
        "image_cameras": image_cameras,
    }, f, allow_unicode=True)
