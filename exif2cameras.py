import argparse

import numpy as np

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--exif-yaml", required=True)
parser.add_argument("--out-npy", required=True)
parser.add_argument("--sensor-size", nargs="*", type=float)
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
        height = exif["ImageHeight"]

        # focal length
        focal_length = 0
        if "FocalLengthIn35mmFilm" in exif:
            focal_length = float(exif["FocalLengthIn35mmFilm"])
        if focal_length > 0:
            sensor_size = [36., 24.]
        else:
            focal_length = float(exif["FocalLength"])
            sensor_size = args.sensor_size

        ## 35 mm movie film dimensions: 36mm x 24mm
        camera_angle_x = float(2 * np.arctan((sensor_size[0] / 2) / focal_length))
        camera_angle_y = float(2 * np.arctan((sensor_size[1] / 2) / focal_length))

        fl_in_pixel = focal_length * width / sensor_size[0]

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
