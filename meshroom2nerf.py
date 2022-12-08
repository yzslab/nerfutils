import os
import argparse
import numpy as np
import json
import internal.transform_matrix

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--sfm-file", required=True, help="path to cameras.sfm file")
parser.add_argument("--out", required=True, help="path to save output")
args = parser.parse_args()

with open(args.sfm_file, "r") as f:
    sfm = json.load(f)

pose_dict = dict()
for pose in sfm["poses"]:
    pose_dict[pose["poseId"]] = pose

cameras = dict()
for intrinsic in sfm["intrinsics"]:
    cameras[intrinsic["intrinsicId"]] = {
        "w": int(intrinsic["width"]),
        "h": int(intrinsic["height"]),
        "fl_x": float(intrinsic["pxFocalLength"]),
        "fl_y": float(intrinsic["pxFocalLength"]),
        "cx": float(intrinsic["principalPoint"][0]),
        "cy": float(intrinsic["principalPoint"][1]),
        "k1": float(intrinsic["distortionParams"][0]),
        "k2": float(intrinsic["distortionParams"][1]),
        "p1": float(intrinsic["distortionParams"][2]),
        "p2": 0.,
    }

images = dict()

pbar = tqdm(sfm["views"])
for view in pbar:
    pbar.set_description(view["path"])
    pose_id = view["poseId"]
    if pose_id not in pose_dict:
        continue

    intrinsic_id = view["intrinsicId"]

    pose = pose_dict[view["viewId"]]["pose"]
    rotation = pose["transform"]["rotation"]
    center = pose["transform"]["center"]

    c2w = np.array([
        [
            float(rotation[0]),
            float(rotation[1]),
            float(rotation[2]),
            float(center[0]),
        ],
        [
            float(rotation[3]),
            float(rotation[4]),
            float(rotation[5]),
            float(center[1]),
        ],
        [
            float(rotation[6]),
            float(rotation[7]),
            float(rotation[8]),
            float(center[2]),
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ]
    ])

    c2w = internal.transform_matrix.convert_pose(c2w)

    images[os.path.basename(view["path"])] = {
        "c2w": c2w,
        "camera_id": intrinsic_id,
    }

np.save(args.out, {
    "cameras": cameras,
    "images": images,
}, allow_pickle=True)
