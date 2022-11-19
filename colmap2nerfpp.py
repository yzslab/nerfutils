import os
import json

import argparse

import internal.transform_matrix
from convert2nerfpp import save_image_pose
from internal.colmap import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to colmap sparse text database directory")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default=None, help="path to images directory")
    parser.add_argument("--scene_scale", type=float, default=1)
    args = parser.parse_args()

    cameras, intrinsics_matrix = parse_cameras_txt(os.path.join(args.path, "cameras.txt"))
    transform_matrix, camera_id = parse_images_txt(os.path.join(args.path, "images.txt"))

    transform_matrix = internal.transform_matrix.recenter(transform_matrix, args.scene_scale)

    transform_matrix = dict(sorted(transform_matrix.items()))

    image_directory = args.image_dir

    visualize_cameras = [{}, {}]

    os.makedirs(args.out, exist_ok=True)
    os.chdir(args.out)

    os.makedirs("camera_path", exist_ok=True)
    os.makedirs("train", exist_ok=True)
    os.makedirs("test", exist_ok=True)
    os.makedirs("validation", exist_ok=True)

    image_count = 0
    for image_filename in transform_matrix:
        image_count += 1

        pose = transform_matrix[image_filename]

        img_width = cameras[camera_id[image_filename]]["w"]
        img_height = cameras[camera_id[image_filename]]["h"]
        intrinsics = intrinsics_matrix[camera_id[image_filename]]

        save_image_pose_args = {
            "origin_image_path": os.path.join(image_directory, image_filename),
            "image_id": image_count,
            "pose": pose,
            "intrinsics": intrinsics,
        }

        intrinsics_flat = intrinsics.reshape([-1])

        visualized = {
            "K": intrinsics_flat.tolist(),
            "W2C": np.linalg.inv(pose).reshape([-1]).tolist(),
            "img_size": [img_width, img_height],
        }

        if image_count % 8 == 0:
            filename = save_image_pose(spilt="test", **save_image_pose_args)
            save_image_pose(spilt="validation", **save_image_pose_args)
            save_image_pose(spilt="camera_path", **save_image_pose_args)
            visualize_cameras[1][filename] = visualized
        else:
            filename = save_image_pose(spilt="train", **save_image_pose_args)
            visualize_cameras[0][filename] = visualized

        os.makedirs(os.path.join(args.out, "visualize_cameras"), exist_ok=True)
        visualize_train_dir = os.path.join(args.out, "visualize_cameras", "train")
        os.makedirs(visualize_train_dir, exist_ok=True)
        visualize_test_dir = os.path.join(args.out, "visualize_cameras", "test")
        os.makedirs(visualize_test_dir, exist_ok=True)
        visualize_camera_path_dir = os.path.join(args.out, "visualize_cameras", "camera_path")
        os.makedirs(visualize_camera_path_dir, exist_ok=True)

        with open(os.path.join(visualize_train_dir, "cam_dict_norm.json"), "w") as f:
            json.dump(visualize_cameras[0], f, indent=4)
        with open(os.path.join(visualize_test_dir, "cam_dict_norm.json"), "w") as f:
            json.dump(visualize_cameras[1], f, indent=4)
        with open(os.path.join(visualize_camera_path_dir, "cam_dict_norm.json"), "w") as f:
            json.dump(visualize_cameras[1], f, indent=4)

