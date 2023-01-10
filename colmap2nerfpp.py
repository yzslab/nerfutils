import os
import json

import argparse

import internal.transform_matrix
from convert2nerfpp import save_image_pose
from internal.colmap.colmap import *
from internal.colmap.read_write_model import *
import internal.utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to colmap model directory")
    parser.add_argument("--image-dir", type=str, default=None, help="path to images directory")
    parser.add_argument("--scene-scale", type=float, default=1)
    parser.add_argument("--split-yaml", type=str)
    parser.add_argument("--down-sample", type=int, default=1)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    cameras, images, points3D = read_model(args.path)

    parsed_cameras = parse_cameras(cameras)
    w2c, c2w, camera_id = parse_images(images)

    intrinsics_matrix = {}
    for idx in parsed_cameras:
        camera = parsed_cameras[idx]
        K = np.identity(4)
        K[0, 0] = camera["fl_x"] // args.down_sample
        K[1, 1] = camera["fl_y"] // args.down_sample
        K[0, 2] = camera["cx"] // args.down_sample
        K[1, 2] = camera["cy"] // args.down_sample
        intrinsics_matrix[idx] = K

    transform_matrix = internal.transform_matrix.recenter(c2w, args.scene_scale)

    transform_matrix = dict(sorted(transform_matrix.items()))

    image_directory = args.image_dir

    visualize_cameras = [{}, {}]

    os.makedirs(args.out, exist_ok=True)
    os.chdir(args.out)

    os.makedirs("camera_path", exist_ok=True)
    os.makedirs("train", exist_ok=True)
    os.makedirs("test", exist_ok=True)
    os.makedirs("validation", exist_ok=True)

    dataset_splitter = internal.utils.get_dataset_spliter(args.split_yaml)

    image_count = 0
    for image_filename in transform_matrix:
        image_count += 1

        pose = transform_matrix[image_filename]

        img_width = parsed_cameras[camera_id[image_filename]]["w"] // args.down_sample
        img_height = parsed_cameras[camera_id[image_filename]]["h"] // args.down_sample
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

        if dataset_splitter(image_filename, image_count) == "test":
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
