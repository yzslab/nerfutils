import os
import shutil
import numpy as np
import json

import internal.arguments
import internal.exif
import internal.transform_matrix

from internal.utils import get_meta_info


def save_image_pose(origin_image_path: str, spilt: str, image_id: int, pose, intrinsics):
    intrinsics_dir = os.path.join(spilt, "intrinsics")
    pose_dir = os.path.join(spilt, "pose")
    rgb_dir = os.path.join(spilt, "rgb")

    os.makedirs(intrinsics_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)

    np.savetxt(os.path.join(intrinsics_dir, f"{image_id:05d}.txt"), intrinsics, newline=' ')
    np.savetxt(os.path.join(pose_dir, f"{image_id:05d}.txt"), pose, newline=' ')
    image_extension = origin_image_path.rsplit(".")[-1].lower()
    if origin_image_path is not None:
        shutil.copyfile(origin_image_path, os.path.join(rgb_dir, f"{image_id:05d}.{image_extension}"))

    return f"{image_id:05d}.{image_extension}"


def convert_pose(c2w):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    c2w = np.matmul(c2w, flip_yz)
    return c2w


if __name__ == "__main__":
    args = internal.arguments.get_parameters()

    img_exif = internal.exif.parse_exif_values_by_directory(args.path)

    img_width, img_height, focal_length_in_pixel, camera_angle_x, camera_angle_y, cx, cy, origin = get_meta_info(
        img_exif, args.down_sample)

    intrinsics = np.asarray([
        [focal_length_in_pixel, 0, cx, 0],
        [0, focal_length_in_pixel, cy, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)

    transform_matrix = internal.transform_matrix.calculate_transform_matrix(
        img_exif,
        origin,
        args.out + ".camera_scatter.png"
    )

    transform_matrix = internal.transform_matrix.recenter(transform_matrix, args.scene_scale)

    os.makedirs(args.out, exist_ok=True)
    os.chdir(args.out)

    os.makedirs("camera_path", exist_ok=True)
    os.makedirs("train", exist_ok=True)
    os.makedirs("test", exist_ok=True)
    os.makedirs("validation", exist_ok=True)

    image_directory = args.image_dir

    # [train, test, camera_path]
    visualize_cameras = [{}, {}]

    image_count = 0
    for image_filename in transform_matrix:
        image_count += 1

        # convert convention from opengl to opencv
        converted_pose = convert_pose(transform_matrix[image_filename])

        save_image_pose_args = {
            "origin_image_path": os.path.join(image_directory, image_filename),
            "image_id": image_count,
            "pose": converted_pose,
            "intrinsics": intrinsics,
        }

        intrinsics_flat = intrinsics.reshape([-1])

        visualized = {
            "K": intrinsics_flat.tolist(),
            "W2C": np.linalg.inv(converted_pose).reshape([-1]).tolist(),
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


