import argparse
import os
import math

import numpy as np
import json
import internal.arguments
import internal.transform_matrix
import internal.utils


def calculate_up_vector(transform_matrix: dict) -> np.ndarray:
    up = np.zeros(3)

    for file in transform_matrix:
        c2w = transform_matrix[file]

        # c2w[0:3, 2] *= -1  # flip the y and z axis
        # c2w[0:3, 1] *= -1
        c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
        c2w[2, :] *= -1  # flip whole world upside down

        transform_matrix[file] = c2w

        up += c2w[0:3, 1]

    return up


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def reorient(up: np.ndarray, transform_matrix: dict) -> dict:
    up = up / np.linalg.norm(up)
    # print("up vector was", up)
    R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    for f in transform_matrix:
        transform_matrix[f] = np.matmul(R, transform_matrix[f])  # rotate up to be the z axis

    return transform_matrix


def convert2ngp(
        transforms_npy_path: str,
        out_path: str,
        image_dir: str,
        down_sample: int,
        reorient_scene: bool,
        recenter_scene: bool,
        aabb_scale: int,
        scene_scale: float,
        scale: float,
        offset: list,
        no_split: bool = False,
):
    transforms_and_cameras = np.load(transforms_npy_path, allow_pickle=True).item()

    transform_matrix = {}
    for image_name in transforms_and_cameras["images"]:
        transform_matrix[image_name] = transforms_and_cameras["images"][image_name]["c2w"]

    if reorient_scene is True:
        print("reorient...")
        up = calculate_up_vector(transform_matrix)
        transform_matrix = reorient(up, transform_matrix)
    if recenter_scene is True:
        print("recenter...")
        transform_matrix = internal.transform_matrix.recenter(transform_matrix, scene_scale)
    # for f in transform_matrix:
    #     transform_matrix[f][0:3, 3] *= 0.02  # scale to "nerf sized"

    dataset_spliter = internal.utils.get_dataset_spliter(args.split_yaml)
    if no_split is True:
        dataset_spliter = lambda a, b: "train"

    # build frames
    image_filename_list = list(transform_matrix.keys())
    image_filename_list.sort()

    train_frames = []
    test_frames = []
    for idx, f in enumerate(image_filename_list):
        camera_id = transforms_and_cameras["images"][f]["camera_id"]
        camera = transforms_and_cameras["cameras"][camera_id]

        img_width = camera["w"]
        img_height = camera["h"]
        # camera_angle_x = camera["angle_x"]
        # camera_angle_y = camera["angle_y"]
        fl_x = camera["fl_x"]
        fl_y = camera["fl_y"]
        cx = camera["cx"]
        cy = camera["cy"]

        k1 = camera.get("k1", 0)
        k2 = camera.get("k2", 0)
        p1 = camera.get("p1", 0)
        p2 = camera.get("p2", 0)

        camera_angle_x = math.atan(img_width / (fl_x * 2)) * 2
        camera_angle_y = math.atan(img_height / (fl_y * 2)) * 2

        if down_sample != 1:
            k1, k2, p1, p2 = 0, 0, 0, 0

        frame = {
            "camera_angle_x": camera_angle_x,
            "camera_angle_y": camera_angle_y,
            "fl_x": fl_x / down_sample,
            "fl_y": fl_y / down_sample,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            "cx": cx / down_sample,
            "cy": cy / down_sample,
            "w": img_width // down_sample,
            "h": img_height // down_sample,
            "file_path": os.path.join(image_dir, f),
            "transform_matrix": transform_matrix[f].tolist(),
        }

        split = dataset_spliter(f, idx)
        if split == "train":
            train_frames.append(frame)
        else:
            test_frames.append(frame)

    # build transform
    def build_transforms(frames):
        transforms = {
            "aabb_scale": aabb_scale,
        }

        if scale is not None:
            transforms["scale"] = scale
        if offset is not None:
            transforms["offset"] = offset

        transforms["frames"] = frames

        return transforms

    with open("{}_train.json".format(out_path), "w") as f:
        json.dump(build_transforms(train_frames), f, indent=4, ensure_ascii=False)
    with open("{}_test.json".format(out_path), "w") as f:
        json.dump(build_transforms(test_frames), f, indent=4, ensure_ascii=False)


def parser_configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--aabb-scale", type=int, default=16, help="only work for instant-ngp")
    parser.add_argument("--scene-scale", type=float, default=1, help="only work with --recenter")
    parser.add_argument("--recenter", action="store_true")
    parser.add_argument("--reorient", action="store_true")
    parser.add_argument("--scale", type=float)
    parser.add_argument("--offset", nargs="+", type=float)
    parser.add_argument("--no-split", action="store_true")


if __name__ == "__main__":
    args = internal.arguments.get_parameters(parser_configure)

    convert2ngp(
        transforms_npy_path=args.transforms_npy,
        out_path=args.out,
        image_dir=args.image_dir,
        down_sample=args.down_sample,
        reorient_scene=args.reorient,
        recenter_scene=args.recenter,
        aabb_scale=args.aabb_scale,
        scene_scale=args.scene_scale,
        scale=args.scale,
        offset=args.offset,
        no_split=args.no_split,
    )
