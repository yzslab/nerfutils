import os
import sys

import numpy as np
import json
import internal.arguments
import internal.transform_matrix


def calculate_up_vector(transform_matrix: dict) -> np.ndarray:
    up = np.zeros(3)

    for file in transform_matrix:
        c2w = transform_matrix[file]

        c2w[0:3, 2] *= -1  # flip the y and z axis
        c2w[0:3, 1] *= -1
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
    print("up vector was", up)
    R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    for f in transform_matrix:
        transform_matrix[f] = np.matmul(R, transform_matrix[f])  # rotate up to be the z axis

    return transform_matrix


if __name__ == "__main__":
    def parser_configure(parser):
        parser.add_argument("--aabb-scale", type=int, default=16, help="only work for instant-ngp")
        parser.add_argument("--scene-scale", type=float, default=1, help="only work with --recenter")
        parser.add_argument("--recenter", action="store_true")
        parser.add_argument("--reorient", action="store_true")
        parser.add_argument("--scale", type=float)
        parser.add_argument("--offset", nargs="+", type=float)


    args = internal.arguments.get_parameters(parser_configure)

    transforms_and_cameras = np.load(args.transforms_npy, allow_pickle=True).item()

    camera = transforms_and_cameras["cameras"][0]

    img_width = camera["w"]
    img_height = camera["h"]
    camera_angle_x = camera["angle_x"]
    camera_angle_y = camera["angle_y"]
    fl_x = camera["fl_x"]
    fl_y = camera["fl_y"]
    cx = camera["cx"]
    cy = camera["cy"]

    k1 = camera.get("k1", 0)
    k2 = camera.get("k2", 0)
    p1 = camera.get("p1", 0)
    p2 = camera.get("p2", 0)
    # if args.down_sample != 1:
    #     k1, k2, p1, p2 = 0, 0, 0, 0

    if args.reorient is True:
        up = calculate_up_vector(transforms_and_cameras["image_c2w"])
        transform_matrix = reorient(up, transforms_and_cameras["image_c2w"])
    if args.recenter is True:
        transform_matrix = internal.transform_matrix.recenter(transforms_and_cameras["image_c2w"], args.scene_scale)
    # for f in transform_matrix:
    #     transform_matrix[f][0:3, 3] *= 0.02  # scale to "nerf sized"

    # build frames
    frames = []
    for f in transforms_and_cameras["image_c2w"]:
        frames.append({
            "file_path": os.path.join(args.image_dir, f),
            "transform_matrix": transforms_and_cameras["image_c2w"][f].tolist(),
        })

    # build transform
    transforms = {
        "camera_angle_x": camera_angle_x,
        "camera_angle_y": camera_angle_y,
        "fl_x": fl_x / args.down_sample,
        "fl_y": fl_y / args.down_sample,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "cx": cx / args.down_sample,
        "cy": cy / args.down_sample,
        "w": img_width // args.down_sample,
        "h": img_height // args.down_sample,
        "aabb_scale": args.aabb_scale,
        # "scale": 1.,
        # "offset": [0, 0, 0]
    }

    if args.scale is not None:
        transforms["scale"] = args.scale
    if args.offset is not None:
        transforms["offset"] = args.offset

    transforms["frames"] = frames

    with open(args.out, "w") as f:
        json.dump(transforms, f, indent=4, ensure_ascii=False)
