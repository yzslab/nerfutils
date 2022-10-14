import os
import numpy as np
import json

import internal.arguments
import internal.exif
import internal.transform_matrix

from internal.utils import get_meta_info


def calculate_up_vector(transform_matrix: dict) -> np.ndarray:
    up = np.zeros(3)

    for file in transform_matrix:
        c2w = transform_matrix[file]

        c2w[0:3, 2] *= -1  # flip the y and z axis
        c2w[0:3, 1] *= -1
        c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
        c2w[2, :] *= -1  # flip whole world upside down

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
    args = internal.arguments.get_parameters()
    img_exif = internal.exif.parse_exif_values_by_directory(args.path)

    img_width, img_height, focal_length_in_pixel, camera_angle_x, camera_angle_y, cx, cy, origin = get_meta_info(img_exif)
    transform_matrix = internal.transform_matrix.calculate_transform_matrix(img_exif, origin,
                                                                            args.out + ".camera_scatter.png")
    # up = calculate_up_vector(transform_matrix)
    # transform_matrix = reorient(up, transform_matrix)
    transform_matrix = internal.transform_matrix.recenter(transform_matrix, args.scene_scale)
    # for f in transform_matrix:
    #     transform_matrix[f][0:3, 3] *= 0.02  # scale to "nerf sized"

    # build frames
    frames = [
        # {
        #     "file_path": "images_4/DJI_0162.JPG",
        #     "transform_matrix": generate_transform_matrix([0, 0, 1], [0, 0, 0]).tolist(),
        # },
        # {
        #     "file_path": "images_4/DJI_0162.JPG",
        #     "transform_matrix": generate_transform_matrix([1, 1, 1], [0, 0, 0]).tolist(),
        # },
    ]
    for f in transform_matrix:
        frames.append({
            "file_path": os.path.join(args.images, os.path.basename(f)),
            "transform_matrix": transform_matrix[f].tolist(),
        })

    # build transform
    transforms = {
        # "camera_angle_x": 1.1274228062553935,
        "camera_angle_x": camera_angle_x,
        # "camera_angle_y": 0.8046679505472203,
        "camera_angle_y": camera_angle_y,
        # "fl_x": 1082.052746592492,
        # "fl_y": 1071.5617003841244,
        # "k1": 0.0158082246418964,
        # "k2": -0.004834364713599845,
        # "p1": -0.001671990931921353,
        # "p2": -0.0021405053213860276,
        # "cx": 680.9970060410507,
        "cx": cx / args.down_sample,
        # "cy": 440.64796009212967,
        "cy": cy / args.down_sample,
        # "w": 1368.0,
        "w": img_width // args.down_sample,
        # "h": 912.0,
        "h": img_height // args.down_sample,
        "aabb_scale": args.aabb_scale,
        "frames": frames,
        # "scale": 1.,
        # "offset": [0, 0, 0]
    }

    with open(args.out, "w") as f:
        json.dump(transforms, f, indent=4)
