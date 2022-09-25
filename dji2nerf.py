import math
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import pymap3d as pm

import internal.arguments
import internal.exif
from scipy.spatial.transform import Rotation as R


def get_meta_info(img_exif):
    first_exif = img_exif[list(img_exif.keys())[0]]

    # width and height
    img_width = first_exif["ImageWidth"]
    img_height = first_exif["ImageLength"]

    # field of view calculation
    fl_in_35mm = first_exif["FocalLengthIn35mmFilm"]
    # 35 mm movie film dimensions: 36mm x 24mm
    camera_angle_x = 2 * np.arctan((36 / 2) / fl_in_35mm)
    camera_angle_y = 2 * np.arctan((24 / 2) / fl_in_35mm)

    # principle point
    cx = img_width / 2
    cy = img_height / 2

    xmp_values = first_exif["xmp"]
    gps = xmp_values["gps"]
    origin = [gps["latitude"], gps["longitude"], gps["relative_altitude"]]

    return img_width, img_height, camera_angle_x, camera_angle_y, cx, cy, origin


def generate_transform_matrix(pos, rot):
    def Rx(theta):
        return np.matrix([[1, 0, 0],
                          [0, np.cos(theta), -np.sin(theta)],
                          [0, np.sin(theta), np.cos(theta)]])

    def Ry(theta):
        return np.matrix([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])

    def Rz(theta):
        return np.matrix([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])

    # intrinsic rotation
    R = Rz(rot[0]) @ Ry(rot[1]) @ Rx(rot[2]) @ Ry(-np.pi/2) @ Rz(-np.pi/2)

    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = pos

    return transform_matrix

    # flip_mat = np.array([
    #     [1, 0, 0, 0],
    #     [0, -1, 0, 0],
    #     [0, 0, -1, 0],
    #     [0, 0, 0, 1]
    # ])
    #
    # transform_matrix = np.matmul(transform_matrix, flip_mat)
    #
    # return transform_matrix

    # barbershop_mirros_hd_dense:
    # - camera plane is y+z plane, meaning: constant x-values
    # - cameras look to +x

    # # Don't ask me...
    # extra_xf = np.matrix([
    #     [-1, 0, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 0, 1]])
    # # NerF will cycle forward, so lets cycle backward.
    # shift_coords = np.matrix([
    #     [0, 0, 1, 0],
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 0, 1]])
    # xf = shift_coords @ extra_xf @ xf_pos
    # assert np.abs(np.linalg.det(xf) - 1.0) < 1e-4
    # xf = xf @ xf_rot
    # return xf


def calculate_transform_matrix(img_exif, origin, save_camera_scatter_figure):
    transform_matrix = {}

    camera_x = []
    camera_lat = []
    camera_y = []
    camera_long = []

    for file in img_exif:
        # if os.path.basename(file) not in ["org_0d70401c25955015_1662713474000.jpg",
        #                                   "org_339ce74e4b30f712_1662713702000.jpg",
        #                                   "org_5135a599f9a58a74_1662713576000.jpg"]:
        #     continue
        exif_values = img_exif[file]
        xmp_values = exif_values["xmp"]

        # convert gimbal euler angles to rotation matrix
        gimbal = xmp_values["gimbal"]
        # gimbal_r = R.from_euler('zyx', [gimbal["yaw"], gimbal["pitch"], gimbal["roll"]], degrees=True)
        # gimbal_rotation_matrix = gimbal_r.as_matrix()

        gps = xmp_values["gps"]
        # NED
        # x = gps["latitude"]
        # y = gps["longitude"]
        # z = -gps["relative_altitude"]
        # ENU
        # x = gps["longitude"]
        # y = gps["latitude"]
        # z = gps["relative_altitude"]

        y, x, z = pm.geodetic2enu(
            gps["latitude"], gps["longitude"], gps["relative_altitude"],
            origin[0], origin[1], origin[2]
        )

        print(
            f"{file}: (lat: {gps['latitude']}, long: {gps['longitude']}, alt: {gps['relative_altitude']}) ({x}, {y}, {z})")

        camera_x.append(x)
        camera_lat.append(gps["latitude"])
        camera_y.append(y)
        camera_long.append(gps["longitude"])

        # print(file)
        # print(gimbal_rotation_matrix)
        # xyz = [[-x], [-y], [-z]]
        # # print(xyz)
        # w2c = np.concatenate([gimbal_rotation_matrix, xyz], axis=-1)
        # w2c = np.concatenate([w2c, [[0, 0, 0, 1]]], axis=0)
        # # print(w2c)
        # c2w = np.linalg.inv(w2c)

        transform_matrix[file] = np.asarray(generate_transform_matrix(
            [x, -y, z],
            [
                math.radians(-gimbal["yaw"]),
                math.radians(-gimbal["pitch"]),
                math.radians(-gimbal["roll"])
            ]
        ))
        # transform_matrix[file] = np.identity(4)
        # transform_matrix[file][:3, 3] = [x, y, z]

    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.set_figwidth(15)
    fig.set_figheight(15)
    fig.tight_layout(pad=5.0)
    ax[0].set_title('longitude - latitude')
    ax[0].plot(camera_long, camera_lat, 'ro')
    ax[0].set_xlabel('longitude')
    ax[0].set_ylabel('latitude')

    ax[1].set_title('NED: y - x')
    ax[1].plot(camera_y, camera_x, 'ro')
    ax[1].set_xlabel('y')
    ax[1].set_ylabel('x')
    fig.savefig(save_camera_scatter_figure, dpi=600)

    return transform_matrix


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


def closest_point_2_lines(oa, da, ob,
                          db):  # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


def reorient(up: np.ndarray, transform_matrix: dict) -> dict:
    up = up / np.linalg.norm(up)
    print("up vector was", up)
    R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    for f in transform_matrix:
        transform_matrix[f] = np.matmul(R, transform_matrix[f])  # rotate up to be the z axis

    return transform_matrix


def recenter(transform_matrix: dict) -> dict:
    # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in transform_matrix:
        mf = transform_matrix[f][0:3, :]
        for g in transform_matrix:
            mg = transform_matrix[g][0:3, :]
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.01:
                totp += p * w
                totw += w
    totp /= totw
    print(totp)  # the cameras are looking at totp
    for f in transform_matrix:
        transform_matrix[f][0:3, 3] -= totp

    avglen = 0.
    for f in transform_matrix:
        avglen += np.linalg.norm(transform_matrix[f][0:3, 3])

    nframes = len(transform_matrix)
    avglen /= nframes
    print("avg camera distance from origin", avglen)
    for f in transform_matrix:
        transform_matrix[f][0:3, 3] *= 4 / avglen  # scale to "nerf sized"

    return transform_matrix


if __name__ == "__main__":
    args = internal.arguments.get_parameters()
    img_exif = internal.exif.parse_exif_values_by_directory(args.path)

    img_width, img_height, camera_angle_x, camera_angle_y, cx, cy, origin = get_meta_info(img_exif)
    transform_matrix = calculate_transform_matrix(img_exif, origin, args.out+".camera_scatter.png")
    # up = calculate_up_vector(transform_matrix)
    # transform_matrix = reorient(up, transform_matrix)
    transform_matrix = recenter(transform_matrix)
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
        "aabb_scale": 16,
        "frames": frames,
        "scale": 1.,
        "offset": [0, 0, 0]
    }

    with open(args.out, "w") as f:
        json.dump(transforms, f, indent=4)
