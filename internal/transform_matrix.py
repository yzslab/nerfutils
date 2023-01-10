import matplotlib.pyplot as plt
import math
import numpy as np
import pymap3d as pm


def rx(theta):
    return np.matrix([[1, 0, 0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta), np.cos(theta)]])


def ry(theta):
    return np.matrix([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])


def rz(theta):
    return np.matrix([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])


def convert_pose(c2w):
    """
    convert camera poses between Opencv and Opengl conventions

    :param c2w:
    :return:
    """
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    c2w = np.matmul(c2w, flip_yz)
    return c2w


def euler_angles_to_rotation_matrix(zyx_angles: list):
    """
    Return rotation matrix, -z ---> scene
    :param zyx_angles: in [z, y, x] order
    :return:
    """
    # intrinsic rotation
    rotation_matrix = rz(zyx_angles[0]) @ ry(zyx_angles[1]) @ rx(zyx_angles[2])
    # make camera point to -z
    rotation_matrix = rotation_matrix @ ry(-np.pi / 2) @ rz(-np.pi / 2)

    # transform_matrix = np.eye(4)
    # transform_matrix[:3, :3] = R
    # transform_matrix[:3, 3] = pos

    return rotation_matrix


def rtv_to_c2w(rotation_matrix, xyz_coordinate) -> np.ndarray:
    """
    :param rotation_matrix:
    :param xyz_coordinate: [x, y, z]
    :return:
    """
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = xyz_coordinate

    return transform_matrix


def generate_transform_matrix(xyz_coordinate, zyx_euler_angles):
    """
    -z ---> scene

    :param xyz_coordinate:
    :param zyx_euler_angles:
    :return:
    """
    # intrinsic rotation
    rotation_matrix = euler_angles_to_rotation_matrix(zyx_euler_angles)
    transform_matrix = rtv_to_c2w(rotation_matrix, xyz_coordinate)

    return transform_matrix


def calculate_transform_matrix(img_exif, origin, save_camera_scatter_figure):
    """

    :param img_exif:
    :param origin:
    :param save_camera_scatter_figure:
    :return: transform matrix dictionary key by image name
    """
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
            f"{file}: (lat: {gps['latitude']}, long: {gps['longitude']}, alt: {gps['relative_altitude']}) (x: {x}, y: {y}, z: {z})")

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


def recenter(transform_matrix: dict, scene_scale: float) -> dict:
    # find a central point they are all looking at
    # print("computing center of attention...")
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
    # print(totp)  # the cameras are looking at totp
    for f in transform_matrix:
        transform_matrix[f][0:3, 3] -= totp

    avglen = 0.
    for f in transform_matrix:
        avglen += np.linalg.norm(transform_matrix[f][0:3, 3])

    nframes = len(transform_matrix)
    avglen /= nframes
    print("avg camera distance from origin", avglen)
    print("scene scale factor: {}".format(4 * scene_scale / avglen))
    for f in transform_matrix:
        transform_matrix[f][0:3, 3] *= 4 * scene_scale / avglen  # scale to "nerf sized"

    return transform_matrix


def build_intrinsics_matrix(fl_x, cx, cy, fl_y=None) -> np.ndarray:
    if fl_y is None:
        fl_y = fl_x
    return np.asarray([
        [fl_x, 0, cx, 0],
        [0, fl_y, cy, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
