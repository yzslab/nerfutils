import os
import numpy as np
import math
import yaml
import internal.transform_matrix


def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2
        ]
    ])


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


def parse_cameras_txt(path_to_txt: str, down_scale: int = 1):
    cameras = []
    intrinsics_matrix = []

    with open(path_to_txt, "r") as f:
        for line in f:
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            if line[0] == "#":
                continue
            els = line.split(" ")

            w = float(els[2]) // down_scale
            h = float(els[3]) // down_scale
            fl_x = float(els[4])
            fl_y = float(els[4])
            k1 = 0
            k2 = 0
            p1 = 0
            p2 = 0
            cx = w / 2
            cy = h / 2

            if els[1] == "SIMPLE_PINHOLE":
                cx = float(els[5])
                cy = float(els[6])
            elif els[1] == "PINHOLE":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
            elif els[1] == "RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
            elif els[1] == "OPENCV":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                p1 = float(els[10])
                p2 = float(els[11])
            else:
                print("unknown camera model ", els[1])

            if down_scale > 1:
                fl_x = fl_x / down_scale
                fl_y = fl_y / down_scale
                cx = cx / down_scale
                cy = cy / down_scale
            # fl = 0.5 * w / tan(0.5 * angle_x);
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2
            fovx = angle_x * 180 / math.pi
            fovy = angle_y * 180 / math.pi

            camera_details = {
                # width and height
                "w": int(w),
                "h": int(h),
                # principle point
                "cx": cx,
                "cy": cy,
                # focal length in pixel
                "fl_x": fl_x,
                "fl_y": fl_y,
                # field of view in radian
                "angle_x": angle_x,
                "angle_y": angle_y,
                # field of view in angle
                "fov_x": fovx,
                "fov_y": fovy,
                "k1": k1,
                "k2": k2,
                "p1": p1,
                "p2": p2,
            }
            cameras.append(camera_details)
            intrinsics_matrix.append(np.asarray([
                [fl_x, 0, cx, 0],
                [0, fl_y, cy, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]))

    return cameras, intrinsics_matrix


def parse_images_txt(path_to_txt: str):
    """


    :param path_to_txt:
    :return: camera-to-world matrix in opencv convention: z ---> scene
    """
    transform_matrix = {}
    camera_id = {}

    with open(path_to_txt, "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])

        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue

            i = i + 1
            if i % 2 == 1:
                # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                elems = line.split(" ")
                # name = str(PurePosixPath(Path(IMAGE_FOLDER, elems[9])))
                # why is this requireing a relitive path while using ^
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3, 1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)

                filename = '_'.join(elems[9:])

                transform_matrix[filename] = c2w
                camera_id[filename] = int(elems[8]) - 1

    return transform_matrix, camera_id


def colmap2c2w(text_dir: str, out_dir: str):
    cameras, intrinsics_matrix = parse_cameras_txt(os.path.join(text_dir, "cameras.txt"))
    transform_matrix, camera_id = parse_images_txt(os.path.join(text_dir, "images.txt"))

    # convert to opengl convention: -z ---> scene
    for i in transform_matrix:
        transform_matrix[i] = internal.transform_matrix.convert_pose(transform_matrix[i])

    transforms = {
        "cameras": cameras,
        "intrinsics_matrix": intrinsics_matrix,
        "image_c2w": transform_matrix,
        "image_camera_id": camera_id,
    }

    # for image_name in transform_matrix:
    #     transforms["images"][image_name] = {
    #         "camera_id": camera_id[image_name],
    #         # convert to opengl convention: -z ---> scene
    #         "c2w": internal.transform_matrix.convert_pose(
    #             transform_matrix[image_name]
    #         ),
    #     }

    np.save(os.path.join(out_dir, "transforms.npy"), transforms, allow_pickle=True)

    for k, v in enumerate(transforms["intrinsics_matrix"]):
        transforms["intrinsics_matrix"][k] = v.tolist()
    for image_name in transforms["image_c2w"]:
        transforms["image_c2w"][image_name] = transforms["image_c2w"][image_name].tolist()

    with open(os.path.join(out_dir, "transforms.yaml"), "w") as f:
        yaml.dump(transforms, f, allow_unicode=True)
