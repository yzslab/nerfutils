import os
import numpy as np
import yaml
import internal.transform_matrix
from internal.colmap.read_write_model import read_model


def parse_cameras(colmap_cameras: dict, down_scale: int = 1):
    cameras = {}
    # intrinsics_matrix = []

    for camera_id in colmap_cameras:
        # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
        # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
        # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
        camera = colmap_cameras[camera_id]

        camera_params = camera[4]

        w = camera[2] // down_scale
        h = camera[3] // down_scale
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[0])
        k1 = 0
        k2 = 0
        p1 = 0
        p2 = 0
        cx = w / 2
        cy = h / 2

        if camera[1] == "SIMPLE_PINHOLE":
            cx = float(camera_params[1])
            cy = float(camera_params[2])
        elif camera[1] == "PINHOLE":
            fl_y = float(camera_params[1])
            cx = float(camera_params[2])
            cy = float(camera_params[3])
        elif camera[1] == "SIMPLE_RADIAL":
            cx = float(camera_params[1])
            cy = float(camera_params[2])
            k1 = float(camera_params[3])
        elif camera[1] == "RADIAL":
            cx = float(camera_params[1])
            cy = float(camera_params[2])
            k1 = float(camera_params[3])
            k2 = float(camera_params[4])
        elif camera[1] == "OPENCV":
            fl_y = float(camera_params[1])
            cx = float(camera_params[2])
            cy = float(camera_params[3])
            k1 = float(camera_params[4])
            k2 = float(camera_params[5])
            p1 = float(camera_params[6])
            p2 = float(camera_params[7])
        else:
            print("unknown camera model ", camera[1])

        if down_scale > 1:
            fl_x = fl_x / down_scale
            fl_y = fl_y / down_scale
            cx = cx / down_scale
            cy = cy / down_scale
        # fl = 0.5 * w / tan(0.5 * angle_x);
        # angle_x = math.atan(w / (fl_x * 2)) * 2
        # angle_y = math.atan(h / (fl_y * 2)) * 2
        # fovx = angle_x * 180 / math.pi
        # fovy = angle_y * 180 / math.pi

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
            # "angle_x": angle_x,
            # "angle_y": angle_y,
            # field of view in angle
            # "fov_x": fovx,
            # "fov_y": fovy,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
        }
        cameras[camera_id] = camera_details
        # intrinsics_matrix.append(np.asarray([
        #     [fl_x, 0, cx, 0],
        #     [0, fl_y, cy, 0],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1],
        # ]))

    return cameras


def parse_images(images: dict):
    """

    :param images:
    :return: camera-to-world matrix in opencv convention: z ---> scene
    """
    camera_id_dict = {}

    bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])

    w2c_dict = {}
    c2w_dict = {}

    for image_id in images:
        image = images[image_id]
        image_name = image.name
        R = image.qvec2rotmat()
        t = image.tvec.reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(m)

        w2c_dict[image_name] = m
        c2w_dict[image_name] = c2w

        camera_id_dict[image_name] = image.camera_id

    return w2c_dict, c2w_dict, camera_id_dict


def colmap2c2w(model_dir: str, out_dir: str, out_name: str = "transforms.npy"):
    cameras, images, points3D = read_model(model_dir)

    parsed_cameras = parse_cameras(cameras)
    w2c, c2w, camera_id = parse_images(images)

    # fine near and far via points3D
    image_points = {}  # [image_id][point index] = [x, y, z]
    for point_id in points3D:
        point = points3D[point_id]
        for image_id in point.image_ids:
            # create empty list if list not exists
            if image_id not in image_points:
                image_points[image_id] = []
            image_points[image_id].append(np.concatenate([point.xyz, [1]]))

    min_near = 999999999
    max_far = -999999999
    near = {}
    far = {}

    ## convert point [x, y, z] to camera coordinate
    for image_id in image_points:
        point_xyz_in_world = np.stack(image_points[image_id])
        image_name = images[image_id].name
        w2c_mat = w2c[image_name]
        point_xyz_in_camera = np.matmul(point_xyz_in_world, w2c_mat.T)
        image_points[image_id] = point_xyz_in_camera

        z_vals = point_xyz_in_camera[:, 2]
        z_val_min = np.min(z_vals)
        z_val_max = np.max(z_vals)

        near[image_name] = z_val_min
        far[image_name] = z_val_max
        if z_val_min < min_near:
            min_near = z_val_min
        if z_val_max > max_far:
            max_far = z_val_max

    # convert to opengl convention: -z ---> scene
    for i in c2w:
        c2w[i] = internal.transform_matrix.convert_pose(c2w[i])

    images = {}
    image_name_list = list(camera_id.keys())
    image_name_list.sort()
    for image_name in image_name_list:
        images[image_name] = {
            "camera_id": camera_id[image_name],
            "c2w": c2w[image_name],
            "depth": [near[image_name], far[image_name]],
        }

    transforms = {
        "depth": [min_near, max_far],
        "cameras": parsed_cameras,
        # "intrinsics_matrix": intrinsics_matrix,
        "images": images,
    }

    # for image_name in transform_matrix:
    #     transforms["images"][image_name] = {
    #         "camera_id": camera_id[image_name],
    #         # convert to opengl convention: -z ---> scene
    #         "c2w": internal.transform_matrix.convert_pose(
    #             transform_matrix[image_name]
    #         ),
    #     }

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, out_name), transforms, allow_pickle=True)

    transforms["depth"] = [float(transforms["depth"][0]), float(transforms["depth"][1])]
    for k in transforms["images"]:
        transforms["images"][k]["c2w"] = transforms["images"][k]["c2w"].tolist()
        transforms["images"][k]["depth"] = [float(transforms["images"][k]["depth"][0]), float(transforms["images"][k]["depth"][1])]

    with open(os.path.join(out_dir, "{}.yaml".format(out_name)), "w") as f:
        yaml.dump(transforms, f, allow_unicode=True)
