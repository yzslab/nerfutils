import argparse
import open3d as o3d
import numpy as np
import internal.transform_matrix


def get_camera_frustum(img_size, K, C2W, frustum_length=0.5, color=[0., 1., 0.]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],  # frustum origin
                               [-half_w, -half_h, frustum_length],  # top-left image corner
                               [half_w, -half_h, frustum_length],  # top-right image corner
                               [half_w, half_h, frustum_length],  # bottom-right image corner
                               [-half_w, half_h, frustum_length]])  # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i + 1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    return frustum_points, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N * 5, 3))  # 5 vertices per frustum
    merged_lines = np.zeros((N * 8, 2))  # 8 lines per frustum
    merged_colors = np.zeros((N * 8, 3))  # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i * 5:(i + 1) * 5, :] = frustum_points
        merged_lines[i * 8:(i + 1) * 8, :] = frustum_lines + i * 5
        merged_colors[i * 8:(i + 1) * 8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset


def visualize_cameras(transforms, sphere_radius, camera_size=0.1, geometry_file=None, geometry_type='mesh'):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    things_to_draw = [sphere, coord_frame]

    frustums = []
    for img_name in transforms['image_c2w']:
        image_camera_id = transforms["image_camera_id"][img_name]
        image_camera = transforms["cameras"][image_camera_id]
        image_c2w = transforms['image_c2w'][img_name]
        intrinsics_matrix = transforms["intrinsics_matrix"][image_camera_id]
        img_size = [image_camera["w"], image_camera["h"]]
        frustums.append(
            get_camera_frustum(img_size, intrinsics_matrix, image_c2w, frustum_length=camera_size, color=[0, 0, 1]))
    camera_frustums = frustums2lineset(frustums)
    things_to_draw.append(camera_frustums)

    if geometry_file is not None:
        if geometry_type == 'mesh':
            geometry = o3d.io.read_triangle_mesh(geometry_file)
            geometry.compute_vertex_normals()
        elif geometry_type == 'pointcloud':
            geometry = o3d.io.read_point_cloud(geometry_file)
        else:
            raise Exception('Unknown geometry_type: ', geometry_type)

        things_to_draw.append(geometry)

    o3d.visualization.draw_geometries(things_to_draw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--transforms-npy", help="colmap2c2w.py output npy file")
    parser.add_argument("--scale", type=float)
    parser.add_argument("--opencv-c2w", action="store_true")
    parser.add_argument("--camera-size", type=float, default=0.1)
    args = parser.parse_args()

    transforms = np.load(args.transforms_npy, allow_pickle=True).item()

    print(f"{len(transforms['image_c2w'])} camera poses loaded")

    if args.scale is not None:
        for i in transforms['image_c2w']:
            transforms['image_c2w'][i][0:3, 3] *= args.scale
    if args.opencv_c2w is False:
        print("flip z axis")
        for i in transforms['image_c2w']:
            transforms['image_c2w'][i] = internal.transform_matrix.convert_pose(transforms['image_c2w'][i])

    sphere_radius = 1.
    camera_size = args.camera_size
    # colored_camera_dicts = [([0, 1, 0], train_cam_dict),
    #                         # ([0, 0, 1], test_cam_dict),
    #                         # ([1, 1, 0], path_cam_dict)
    #                         ]

    # geometry_file = os.path.join(base_dir, 'mesh_norm.ply')
    geometry_type = 'mesh'

    visualize_cameras(transforms, sphere_radius,
                      camera_size=camera_size, geometry_file=None, geometry_type=geometry_type)
