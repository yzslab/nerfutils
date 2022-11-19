import json
import os
import argparse
import math
import sys

import numpy as np
import yaml

import internal.transform_matrix


def convert_pose(c2w):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    c2w = np.matmul(c2w, flip_yz)
    return c2w


def apply_transform_to_images(transform, images: dict):
    for i in images:
        images[i] = np.matmul(transform, images[i])


def apply_transform(transform, block):
    return apply_transform_to_images(transform, block["image_c2w"])


def transform_coordinate_system(base_image_name, block):
    w2c = np.linalg.inv(block["image_c2w"][base_image_name])
    apply_transform(w2c, block)


def align_block(reference_c2w: dict, c2w_to_align: dict):
    reference_images = list(reference_c2w.keys())[:2]

    def image_distance(a, b):
        return np.linalg.norm(a[0:3, 3] - b[0:3, 3])

    # size align
    reference_size = image_distance(reference_c2w[reference_images[0]], reference_c2w[reference_images[1]])
    actual_size = image_distance(c2w_to_align[reference_images[0]], c2w_to_align[reference_images[1]])
    scale_factor = reference_size / actual_size

    for i in c2w_to_align:
        c2w_to_align[i][0:3, 3] *= scale_factor

    # world coordinate system align
    ## transform to image coordinate
    w2c = np.linalg.inv(c2w_to_align[reference_images[0]])
    align_transform_matrix = np.matmul(reference_c2w[reference_images[0]], w2c)

    apply_transform_to_images(align_transform_matrix, c2w_to_align)

    return align_transform_matrix


parser = argparse.ArgumentParser()
parser.add_argument("--exif-yaml", required=True, type=str)
parser.add_argument("--xyz-yaml", required=True, type=str)
parser.add_argument("--block-dir", required=True, type=str)
# parser.add_argument("--out-merged", type=str)
parser.add_argument("--skip-missing-npy", action="store_true")
args = parser.parse_args()

with open(os.path.join(args.block_dir, "block-list.yaml"), "rb") as f:
    block_list = yaml.safe_load(f)

blocks = []
for block_id in block_list:
    base_dir = os.path.join(args.block_dir, block_id)
    transform_path = os.path.join(base_dir, "transforms.npy")
    if os.path.exists(transform_path) is False:
        print(f"{transform_path} not exists", file=sys.stderr)
        if args.skip_missing_npy is True:
            continue
        else:
            sys.exit(-1)
    blocks.append({
        "path": base_dir,
        "id": block_id,
        "transforms": np.load(transform_path, allow_pickle=True).item(),
    })

aligned = blocks[0]["transforms"]
aligned_transforms = aligned
blocks_to_align = blocks[1:]

# first of all: align the first block with GPS
## get c2w from GPS and Euler Angles
with open(args.exif_yaml, "r") as f:
    image_exif = yaml.safe_load(f)
with open(args.xyz_yaml, "r") as f:
    image_xyz = yaml.safe_load(f)
reference_images = list(aligned_transforms["image_c2w"].keys())
reference_c2w = {}
for i in range(0, 2):
    reference_c2w[reference_images[i]] = internal.transform_matrix.generate_transform_matrix(
            [image_xyz[reference_images[i]][0], -image_xyz[reference_images[i]][1], image_xyz[reference_images[i]][2]],
            [
                math.radians(-image_exif[reference_images[i]]["xmp"]["gimbal"]["yaw"]),
                math.radians(-image_exif[reference_images[i]]["xmp"]["gimbal"]["pitch"]),
                math.radians(-image_exif[reference_images[i]]["xmp"]["gimbal"]["roll"])
            ]
        )

## align with GPS
align_transform_matrix = align_block(reference_c2w, aligned_transforms["image_c2w"])
np.save(os.path.join(blocks[0]["path"], "align_transform.npy"), align_transform_matrix, allow_pickle=True)
np.save(os.path.join(blocks[0]["path"], "aligned_transforms.npy"), aligned_transforms, allow_pickle=True)

# # try to use the first image of block as the world coordinate
# block = aligned[0]
# first_image = None
# ## transforms-base.json
# transforms = block["cameras"][0]
# transforms["aabb_scale"] = 8
# transforms["offset"] = [0, 0, 0]
# transforms["scale"] = 0.2
# transforms["frames"] = []
# for image_name in block["images"]:
#     if first_image is None:
#         first_image = image_name
#     transforms["frames"].append({
#         "file_path": os.path.join("images", image_name),
#         # "sharpness": sharpness(os.path.join(args.blocks[0], "images", image_name)),
#         "transform_matrix": convert_pose(block["images"][image_name]["c2w"]).tolist(),
#     })
#
# with open(os.path.join(args.blocks[0], "transforms-base.json"), "w") as f:
#     json.dump(transforms, f, indent=2, ensure_ascii=False)
#
# ## transforms-first_image.json
# transforms["frames"] = []
# first_image_c2w_inverse = np.linalg.inv(block["images"][first_image]["c2w"])
# print(first_image_c2w_inverse)
#
# for image_name in block["images"]:
#     if first_image is None:
#         first_image = image_name
#     transforms["frames"].append({
#         "file_path": os.path.join("images", image_name),
#         # "sharpness": sharpness(os.path.join(args.blocks[0], "images", image_name)),
#         "transform_matrix": convert_pose(np.matmul(first_image_c2w_inverse, block["images"][image_name]["c2w"])).tolist(),
#     })
#
# with open(os.path.join(args.blocks[0], "transforms-first_image.json"), "w") as f:
#     json.dump(transforms, f, indent=2, ensure_ascii=False)


for block in blocks_to_align:
    block_transforms = block["transforms"]
    # count overlap images with aligned
    common_images_list = []
    reference_c2w = {}
    count = 0
    # calculate for single block
    for image_name in block_transforms["image_c2w"]:
        if image_name in aligned_transforms["image_c2w"]:
            common_images_list.append(image_name)
            reference_c2w[image_name] = aligned_transforms["image_c2w"][image_name]
            count = count + 1

    if count < 2:
        raise ValueError(f"image not enough to align")

    align_transform_matrix = align_block(reference_c2w, block_transforms["image_c2w"])

    # # calculate scale
    # ## target
    # xyz1 = aligned_transforms["images"][common_images_list[0]]["c2w"][0:3, 3]
    # xyz2 = aligned_transforms["images"][common_images_list[1]]["c2w"][0:3, 3]
    # target_distance = np.linalg.norm(xyz2 - xyz1)
    # ## current
    # xyz1 = block_transforms["images"][common_images_list[0]]["c2w"][0:3, 3]
    # xyz2 = block_transforms["images"][common_images_list[1]]["c2w"][0:3, 3]
    # current_distance = np.linalg.norm(xyz2 - xyz1)
    #
    # scale_factor = target_distance / current_distance
    # print(f"target distance: {target_distance}, current distance: {current_distance}, scale factor: {scale_factor}")
    #
    # # convert to same scale
    # for i in block_transforms["images"]:
    #     block_transforms["images"][i]["c2w"][0:3, 3] *= scale_factor
    # xyz1 = block_transforms["images"][common_images_list[0]]["c2w"][0:3, 3]
    # xyz2 = block_transforms["images"][common_images_list[1]]["c2w"][0:3, 3]
    # current_distance = np.linalg.norm(xyz2 - xyz1)
    # print(f"current distance: {current_distance}")
    #
    # # calculate coordinate transform matrix
    # ## first: transform the unaligned block coordinate system to the first common image
    # first_image_w2c = np.linalg.inv(block_transforms["images"][common_images_list[0]]["c2w"])
    # ## second: transform to global world coordinate system by the camera-to-world matrix of the common image
    # align_transform_matrix = np.matmul(aligned_transforms["images"][common_images_list[0]]["c2w"], first_image_w2c)
    # apply_transform(align_transform_matrix, block_transforms)

    np.save(os.path.join(block["path"], "align_transform.npy"), align_transform_matrix, allow_pickle=True)
    np.save(os.path.join(block["path"], "aligned_transforms.npy"), block_transforms, allow_pickle=True)

    # print transform result
    # for i in common_images_list:
    #     print(f"{i}:\n"
    #           f"    {aligned_transforms['image_c2w'][i][0:3, 3].reshape(-1).tolist()}\n    {block_transforms['image_c2w'][i][0:3, 3].reshape(-1).tolist()}\n"
    #           f"    {aligned_transforms['image_c2w'][i][0:3, 0:3].reshape(-1).tolist()}\n    {block_transforms['image_c2w'][i][0:3, 0:3].reshape(-1).tolist()}")

    # merge
    ## merge cameras
    camera_id_offset = len(aligned_transforms["cameras"])
    aligned_transforms["cameras"] += block_transforms["cameras"]
    aligned_transforms["intrinsics_matrix"] += block_transforms["intrinsics_matrix"]
    for image_name in block_transforms["image_c2w"]:
        if image_name in aligned_transforms["image_c2w"]:
            continue
        aligned_transforms["image_c2w"][image_name] = block_transforms["image_c2w"][image_name]
        aligned_transforms["image_camera_id"][image_name] = block_transforms["image_camera_id"][image_name] + camera_id_offset


# save merged result
# for camera in aligned_transforms["cameras"]:
#     camera["intrinsics_matrix"] = np.asarray([
#         [camera["fl_x"], 0, camera["cx"], 0],
#         [0, camera["fl_y"], camera["cy"], 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1],
#     ])
#     cameras.append(camera)
# for image_name in aligned_transforms["images"]:
#     image_cameras[image_name] = aligned_transforms["images"][image_name]["camera_id"]
#     c2w[image_name] = aligned_transforms["images"][image_name]["c2w"]

np.save(os.path.join(args.block_dir, "merged_transforms.npy"), aligned_transforms)

# np.save(f"{args.out_merged}_cameras.npy", {
#     "cameras": cameras,
#     "image_cameras": image_cameras,
# })
# np.save(f"{args.out_merged}_c2w.npy", c2w)


# transforms = aligned["cameras"][0]
# transforms["aabb_scale"] = 8
# transforms["offset"] = [0, 0, 0]
# transforms["scale"] = 1
# transforms["frames"] = []
# for image_name in aligned["images"]:
#     transforms["frames"].append({
#         "file_path": os.path.join("images", image_name),
#         # "sharpness": sharpness(os.path.join(args.blocks[0], "images", image_name)),
#         "transform_matrix": convert_pose(aligned["images"][image_name]["c2w"]).tolist(),
#     })
#
# with open(os.path.join("/mnt/x/dataset/JNU/transforms-merged.json"), "w") as f:
#     json.dump(transforms, f, indent=2, ensure_ascii=False)
