import os
import argparse
import matplotlib.pyplot as plt

import yaml
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--xyz-yaml", required=True, type=str)
parser.add_argument("--block-size", required=True, type=int)
parser.add_argument("--overlap", required=True, type=float)
parser.add_argument("--min-images", default=30)
parser.add_argument("--block-size-enlarge-factor", default=0.1)
parser.add_argument("--out-block-dir", type=str)
parser.add_argument("--plot-only", action="store_true")
parser.add_argument("--image-dir", type=str)
args = parser.parse_args()

global_origin = np.asarray([0., 0., 0.])

with open(args.xyz_yaml, "r") as f:
    image_coordinates = yaml.safe_load(f)

bounding_box_min = np.asarray([999999999., 999999999., 999999999.])
bounding_box_max = np.asarray([-999999999., -999999999., -999999999.])


def update_bbox(bbox_to_update: np.ndarray, bbox_to_compare: np.ndarray, type: int):
    """
    :param bbox_to_update:
    :param bbox_to_compare:
    :param type: 0 - min, 1 - max
    :return:
    """

    bbox_len = range(len(bbox_to_update))
    if type == 0:
        for i in bbox_len:
            # update minimum
            if bbox_to_update[i] > bbox_to_compare[i]:
                bbox_to_update[i] = bbox_to_compare[i]
    elif type == 1:
        for i in bbox_len:
            # update maximum
            if bbox_to_update[i] < bbox_to_compare[i]:
                bbox_to_update[i] = bbox_to_compare[i]
    else:
        raise ValueError(f"unsupported type {type}")


# calculate bounding box
for image_name in image_coordinates:
    coordinate = image_coordinates[image_name]
    update_bbox(bounding_box_min, coordinate, 0)
    update_bbox(bounding_box_max, coordinate, 1)
    # for i in range(3):
    #     # update minimum
    #     if bounding_box_min[i] > coordinate[i]:
    #         bounding_box_min[i] = coordinate[i]
    #     # update maximum
    #     if bounding_box_max[i] < coordinate[i]:
    #         bounding_box_max[i] = coordinate[i]

print(f"bounding box: {bounding_box_min} {bounding_box_max}")
bounding_box_size = np.asarray(bounding_box_max) - np.asarray(bounding_box_min)
block_center_distance = args.block_size - (args.overlap * args.block_size)
print(f"block_center_distance: {block_center_distance}")
# n_blocks = bounding_box_size / block_center_distance
# n_blocks_ceil = np.ceil(n_blocks).astype(int)
# print(n_blocks, n_blocks_ceil)

# calculate block number
# how many block can place on the left side of the origin point (not count origin point block)
n_blocks_negative_side = np.clip(
    np.floor((global_origin - bounding_box_min) / block_center_distance).astype(int),
    0,
    999999999,
)
# right side
n_blocks_positive_side = np.clip(
    np.floor((bounding_box_max - global_origin) / block_center_distance).astype(int),
    0,
    999999999,
)
n_blocks = n_blocks_negative_side + n_blocks_positive_side + np.asarray([1, 1, 1])  # plus an origin point block
print(n_blocks_negative_side, n_blocks_positive_side, n_blocks)

# build block center and bounding box
base_block = 0. - (block_center_distance * n_blocks_negative_side)
base_block = base_block[:-1]  # only use (x, y)
blocks = []
for ny in range(0, n_blocks[1]):
    base_y = base_block[1] + ny * block_center_distance
    # row = []
    for nx in range(0, n_blocks[0]):
        base_x = base_block[0] + nx * block_center_distance
        block_center = np.asarray([base_x, base_y])
        blocks.append({
            "c": block_center,
            "bbox": {
                "min": block_center - args.block_size,
                "max": block_center + args.block_size,
            },
            "images": {},
        })
    # blocks.append(row)


# assign images to blocks
def assign_images_to_blocks(block_list: list):
    for image_name in image_coordinates:
        coordinate = np.asarray(image_coordinates[image_name])[:-1]
        for block in block_list:
            if np.all(coordinate >= block["bbox"]["min"]) and np.all(coordinate <= block["bbox"]["max"]):
                block["images"][image_name] = True
                # print(f"{image_name}:{coordinate.tolist()} assigned to {block['c'].tolist()}")


blocks_to_reassign_images = blocks
reassign_count = 0
while len(blocks_to_reassign_images) > 0 and reassign_count < 6:
    assign_images_to_blocks(blocks_to_reassign_images)
    reassign_count = reassign_count + 1

    # find blocks with images < args.min_images, enlarge it and reassign images
    blocks_to_reassign_images = []
    for block in blocks:
        block_image_count = len(block["images"])
        if block_image_count == 0 or block_image_count >= args.min_images:
            continue

        # enlarge block size
        block_center = block["c"]
        scaled_block_size = ((1. + (reassign_count * args.block_size_enlarge_factor)) * args.block_size)
        block["bbox"] = {
            "min": block_center - scaled_block_size,
            "max": block_center + scaled_block_size,
        }
        print(f"block ({block['c'][0]}, {block['c'][1]}) enlarged to {scaled_block_size}")
        blocks_to_reassign_images.append(block)


# plot scatter
image_x = []
image_y = []
for image_name in image_coordinates:
    coordinate = image_coordinates[image_name]
    image_x.append(coordinate[0])
    image_y.append(coordinate[1])

block_x = []
block_y = []
for block in blocks:
    if len(block["images"]) == 0:
        continue
    block_x.append(block["c"][0])
    block_y.append(block["c"][1])

fig, ax = plt.subplots(nrows=1, ncols=1)
fig.set_figwidth(20)
fig.set_figheight(20)
fig.tight_layout(pad=5.0)
ax.set_title('NED: y - x')
ax.set_xlabel('y - east')
ax.set_ylabel('x - north')
ax.plot(image_y, image_x, 'ro')
ax.plot(block_y, block_x, 'bs')
if args.plot_only is True:
    plt.show()
    exit(0)

for block in blocks:
    block["c"] = block["c"].tolist()
    block["bbox"]["min"] = block["bbox"]["min"].tolist()
    block["bbox"]["max"] = block["bbox"]["max"].tolist()
    print(f"{block['c']} has {len(block['images'])} images")

if args.out_block_dir is not None:
    # build block coordinates
    block_x = range(-n_blocks_negative_side[0], n_blocks_positive_side[0] + 1)
    block_y = range(-n_blocks_negative_side[1], n_blocks_positive_side[1] + 1)

    image_base_dir = args.image_dir
    if image_base_dir is None:
        image_base_dir = os.path.dirname(args.xyz_yaml)

    os.makedirs(args.out_block_dir, exist_ok=True)

    fig.savefig(os.path.join(args.out_block_dir, "blocks.png"), dpi=600)

    block_count = 0
    block_list = []
    for y in block_y:
        for x in block_x:
            block = blocks[block_count]
            print(f"building block ({x}, {y}) - ({block['c'][0]}, {block['c'][1]})")
            block_list.append(f"{x},{y}")

            block_base_path = os.path.join(args.out_block_dir, f"{x},{y}")
            block_images_path = os.path.join(block_base_path, "images")
            os.makedirs(block_images_path, exist_ok=True)

            for image_path in block["images"]:
                os.makedirs(os.path.join(block_images_path, os.path.dirname(image_path)), exist_ok=True)
                src_path = os.path.join(image_base_dir, image_path)
                dst_path = os.path.join(block_images_path, image_path)
                if src_path == dst_path:
                    raise ValueError(f"output path is the same as image path: {dst_path}")
                # delete exists file
                if os.path.exists(dst_path):
                    os.unlink(dst_path)
                # create hard link
                os.link(
                    src_path,
                    dst_path,
                )

            # write block yaml file
            with open(os.path.join(block_base_path, "block.yaml"), "w") as f:
                yaml.dump(block, f, allow_unicode=True)

            block_count = block_count + 1

    with open(os.path.join(args.out_block_dir, "block-list.yaml"), "w") as f:
        yaml.dump(block_list, f, allow_unicode=True)

