import argparse
import os.path

import yaml

from internal.colmap.read_write_model import read_model

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, type=str)
parser.add_argument("--model-dir", required=True, type=str)
parser.add_argument("--split-yaml", type=str)
parser.add_argument("--out-tsv", required=True, type=str)
args = parser.parse_args()

cameras, images, points3D = read_model(args.model_dir)

with open(args.out_tsv, "w") as f:
    f.write("filename\tid\tsplit\tdataset\n")

    image_name_list = []
    image_name_to_id = {}

    for image_id in images:
        image = images[image_id]
        image_name_list.append(image.name)
        image_name_to_id[image.name] = image_id

    image_name_list.sort()

    if os.path.exists(args.split_yaml):
        with open(args.split_yaml, "r") as split_file_stream:
            split_list = yaml.safe_load(split_file_stream)


        def split_set(filename: str, idx: int):
            if filename in split_list["test"]:
                return "test"
            return "train"
    else:
        def split_set(filename: str, idx: int):
            if idx % 8 == 0:
                return "test"
            return "train"

    for idx, image_name in enumerate(image_name_list):
        split = split_set(image_name, idx)

        image_id = image_name_to_id[image_name]

        f.write(f"{image_name}\t{image_id}\t{split}\t{args.name}\n")
