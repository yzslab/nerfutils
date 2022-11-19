import os
import sys
import hashlib
import argparse
import shutil


import yaml


def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)


parser = argparse.ArgumentParser()
parser.add_argument("--block-dir")
parser.add_argument("--vocab-tree-path")
args = parser.parse_args()

with open(os.path.join(args.block_dir, "block-list.yaml"), "r") as f:
    block_list = yaml.safe_load(f)

for block in block_list:
    block_dir = os.path.join(args.block_dir, block)

    colmap_sfm_version_file_path = os.path.join(block_dir, "colmap_sfm_version")

    with open(os.path.join(block_dir, "block.yaml"), "rb") as f:
        fbytes = f.read()
        readable_hash = hashlib.sha256(fbytes).hexdigest()

    if os.path.exists(colmap_sfm_version_file_path) is True:
        with open(colmap_sfm_version_file_path, "r") as f:
            current_version = f.readline()
        if current_version == readable_hash:
            print(f"skip block {block}")
            continue

    wd = os.path.join(args.block_dir, block)
    db_path = f"{wd}/colmap.db"

    if os.path.exists(db_path):
        os.remove(db_path)

    do_system(f"colmap feature_extractor "
              f"--database_path {db_path} "
              f"--image_path {wd}/images "
              f"--ImageReader.camera_model OPENCV "
              f"--SiftExtraction.max_num_features 32768")
    do_system(f"colmap vocab_tree_matcher "
              f"--database_path {db_path} "
              f"--VocabTreeMatching.vocab_tree_path {args.vocab_tree_path}")

    sparse_path = f"{wd}/sparse"
    if os.path.exists(sparse_path):
        shutil.rmtree(sparse_path)

    os.makedirs(os.path.join(wd, "sparse"), exist_ok=True)
    do_system(f"colmap mapper "
              f"--database_path {db_path} "
              f"--image_path {wd}/images "
              f"--output_path {sparse_path} ")

    if os.path.exists(f"{sparse_path}_text"):
        shutil.rmtree(f"{sparse_path}_text")

    os.makedirs(os.path.join(wd, "sparse_text"), exist_ok=True)
    do_system(f"colmap model_converter "
              f"--input_path {sparse_path}/0 "
              f"--output_path {sparse_path}_text "
              f"--output_type TXT")

    with open(colmap_sfm_version_file_path, "w") as f:
        f.writelines(f"{readable_hash}")


