import os
import yaml


def load_block_list(block_dir: str) -> list:
    with open(os.path.join(block_dir, "block-list.yaml"), "r") as f:
        return yaml.safe_load(f)


def load_block_information(block_dir: str, block_id: str) -> dict:
    with open(os.path.join(block_dir, block_id, "block.yaml"), "r") as f:
        return yaml.safe_load(f)
