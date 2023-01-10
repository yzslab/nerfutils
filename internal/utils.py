import os
import yaml
import numpy as np


def get_meta_info(img_exif, down_sample_factor: int = None):
    first_exif = img_exif[list(img_exif.keys())[0]]

    # width and height
    img_width = first_exif["ImageWidth"]
    img_height = first_exif["ImageLength"]

    if down_sample_factor is not None:
        img_width = img_width // down_sample_factor
        img_height = img_height // down_sample_factor

    # field of view calculation
    fl_in_35mm = first_exif["FocalLengthIn35mmFilm"]
    # 35 mm movie film dimensions: 36mm x 24mm
    camera_angle_x = 2 * np.arctan((36 / 2) / fl_in_35mm)
    camera_angle_y = 2 * np.arctan((24 / 2) / fl_in_35mm)

    focal_length_in_pixel = fl_in_35mm * img_width / 36

    # principle point
    cx = img_width / 2
    cy = img_height / 2

    xmp_values = first_exif["xmp"]
    gps = xmp_values["gps"]
    origin = [gps["latitude"], gps["longitude"], gps["relative_altitude"]]

    return img_width, img_height, focal_length_in_pixel, camera_angle_x, camera_angle_y, cx, cy, origin


def get_dataset_spliter(split_yaml: str, each: int = 8):
    if split_yaml is not None:
        if os.path.exists(split_yaml) is False:
            raise ValueError("{} not found".format(split_yaml))
        with open(split_yaml, "r") as split_file_stream:
            split_list = yaml.safe_load(split_file_stream)

        def split_set(filename: str, idx: int):
            if filename in split_list["test"]:
                return "test"
            return "train"
    elif each > 1:
        def split_set(filename: str, idx: int):
            if idx % each == 0:
                return "test"
            return "train"
    else:
        def split_set(filename: str, idx: int):
            return "train"

    return split_set
