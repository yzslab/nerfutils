import os
import argparse
from internal.exif import parse_exif_values_by_directory
import yaml
import PIL.TiffImagePlugin

parser = argparse.ArgumentParser()
parser.add_argument("path", help="path to directory storing images taken by DJI Drone")
parser.add_argument("out", help="path to output yaml file")
args = parser.parse_args()

data = {}
for root, dirs, files in os.walk(args.path):
    img_exif = parse_exif_values_by_directory(root)

    for filename in img_exif:
        img_exif_data = img_exif[filename]
        simplified_filename = filename[len(args.path) + 1:]
        SimplifiedGPSInfo = img_exif_data["SimplifiedGPSInfo"]
        data[simplified_filename] = {
            "ImageWidth": img_exif_data["ImageWidth"],
            "ImageHeight": img_exif_data["ImageHeight"],
            "Model": img_exif_data["Model"],
            # "BitsPerSample": list(img_exif_data["BitsPerSample"]),
            # "FocalLengthIn35mmFilm": img_exif_data["FocalLengthIn35mmFilm"],
            "SimplifiedGPSInfo": SimplifiedGPSInfo,
            "xmp": img_exif_data["xmp"],
        }
        for key in ["FocalLengthIn35mmFilm", "FocalLength"]:
            if key in img_exif_data:
                value = img_exif_data[key]
                if type(value) is PIL.TiffImagePlugin.IFDRational:
                    value = float(value)
                data[simplified_filename][key] = value

save_to = args.out
with open(save_to, mode="w", encoding="utf-8") as f:
    yaml.dump(data, f, allow_unicode=True)

print(f"exif information saved to: {save_to}")
