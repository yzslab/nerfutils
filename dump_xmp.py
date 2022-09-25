import glob
import PIL.Image
import os
import argparse
from internal.exif import parse_exif_values_by_directory
import yaml

import internal.arguments

args = internal.arguments.get_parameters()

img_xmp = parse_exif_values_by_directory(args.path)
save_to = os.path.join(args.path, "xmp.yaml")
with open(save_to, mode="w") as f:
    yaml.dump(img_xmp, f)

print(f"xmp information saved to: {save_to}")
