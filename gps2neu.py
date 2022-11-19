import argparse
import pymap3d as pm
import yaml
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--yaml", required=True, help="path to yaml file")
parser.add_argument("--origin", nargs="+", type=float, required=True)
parser.add_argument("--out-plot", required=True)
parser.add_argument("--out-yaml", required=True)
args = parser.parse_args()

with open(args.yaml, "r") as f:
    data = yaml.safe_load(f)

camera_x = []
camera_y = []

image_coordinates = {}

for i in data:
    exif_data = data[i]
    gps = exif_data["xmp"]["gps"]

    # x: north, y: east, z: up
    y, x, z = pm.geodetic2enu(
        gps["latitude"], gps["longitude"], gps["relative_altitude"],
        *args.origin
    )

    image_coordinates[i] = [float(x), float(y), float(z)]

    camera_x.append(x)
    camera_y.append(y)

fig, ax = plt.subplots(nrows=1, ncols=1)
fig.set_figwidth(20)
fig.set_figheight(20)
fig.tight_layout(pad=5.0)
ax.set_title('NED: y - x')
ax.plot(camera_y, camera_x, 'ro')
ax.set_xlabel('y - east')
ax.set_ylabel('x - north')
fig.savefig(args.out_plot, dpi=600)

with open(args.out_yaml, "w", encoding="utf-8") as f:
    yaml.dump(image_coordinates, f, allow_unicode=True)
