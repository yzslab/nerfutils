# DJI to NeRF
Converting DJI euler angles and GPS coordinates to NeRF transform matrix

Unfinished yet

## Usage
```bash
python dji2nerf.py \
    PATH_TO_YOUUR_DJI_IMAGE_DIRECTORY \
    --out PATH_TO_OUTPUT_transforms.json \
    --down_sample DOWN_SAMPLE_FACTOR \
    --images DOWN_SAMPLED_IMAGE_DIRECTORY
```
### Example
```bash
python dji2nerf.py \
    ./images \
    --out ./transforms.json \
    --down_sample 4 \
    --images images_4
```