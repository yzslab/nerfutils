# NeRF Utils

## Conventions

- Fields of Camera Dict
    - `angle_x`: the field of view of the X axis in radian
    - `angle_y`: ... Y axis
    - `cx`: principle point of the X axis
    - `cy`: ... Y axis
    - `fl_x`: focal length of the X axis in pixel
    - `fl_y`: ... Y axis
    - `w`: width in pixel
    - `h`: height ...
    - `k1`, `k2`, `p1`, `p2`: OpenCV distortion parameters
    - `intrinsics_matirx`: intrinsics matrix
- Fields of transforms.npy
    - `cameras`: a list of `Camera Dict` mentioned above
    - `intrinsics_matrix`: camera intrinsics matrix of `cameras`, use same index as the `cameras` list
    - `image_c2w`: a dictionary of camera-to-world matrix, key by image file path
    - `image_camera_id`: a dictionary of camera id of the images, key by image file path
- Coordinate System
    - GPS to XYZ Coordinate System
        - `X`: north
        - `Y`: east
        - `Z`: up
    - Camera-to-World matrix Coordinate System
        - `X`: north
        - `Y`: west (different from the XYZ from GPS)
        - `Z`: up
    - Camera Coordinate System
        - `X`: right
        - `Y`: down
        - `-Z`: scene (different from the colmap)

## Usage

- Extract GPS and Euler Angles to yaml file
    ```bash
    python dump_exif_to_yaml.py \
      IMAGES_DIRECTORY \
      OUTPUT_YAML_FILE_PATH
    ```
- Convert GPS to xyz (NEU)
    ```bash
    python gps2neu.py \
      --yaml dump_exif_to_yaml.py_OUTPUT_YAML_FILE_PATH \
      --out-plot OUTPUT_PLOT_FILE_PATH \
      --out-yaml OUTPUT_XYZ_YAML_FILE_PATH \
      --origin GPS_LAT_0 GPS_LON_0 GPS_ALT_0 # The GPS coordinate use as xyz (0, 0, 0)
    ```
- Extract cameras from EXIF to yaml file
    ```bash
    python exif2cameras.py \
      --exif-yaml dump_exif_to_yaml.py_OUTPUT_YAML_FILE \
      --out-npy OUTPUT_NPY_FILE
    ```
- Convert DJI GPS & Euler Angles to NeRF camera-to-world matrix
    ```bash
    python dji2c2w.py \
      --exif-yaml dump_exif_to_yaml.py_OUTPUT_YAML \
      --xyz-yaml gps2neu.py_OUTPUT_YAML \
      --out-npy OUTPUT_C2W_NPY_FILE_PATH
    ```
- Combine cameras and camera-to-world matrix to single file (`transforms.npy`)
    ```bash
  python combine_cameras_and_c2w.py \
    --camera-npy exif2cameras.py_OUTPUT_NPY_FILE_PATH \
    --c2w-npy dji2c2w.py_OUTPUT_NPY_FILE_PATH \
    --out-npy transforms.npy
    ```
- Convert `transforms.npy` to specific NeRF dataset format
  ```bash
  # instant-ngp format, only use single camera
  python convert2ngp.py \
      --transforms-npy transforms.npy \
      --out PATH_TO_OUTPUT_transforms.json \
      --down-sample DOWN_SAMPLE_FACTOR \
      --image-dir DOWN_SAMPLED_IMAGE_DIRECTORY
      
  # NeRF Plus Plus format
  python convert2nerfpp.py \
      ...
  ```
    - Example
      ```bash
      python convert2ngp.py \
          --transforms-npy transforms.npy \
          --out ./transforms.json \
          --down_sample 4 \
          --image_dir images_4 \
          --recenter
      ```

- Convert colmap text format database to NeRF Plus Plus format dataset
    ```bash
    python colmap2nerfpp.py \
      PATH_TO_COLMAP_SPARSE_TEXT_DATABASE_DIRECTORY \
      --out OUTPUT_PATH \
      --image-dir IMAGES_DIRECTORY \
      --scene-scale SCENE_SCALE_FACTOR
    ```

- Block NeRF
    1. Extract GPS and Euler Angles to yaml file
    2. Convert GPS to xyz (NEU)
    3. Split to multi blocks according to xyz coordinates
        ```bash
       python split2blocks.py \
         --xyz-yaml gps2neu.py_OUTPUT_YAML_FILE_PATH \
         --out-block-dir BLOCK_OUTPUT_DIR \
         --block-size BLOCK_RADIUS \  # open gps2neu.py output plot to get suitable value
         --overlap OVERLAP_RATIO  # 0.5 is recommend
        ```
    4. Run colmap SfM for all blocks
        ```bash
       python block_sfm.py \
         --block-dir split2blocks.py_BLOCK_OUTPUT_DIRECTORY \
         --vocab-tree-path VOCAB_TREE_FILE_PATH
        ```
    5. Convert colmap SfM to camera-to-world transform matrix
        ```bash
       python block_sfm2c2w.py \
         --block-dir split2blocks.py_BLOCK_OUTPUT_DIRECTORY
        ```
    6. Align scene scale and world coordinate system with GPS
        ```bash
       python align.py \
         --exif-yaml EXIF_YAML_FILE_PATH \
         --xyz-yaml XYZ_YAML_FILE_PATH \
         --block-dir BLOCK_DIR_PATH
        ```
    7. Visualize aligned camera poses
        ```bash
        python visualize_cameras.py \
          --transforms-npy BLOCK_DIR_PATH/merged_transforms.npy \
          --scale SCENE_SCALE_FACTOR # use a small value (like 0.02) if the scene large
        ```