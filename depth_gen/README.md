# Depth generation

Metric depth for the UMI wrist cameras, computed from each gripper's stereo fisheye pair with
[Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo). The same code generates
depth for the training data and serves depth at deployment time.

## Installation

Clone Fast-FoundationStereo into this folder, put its `23-36-37` checkpoint at
`Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth`, and create its environment
(Python 3.12) by following its README. The depth server additionally needs `pyzmq` and `msgpack`:

```bash
cd depth_gen
git clone https://github.com/NVlabs/Fast-FoundationStereo.git
pip install pyzmq msgpack
```

`ffmpeg` and `ffprobe` must be installed.

## Data preparation

### Dataset

The input is a UMI recording in the [LeRobot](https://github.com/huggingface/lerobot) v2.1
dataset format, with the two fisheye cameras of each gripper (`cam0`, `cam1`, 640x480). The
directory structure should look like

```
dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ...
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   └── tasks.jsonl
└── videos/
    └── chunk-000/
        ├── observation.gripper.left.cam0/
        │   ├── episode_000000.mp4
        │   └── ...
        ├── observation.gripper.left.cam1/
        ├── observation.gripper.right.cam0/
        └── observation.gripper.right.cam1/
```

### Calibration

Each calibration set is a folder under `data/` with one file per camera,
`{left,right}_cam{0,1,2}_sensor.yaml`, in the format

```yaml
resolution: [640, 480]
camera_model: kb4
intrinsics: [fu, fv, cu, cv]
distortion_coefficients: [k1, k2, k3, k4]
T_BS:            # camera pose in the gripper frame, 4x4 row-major
  rows: 4
  cols: 4
  data: [...]
```

Select a set with `UMI_CALIB_SET=<folder name>`.

## Generate depth

The output is a copy of the dataset with a depth video added for each gripper
(`observation.gripper.{left,right}.depth`):

```bash
cd depth_gen
UMI_CALIB_SET=cam_params python gen_umi_depth.py \
    --dataset-root /path/to/dataset --output-root /path/to/output
```

## Depth server for deployment

```bash
cd depth_gen
UMI_CALIB_SET=cam_params python -m pipeline.depth_sidecar --port 5701 --backend pytorch1
```
