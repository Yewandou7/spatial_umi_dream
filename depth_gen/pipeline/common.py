"""Calibration loading and stereo geometry shared by gen_umi_depth.py and the sidecar."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
# Keep model/hub caches inside depth_gen/ so the pipeline is self-contained (its env is separate
# from the policy env).
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault(
    "HUGGINGFACE_HUB_CACHE", str(ROOT / ".cache" / "huggingface" / "hub")
)
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
# Per-gripper factory calibration set, selected with UMI_CALIB_SET=<dir name under data/>.
CALIB_ROOT = ROOT / "data" / os.environ.get("UMI_CALIB_SET", "cam_params")
if not CALIB_ROOT.is_dir():
    raise FileNotFoundError(f"UMI_CALIB_SET points to a missing calibration dir: {CALIB_ROOT}")
print(f"[depth_gen] calibration set: {CALIB_ROOT}", file=sys.stderr)

SOURCE_SIZE = (640, 480)
OUTPUT_SIZE = (448, 336)
OUTPUT_WIDTH, OUTPUT_HEIGHT = OUTPUT_SIZE
SCALE_X = OUTPUT_WIDTH / SOURCE_SIZE[0]
SCALE_Y = OUTPUT_HEIGHT / SOURCE_SIZE[1]
# The unified pinhole intrinsics are not a fixed constant (a fixed f=180 cropped
# the fisheye FOV on all sides); they are auto-estimated per hand from cam0 via
# common_camera_matrix() below (balance=0: the focal is chosen so the tightest
# fisheye direction is fully covered).
MIN_DEPTH_METERS = 0.05
MAX_DEPTH_METERS = 10.0


@dataclass(frozen=True)
class CameraCalibration:
    K: np.ndarray
    D: np.ndarray
    T_BS: np.ndarray

    def scaled(self) -> "CameraCalibration":
        K = self.K.copy()
        K[0] *= SCALE_X
        K[1] *= SCALE_Y
        return CameraCalibration(K=K, D=self.D.copy(), T_BS=self.T_BS.copy())


@dataclass(frozen=True)
class StereoPair:
    name: str
    first_camera: int
    second_camera: int
    reference_camera: int
    output_right: bool


# Physical left-to-right ordering is cam1, cam0, cam2. Pair 01 therefore
# requires right-view inference to produce the requested cam0-aligned result.
PAIR_01 = StereoPair("01", first_camera=1, second_camera=0, reference_camera=0, output_right=True)


def load_calibration(hand: str, camera: int) -> CameraCalibration:
    path = CALIB_ROOT / f"{hand}_cam{camera}_sensor.yaml"
    with path.open() as f:
        data = yaml.safe_load(f)
    fu, fv, cu, cv = map(float, data["intrinsics"])
    K = np.array([[fu, 0.0, cu], [0.0, fv, cv], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.asarray(data["distortion_coefficients"], dtype=np.float64)
    T_BS = np.asarray(data["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
    return CameraCalibration(K=K, D=D, T_BS=T_BS)


def load_hand_calibrations(hand: str) -> dict[int, CameraCalibration]:
    return {camera: load_calibration(hand, camera) for camera in range(3)}


def common_camera_matrix(calibrations: dict[int, CameraCalibration]) -> np.ndarray:
    """Unified pinhole intrinsics for one hand's undistorted space.

    Auto-estimated with balance=0 from cam0's fisheye calibration (OpenCV
    picks the focal from the tightest of the four edge midpoints and shifts
    the principal point to the undistorted center of mass), so the fisheye's
    narrower direction is covered edge-to-edge instead of being cropped.
    """
    cam0 = calibrations[0]
    return cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        cam0.K,
        cam0.D.reshape(-1, 1),
        SOURCE_SIZE,
        np.eye(3),
        balance=0.0,
        new_size=OUTPUT_SIZE,
        fov_scale=1.0,
    )


def transform_camera_to_camera(
    calibrations: dict[int, CameraCalibration], source_camera: int, target_camera: int
) -> np.ndarray:
    """Return T_Starget_Ssource for column-vector points."""
    return (
        np.linalg.inv(calibrations[target_camera].T_BS)
        @ calibrations[source_camera].T_BS
    )


def stereo_rectification(
    calibrations: dict[int, CameraCalibration], pair: StereoPair
) -> tuple[np.ndarray, np.ndarray, float]:
    first = calibrations[pair.first_camera]
    second = calibrations[pair.second_camera]
    T_second_first = transform_camera_to_camera(
        calibrations, pair.first_camera, pair.second_camera
    )
    R_first, R_second, _, _, _ = cv2.fisheye.stereoRectify(
        first.K,
        first.D,
        second.K,
        second.D,
        SOURCE_SIZE,
        T_second_first[:3, :3],
        T_second_first[:3, 3],
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=OUTPUT_SIZE,
        balance=0.0,
        fov_scale=1.0,
    )
    baseline = float(np.linalg.norm(T_second_first[:3, 3]))
    return R_first, R_second, baseline


def undistort_map(
    calibration: CameraCalibration, rotation: np.ndarray, new_K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return cv2.fisheye.initUndistortRectifyMap(
        calibration.K,
        calibration.D,
        rotation,
        new_K,
        OUTPUT_SIZE,
        cv2.CV_32FC1,
    )


def disparity_to_rectified_depth(
    disparity: np.ndarray, baseline: float, output_right: bool, focal: float
) -> np.ndarray:
    disparity = np.asarray(disparity, dtype=np.float32)
    height, width = disparity.shape
    uu = np.broadcast_to(np.arange(width, dtype=np.float32)[None], (height, width))
    support_u = uu + disparity if output_right else uu - disparity
    valid = (
        np.isfinite(disparity)
        & (disparity > 0.05)
        & (support_u >= 0)
        & (support_u <= width - 1)
    )
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[valid] = focal * baseline / disparity[valid]
    depth[(depth < MIN_DEPTH_METERS) | (depth > MAX_DEPTH_METERS)] = np.nan
    return depth


def kb4_pixels_to_rays(
    u: np.ndarray, v: np.ndarray, calibration: CameraCalibration
) -> np.ndarray:
    x_distorted = (u - calibration.K[0, 2]) / calibration.K[0, 0]
    y_distorted = (v - calibration.K[1, 2]) / calibration.K[1, 1]
    theta_distorted = np.sqrt(x_distorted**2 + y_distorted**2)
    k1, k2, k3, k4 = calibration.D

    # KB4 is only invertible over the initial monotonically increasing branch.
    # Some resized image corners lie outside that calibrated branch. A clipped,
    # bracketed Newton solve keeps those pixels from driving the iteration to
    # infinity while marking them invalid below.
    theta_samples = np.linspace(0.0, np.pi, 4097, dtype=np.float64)
    sample_t2 = theta_samples * theta_samples
    sample_derivative = (
        1
        + 3 * k1 * sample_t2
        + 5 * k2 * sample_t2**2
        + 7 * k3 * sample_t2**3
        + 9 * k4 * sample_t2**4
    )
    non_monotonic = np.flatnonzero(sample_derivative <= 0)
    theta_upper = (
        float(theta_samples[non_monotonic[0]])
        if non_monotonic.size
        else float(np.pi)
    )
    upper_t2 = theta_upper * theta_upper
    distorted_upper = theta_upper * (
        1
        + k1 * upper_t2
        + k2 * upper_t2**2
        + k3 * upper_t2**3
        + k4 * upper_t2**4
    )
    valid_domain = theta_distorted <= distorted_upper
    target = np.minimum(theta_distorted, distorted_upper)
    lower = np.zeros_like(target)
    upper = np.full_like(target, theta_upper)
    theta = np.minimum(target, theta_upper)
    for _ in range(12):
        t2 = theta * theta
        polynomial = 1 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4
        derivative = 1 + 3 * k1 * t2 + 5 * k2 * t2**2 + 7 * k3 * t2**3 + 9 * k4 * t2**4
        residual = theta * polynomial - target
        upper = np.where(residual > 0, theta, upper)
        lower = np.where(residual <= 0, theta, lower)
        newton = theta - residual / np.maximum(derivative, 1e-12)
        midpoint = 0.5 * (lower + upper)
        theta = np.where(
            (newton > lower) & (newton < upper) & np.isfinite(newton),
            newton,
            midpoint,
        )
    scale = np.zeros_like(theta)
    nonzero = theta_distorted > 1e-10
    scale[nonzero] = np.sin(theta[nonzero]) / theta_distorted[nonzero]
    rays = np.stack(
        [
            x_distorted * scale,
            y_distorted * scale,
            np.cos(theta),
        ],
        axis=-1,
    )
    rays[~nonzero] = np.array([0.0, 0.0, 1.0])
    rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-12)
    rays[~valid_domain] = np.nan
    return rays
