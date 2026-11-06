"""Shared depth helpers for gen_umi_depth.py and the UMI depth sidecar.

Pipeline:

    cam0 + cam1 (same hand)
      -> stereo-rectify to the cam0-derived virtual K
      -> Fast-FoundationStereo disparity -> rectified cam0 metric depth (336,448)
      -> STATIC same-camera redistort into the cam0 KB4 FISHEYE plane
      -> (336, 448) cam0-fisheye metric depth
      -> spatial_pixel_depth Turbo render

The depth ends up in the SAME raw cam0 fisheye geometry as video.robot*_camera
(the raw cam0 RGB), pixel-aligned with it. Because cam0 is itself one of the
stereo cameras, restoring the fisheye geometry is a pure rotation about the
cam0 optical centre (no baseline / parallax / occlusion), so it reduces to a
precomputed gather remap + per-pixel depth scale (~1 ms) rather than a
per-frame 3D reprojection with z-buffering.

Both hands' stereo run in one batch-2 forward.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.common import (  # noqa: E402
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    PAIR_01,
    common_camera_matrix,
    disparity_to_rectified_depth,
    kb4_pixels_to_rays,
    load_hand_calibrations,
    stereo_rectification,
    undistort_map,
)

# Upstream Fast-FoundationStereo checkout (repo + weights, not in git; see README).
FAST_STEREO_ROOT = ROOT / "Fast-FoundationStereo"
FAST_CHECKPOINT = FAST_STEREO_ROOT / "weights" / "23-36-37" / "model_best_bp2_serialize.pth"
BACKENDS = ("pytorch1", "tensorrt")

# spatial_pixel_depth constants (mm).
_DEPTH_MIN_MM = 10.0
_DEPTH_MAX_MM = 1500.0
_DEPTH_SPAN_MM = _DEPTH_MAX_MM - _DEPTH_MIN_MM  # 1490


def render_spatial_pixel_depth(depth_m: np.ndarray) -> np.ndarray:
    """Absolute-metric-depth Turbo RGB, per spatial_pixel_depth.

    Input: (H, W) float32 metres, invalid = NaN.
    Output: (H, W, 3) uint8 RGB. Invalid depth is black [0, 0, 0].
    Byte cases: 10 mm -> 255, 1500 mm -> 0, invalid -> black.
    """
    valid = np.isfinite(depth_m) & (depth_m > 0)
    depth_mm = np.where(valid, depth_m * 1000.0, _DEPTH_MAX_MM)
    clipped_mm = np.clip(depth_mm, _DEPTH_MIN_MM, _DEPTH_MAX_MM)
    index = np.rint(
        255.0 * (_DEPTH_MAX_MM - clipped_mm) / _DEPTH_SPAN_MM
    ).astype(np.uint8)
    depth_bgr = cv2.applyColorMap(index, cv2.COLORMAP_TURBO)
    depth_rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)
    depth_rgb[~valid] = 0
    return depth_rgb


def valid_fraction(depth_m: np.ndarray) -> float:
    finite = np.isfinite(depth_m) & (depth_m > 0)
    return float(finite.mean())


def load_fast_stereo(backend: str):
    """Load Fast-FoundationStereo (23-36-37) for `backend`; returns (model, InputPadder).

    pytorch1: the pickled upstream nn.Module, called via .forward(iters=8, ...).
    tensorrt: the fixed-shape FP16 engine runner (pipeline.fast_trt), called directly.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    import torch

    sys.path.insert(0, str(FAST_STEREO_ROOT))
    from core.utils.utils import InputPadder

    if backend == "tensorrt":
        from pipeline.fast_trt import FastFoundationStereoTensorRT

        model = FastFoundationStereoTensorRT()
        print(f"[depth_gen] Fast-FoundationStereo TensorRT engine: {model.engine_path}", file=sys.stderr)
    else:
        model = torch.load(FAST_CHECKPOINT, map_location="cpu", weights_only=False)
        model.args.valid_iters = 8
        model.args.max_disp = 192
        model = model.cuda().eval()
        print(f"[depth_gen] Fast-FoundationStereo checkpoint: {FAST_CHECKPOINT}", file=sys.stderr)
    return model, InputPadder


class _HandRig:
    """Per-hand pair01 rectification + static cam0 fisheye redistort maps."""

    def __init__(self, hand: str) -> None:
        self.calibrations = load_hand_calibrations(hand)  # cam0, cam1, cam2
        self.new_K = common_camera_matrix(self.calibrations)  # cam0-derived
        pair = PAIR_01  # first=cam1, second=cam0, ref=cam0
        self.output_right = pair.output_right  # True
        # R_cam1, R_cam0 are the per-camera rectification rotations for pair01.
        self.R_cam1, self.R_cam0, self.baseline_m = stereo_rectification(
            self.calibrations, pair
        )
        self._cam1_map = undistort_map(self.calibrations[1], self.R_cam1, self.new_K)
        self._cam0_map = undistort_map(self.calibrations[0], self.R_cam0, self.new_K)
        self._build_fisheye_maps()

    def _build_fisheye_maps(self) -> None:
        """Precompute the static rectified-cam0 -> cam0-fisheye gather.

        For each output fisheye pixel (uf, vf) at OUTPUT_SIZE using cam0's
        scaled KB4 intrinsics: unproject to a ray in the original cam0 frame,
        rotate into the rectified frame, and pinhole-project with new_K to the
        rectified sample coords. The stored depth (rectified-frame Z) is
        converted to cam0-frame Z by the static scale ray_orig_z / ray_rect_z.
        """
        cam0_scaled = self.calibrations[0].scaled()
        vv, uu = np.indices((OUTPUT_HEIGHT, OUTPUT_WIDTH), dtype=np.float64)
        rays_orig = kb4_pixels_to_rays(uu, vv, cam0_scaled)  # (H,W,3) unit, cam0
        dir_rect = np.einsum("ij,hwj->hwi", self.R_cam0, rays_orig)
        zr = dir_rect[..., 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            ur = self.new_K[0, 0] * dir_rect[..., 0] / zr + self.new_K[0, 2]
            vr = self.new_K[1, 1] * dir_rect[..., 1] / zr + self.new_K[1, 2]
            scale = rays_orig[..., 2] / zr  # rectified-Z -> cam0-fisheye-Z
        valid = (
            np.isfinite(ur) & np.isfinite(vr) & np.isfinite(scale)
            & (zr > 1e-6)
            & (ur >= 0) & (ur <= OUTPUT_WIDTH - 1)
            & (vr >= 0) & (vr <= OUTPUT_HEIGHT - 1)
        )
        self._fish_mapx = np.where(valid, ur, -1).astype(np.float32)
        self._fish_mapy = np.where(valid, vr, -1).astype(np.float32)
        self._fish_scale = np.where(valid, scale, np.nan).astype(np.float32)
        self._fish_valid = valid

    def rectify(self, cam0_rgb, cam1_rgb):
        c0 = cv2.remap(cam0_rgb, self._cam0_map[0], self._cam0_map[1],
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        c1 = cv2.remap(cam1_rgb, self._cam1_map[0], self._cam1_map[1],
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return c0, c1

    def depth_cam0_fisheye(self, disparity_cam0: np.ndarray) -> np.ndarray:
        """Rectified cam0 disparity -> cam0 fisheye metric depth (336,448)."""
        depth_rect = disparity_to_rectified_depth(
            disparity_cam0, self.baseline_m, self.output_right,
            focal=float(self.new_K[0, 0]),
        )
        # NEAREST gather so invalid (NaN) rectified pixels don't bleed into
        # valid ones; then rescale rectified-Z to cam0-fisheye-Z.
        gathered = cv2.remap(
            depth_rect, self._fish_mapx, self._fish_mapy,
            interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            borderValue=float("nan"),
        )
        out = gathered * self._fish_scale
        out[~self._fish_valid] = np.nan
        return out.astype(np.float32, copy=False)


class BatchedDualStereo:
    """Both hands' cam0/cam1 -> cam0-fisheye Turbo depth, one batch-2 forward."""

    def __init__(self, backend: str = "pytorch1") -> None:
        self.backend = backend
        self.left = _HandRig("left")
        self.right = _HandRig("right")
        self._model, self._InputPadder = load_fast_stereo(backend)
        import torch

        self._torch = torch

    def _pair_tensors(self, cam0_rect, cam1_rect):
        """pair01, output_right=True: flip both views and swap them so the
        model predicts the (flipped) cam0 view as its left image."""
        torch = self._torch
        first = torch.from_numpy(cam1_rect).cuda().float().permute(2, 0, 1)[None]
        second = torch.from_numpy(cam0_rect).cuda().float().permute(2, 0, 1)[None]
        first, second = torch.flip(second, dims=[3]), torch.flip(first, dims=[3])
        padder = self._InputPadder(first.shape, divis_by=32, force_square=False)
        first, second = padder.pad(first, second)
        return first, second, padder

    def _disparities(self, lc0, lc1, rc0, rc1):
        """Return (left, right) rectified-cam0 disparities (336,448)."""
        torch = self._torch
        lf, ls, lpad = self._pair_tensors(lc0, lc1)
        rf, rs, rpad = self._pair_tensors(rc0, rc1)

        def _finish(disp, pad):
            # undo the output_right horizontal flip, unpad
            return torch.flip(pad.unpad(disp.float()), dims=[3])[0, 0].cpu().numpy().clip(0, None)

        if self.backend == "tensorrt":
            # The engine is built for batch 1; two forwards.
            with torch.inference_mode():
                return _finish(self._model(lf, ls), lpad), _finish(self._model(rf, rs), rpad)
        first = torch.cat([lf, rf], dim=0)
        second = torch.cat([ls, rs], dim=0)
        with torch.inference_mode(), torch.amp.autocast(
            "cuda", enabled=True, dtype=torch.float16
        ):
            disparity = self._model.forward(
                first, second, iters=8, test_mode=True,
                optimize_build_volume=self.backend,
            )
        return _finish(disparity[0:1], lpad), _finish(disparity[1:2], rpad)

    def process(self, l0, l1, r0, r1):
        """(left/right cam0, cam1 RGB) -> (left_depth_rgb, right_depth_rgb, meta).

        Inputs uint8 RGB (480,640,3). Outputs (336,448,3) uint8 RGB Turbo depth
        in the cam0 fisheye plane.
        """
        t0 = time.perf_counter()
        lc0, lc1 = self.left.rectify(l0, l1)
        rc0, rc1 = self.right.rectify(r0, r1)
        left_disp, right_disp = self._disparities(lc0, lc1, rc0, rc1)
        t1 = time.perf_counter()
        left_depth = self.left.depth_cam0_fisheye(left_disp)
        right_depth = self.right.depth_cam0_fisheye(right_disp)
        t2 = time.perf_counter()
        left_rgb = render_spatial_pixel_depth(left_depth)
        right_rgb = render_spatial_pixel_depth(right_depth)
        t3 = time.perf_counter()
        meta = {
            "stereo_ms": (t1 - t0) * 1000.0,
            "redistort_ms": (t2 - t1) * 1000.0,
            "render_ms": (t3 - t2) * 1000.0,
            "total_ms": (t3 - t0) * 1000.0,
            "left_valid_fraction": valid_fraction(left_depth),
            "right_valid_fraction": valid_fraction(right_depth),
        }
        return left_rgb, right_rgb, meta
