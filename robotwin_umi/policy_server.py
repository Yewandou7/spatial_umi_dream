#!/usr/bin/env python3
"""RoboTwin-UMI policy server (single-GPU HTTP).

Loads a checkpoint and serves action predictions over HTTP.
Client: policy/SUD in the RoboTwin-UMI checkout.

Usage:
    CUDA_VISIBLE_DEVICES=0 python robotwin_umi/policy_server.py \
        --ckpt checkpoints/robotwin_umi/checkpoint-3000 \
        --port 8765

Protocols:
  POST /contract:
    Return the checkpoint-owned ``depth`` flag, ordered image keys, lossless
    transport, and depth-encoding schema once during client startup.

  POST /reset:
    Drop the session's frame buffer and model streaming state
    (``{"session_id": ...}``; omit the id to reset every session).

  POST /push:
    Real-deploy-style protocol. Append frames to the per-session ring buffer;
    no inference is run.

  POST /infer:
    Real-deploy-style protocol. Append the current frames, sample the latest
    8 contiguous client-downsampled frames, and run inference with the current
    proprio state, without resetting the model stream.

    Request JSON:
        {
          "images": {"observation.images.robot0_camera": "<base64 npy>", ...},
          "state":  "<base64 npy>",    # (16,) or (T, 16) float32 — latest raw RoboTwin quat state
          "task_text": "instruction",
          "session_id": "optional-session"
        }
    Response JSON:
        {
          "actions": "<base64 npy>"    # (n_exec_steps, 16) float32 — quat format
        }

    Every request carries the two wrist RGB streams plus ``robot0_depth`` and
    ``robot1_depth``. The client checks this contract once at startup through
    ``POST /contract``; the frame hot path does not repeat the same schema
    validation on every tick.

State / action layouts (all concatenated flat):
    Quat  (16D, from client / to client) — scalar-first [w, x, y, z]:
        [left_xyz(3), left_quat_wxyz(4), left_gripper(1),
         right_xyz(3), right_quat_wxyz(4), right_gripper(1)]
    Rot6d (20D, internal / model):
        [left_xyz(3), left_rot6d(6), left_gripper(1),
         right_xyz(3), right_rot6d(6), right_gripper(1)]
"""

import json
import logging
import os
import sys
from argparse import ArgumentParser
from base64 import b64encode, b64decode
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from spatial_umi_dream.inference.policy import WorldActionPolicy
from spatial_umi_dream.data.schema import EmbodimentTag
from spatial_umi_dream.data.transform.se3_utils import (
    relative_pose9,
    rot6d_to_rotmat,
    rotmat_to_rot6d,
)
from spatial_umi_dream.data.transform.video import VideoToTensor, VideoCrop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Camera key mapping (client → model)
# ---------------------------------------------------------------------------

IMAGE_KEY_MAP = {
    "observation.images.robot0_camera": "video.robot0_camera",
    "observation.images.robot1_camera": "video.robot1_camera",
    "observation.images.robot0_depth": "video.robot0_depth",
    "observation.images.robot1_depth": "video.robot1_depth",
}

# Rot6d state split (model internal format, 20D total)
STATE_SPLIT_SPEC = [
    ("state.left_pose", 9),
    ("state.left_gripper", 1),
    ("state.right_pose", 9),
    ("state.right_gripper", 1),
]

# Rot6d action concatenation keys (model internal format)
ACTION_CONCAT_KEYS = [
    "action.left_pose",
    "action.left_gripper",
    "action.right_pose",
    "action.right_gripper",
]

LANGUAGE_KEY = "annotation.task"

# RoboTwin client performs the 30 Hz -> 10 Hz temporal downsample before
# sending frames.  The server therefore receives one frame per sampled tick
# and keeps the eight contiguous 10 Hz frames the model is conditioned on.
REAL_DEPLOY_VIDEO_OFFSETS = (-7, -6, -5, -4, -3, -2, -1, 0)
REAL_DEPLOY_BUFFER_LEN = 8
REAL_DEPLOY_PUSH_STRIDE = 3
REAL_DEPLOY_COLD_START_PUSHES = REAL_DEPLOY_BUFFER_LEN - 1
# Model attributes that make up one session's streaming state. The
# server hosts several RoboTwin workers on one model, so these are swapped
# in and out when the active session changes.
MODEL_STREAM_ATTRS = (
    "current_start_frame",
    "language",
    "clip_feas",
    "kv_cache",
    "_history_buffer",
)

# ---------------------------------------------------------------------------
# Quaternion ↔ rot6d conversion
# ---------------------------------------------------------------------------


def _quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    """Convert scalar-first quaternions to scipy's [x, y, z, w] convention."""
    quat = np.asarray(quat, dtype=np.float64)
    return np.concatenate([quat[..., 1:4], quat[..., 0:1]], axis=-1)


def _quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    """Convert scipy's [x, y, z, w] quaternions to scalar-first convention."""
    quat = np.asarray(quat, dtype=np.float64)
    return np.concatenate([quat[..., 3:4], quat[..., 0:3]], axis=-1)


def _quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] → rot6d (first two columns of rotmat)."""
    quat = np.asarray(quat, dtype=np.float64)
    rotmat = Rotation.from_quat(
        _quat_wxyz_to_xyzw(quat).reshape(-1, 4)
    ).as_matrix().reshape(quat.shape[:-1] + (3, 3))
    return rotmat_to_rot6d(rotmat).astype(np.float32)


def _rot6d_to_quat(rot6d: np.ndarray) -> np.ndarray:
    """Convert rot6d (first two columns of rotmat) → quat [w, x, y, z] (scalar-first, SAPIEN/t3d convention)."""
    rot6d = np.asarray(rot6d, dtype=np.float64)
    rotmat = rot6d_to_rotmat(rot6d)
    q_xyzw = Rotation.from_matrix(rotmat.reshape(-1, 3, 3)).as_quat()
    q_wxyz = _quat_xyzw_to_wxyz(q_xyzw).reshape(rot6d.shape[:-1] + (4,))
    return q_wxyz.astype(np.float32)


def _state_quat_to_rot6d(state_quat: np.ndarray) -> np.ndarray:
    """Convert (..., 16) quat state → (..., 20) rot6d state.
    Client sends [w,x,y,z] scalar-first quaternions from SAPIEN."""
    left_xyz = state_quat[..., 0:3]
    left_quat = state_quat[..., 3:7]  # [w, x, y, z]
    left_gripper = state_quat[..., 7:8]
    right_xyz = state_quat[..., 8:11]
    right_quat = state_quat[..., 11:15]  # [w, x, y, z]
    right_gripper = state_quat[..., 15:16]

    left_rot6d = _quat_to_rot6d(left_quat)
    right_rot6d = _quat_to_rot6d(right_quat)

    return np.concatenate(
        [
            left_xyz,
            left_rot6d,
            left_gripper,
            right_xyz,
            right_rot6d,
            right_gripper,
        ],
        axis=-1,
    ).astype(np.float32)


def _apply_relative_state_rot6d(
    state_20d: np.ndarray, relative_state_keys: set[str]
) -> np.ndarray:
    """Convert selected bimanual pose state keys into partner-arm frames.

    The training dataset uses ``inv(T_partner) @ T_this`` for cross-arm
    relative state. Keep grippers unchanged and compute both pose transforms
    from the original absolute poses so the second conversion does not depend
    on the first assignment.
    """
    if not relative_state_keys.intersection({"left_pose", "right_pose"}):
        return state_20d
    if not {"left_pose", "right_pose"}.issubset(relative_state_keys):
        raise ValueError(
            "relative_state requires both left_pose and right_pose for the "
            "RoboTwin bimanual state layout"
        )

    state = np.asarray(state_20d, dtype=np.float32)
    if state.shape[-1] != 20:
        raise ValueError(f"Expected 20D rot6d state, got {state.shape}")
    out = state.copy()
    left_pose = state[..., 0:9].copy()
    right_pose = state[..., 10:19].copy()
    out[..., 0:9] = relative_pose9(left_pose, right_pose)
    out[..., 10:19] = relative_pose9(right_pose, left_pose)
    return out


def _action_rot6d_to_quat(action_rot6d: np.ndarray) -> np.ndarray:
    """Convert (..., 20) rot6d action → (..., 16) quat action (scalar-first, SAPIEN convention)."""
    left_xyz = action_rot6d[..., 0:3]
    left_rot6d = action_rot6d[..., 3:9]
    left_gripper = action_rot6d[..., 9:10]
    right_xyz = action_rot6d[..., 10:13]
    right_rot6d = action_rot6d[..., 13:19]
    right_gripper = action_rot6d[..., 19:20]

    left_quat = _rot6d_to_quat(left_rot6d)
    right_quat = _rot6d_to_quat(right_rot6d)

    return np.concatenate(
        [
            left_xyz,
            left_quat,
            left_gripper,
            right_xyz,
            right_quat,
            right_gripper,
        ],
        axis=-1,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Base64 numpy helpers (matching RoboTwin HTTPClient convention)
# ---------------------------------------------------------------------------


def _npy_to_b64(arr: np.ndarray) -> str:
    buf = BytesIO()
    np.save(buf, arr)
    return b64encode(buf.getvalue()).decode("ascii")


def _b64_to_npy(s: str) -> np.ndarray:
    return np.load(BytesIO(b64decode(s)))


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class RoboTwinPolicyServer:
    def __init__(
        self,
        ckpt_path: str,
        embodiment_tag: str,
        n_exec_steps: int,
        tokenizer_path: str | None = None,
    ):
        self.ckpt_path = ckpt_path
        self.embodiment_tag = embodiment_tag
        self.n_exec_steps = n_exec_steps
        # Session-local count of inference calls; swapped with the model stream.
        self.request_count = 0
        self.frame_buffers: dict[str, dict[str, list[np.ndarray]]] = {}
        self.model_stream_states: dict[str, dict[str, object]] = {}
        self.active_session_id: str | None = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        original_resolutions = self._read_video_resolutions(ckpt_path, embodiment_tag)

        logger.info(
            "Loading WorldActionPolicy from %s (model_embodiment=%s; execution_interface=robotwin)",
            ckpt_path,
            embodiment_tag,
        )
        self.policy = WorldActionPolicy(
            model_path=ckpt_path,
            embodiment_tag=EmbodimentTag(embodiment_tag),
            device=self.device,
            tokenizer_path=tokenizer_path,
        )

        train_cfg = self.policy.train_cfg
        self.image_key_map = dict(IMAGE_KEY_MAP)
        expected_modalities = {
            model_key.removeprefix("video.")
            for model_key in self.image_key_map.values()
        }
        actual_modalities = set(original_resolutions)
        missing_modalities = expected_modalities - actual_modalities
        if missing_modalities:
            raise ValueError(
                "Checkpoint visual contract is missing required video metadata: "
                f"missing={sorted(missing_modalities)}, "
                f"required={sorted(expected_modalities)}, "
                f"metadata={sorted(actual_modalities)}"
            )
        extra_modalities = actual_modalities - expected_modalities
        if extra_modalities:
            logger.info(
                "Ignoring video metadata not consumed by the active checkpoint "
                "transform: %s",
                sorted(extra_modalities),
            )
        original_resolutions = {
            key: resolution
            for key, resolution in original_resolutions.items()
            if key in expected_modalities
        }
        logger.info(
            "Visual contract: client_image_keys=%s",
            list(self.image_key_map),
        )
        self.relative_state = bool(train_cfg.get("relative_state", False))
        self.relative_state_keys = set(
            train_cfg.get("relative_state_keys", []) or []
        )
        logger.info(
            "State contract: relative_state=%s relative_state_keys=%s",
            self.relative_state,
            sorted(self.relative_state_keys),
        )

        self._restore_video_resolutions(original_resolutions)
        self._patch_streaming_query_horizons()

        model = self.policy.model
        # When current_start_frame exceeds local_attn_size, replay buffered
        # frames to rebuild the KV cache rather than restarting the stream.
        model.replay_kv_cache = True
        logger.info("Inference path: KV cache replay enabled")

        logger.info(
            "Policy loaded. action_horizon=%s state_horizon=%s",
            self.policy.model.action_horizon,
            self.state_horizon,
        )

    def _patch_streaming_query_horizons(self) -> None:
        """Patch eval-only horizons for one-block streaming inference.

        Multi-block training uses action_horizon/state_horizon over the whole
        sampled training window. The replay inference kernel predicts one block
        per request, so its action/state register uses one current block:
        num_action_per_block action tokens plus num_state_per_block current
        state tokens.
        """
        model = self.policy.model
        query_action_horizon = int(model.dit.num_action_per_block)
        query_state_horizon = max(1, int(model.dit.num_state_per_block))

        model.action_horizon = query_action_horizon
        model.config.action_horizon = query_action_horizon

        # Only the state/action transforms carry horizons; the video ones don't.
        for transform in self.policy.eval_transform.transforms:
            if hasattr(transform, "action_horizon"):
                transform.action_horizon = query_action_horizon
            if hasattr(transform, "state_horizon"):
                transform.state_horizon = query_state_horizon

        self.action_horizon = query_action_horizon
        self.state_horizon = query_state_horizon
        logger.info(
            "Streaming query horizons patched: "
            "query_action_horizon=%s query_state_horizon=%s",
            self.action_horizon,
            self.state_horizon,
        )

    def _reset_model_stream(self) -> None:
        """Reset model-side streaming/KV state without touching HTTP frame buffers."""
        self.policy.model.reset_stream()
        self.request_count = 0

    def _capture_model_stream(self) -> dict[str, object]:
        """Capture the model's streaming state for the active HTTP session."""
        state: dict[str, object] = {}
        for name in MODEL_STREAM_ATTRS:
            value = getattr(self.policy.model, name)
            if name == "_history_buffer" and value is not None:
                value = list(value)
            state[name] = value
        state["request_count"] = self.request_count
        return state

    def _restore_model_stream(self, session_id: str) -> None:
        """Load a session's streaming state into the shared model."""
        state = self.model_stream_states.get(session_id)
        if state is None:
            self._reset_model_stream()
            return

        for name in MODEL_STREAM_ATTRS:
            value = state.get(name)
            if name == "_history_buffer" and value is not None:
                value = list(value)
            setattr(self.policy.model, name, value)
        self.request_count = int(state.get("request_count", 0))

    def _save_active_model_stream(self) -> None:
        if self.active_session_id is not None:
            self.model_stream_states[self.active_session_id] = (
                self._capture_model_stream()
            )

    def _drop_model_stream(self, session_id: str | None) -> None:
        if session_id is None:
            self.model_stream_states.clear()
            self.active_session_id = None
            self._reset_model_stream()
            return
        self.model_stream_states.pop(session_id, None)
        if self.active_session_id == session_id:
            self.active_session_id = None
            self._reset_model_stream()

    def _activate_model_stream(self, session_id: str) -> None:
        if self.active_session_id == session_id:
            return
        old_session_id = self.active_session_id
        self._save_active_model_stream()
        self._restore_model_stream(session_id)
        self.active_session_id = session_id
        logger.info(
            "Activated model stream for session switch %r -> %r",
            old_session_id,
            session_id,
        )

    def reset_stream(
        self, reason: str = "manual", session_id: str | None = None
    ) -> None:
        """Reset cached streaming state for a new episode."""
        if session_id is None:
            self._drop_model_stream(None)
            self.frame_buffers.clear()
        else:
            self._drop_model_stream(session_id)
            self.frame_buffers.pop(session_id, None)
        logger.info(
            "Reset streaming state (%s, session=%s)",
            reason,
            session_id or "*",
        )

    @staticmethod
    def _num_image_frames(images_dict: dict) -> int:
        """Return the max temporal length across image views in a request."""
        n = 0
        for img in images_dict.values():
            if img.ndim == 3:
                n = max(n, 1)
            elif img.ndim == 4:
                n = max(n, int(img.shape[0]))
        return n

    def _get_frame_buffer(self, session_id: str) -> dict[str, list[np.ndarray]]:
        if session_id not in self.frame_buffers:
            self.frame_buffers[session_id] = {k: [] for k in self.image_key_map}
        return self.frame_buffers[session_id]

    def contract_payload(self) -> dict[str, object]:
        """Return the checkpoint-owned visual contract for one startup handshake."""
        return {
            "depth": True,
            "image_keys": list(self.image_key_map),
            "image_transport": "npy_base64",
            "depth_encoding_schema": "spatial_pixel_depth",
            "rgb_layout": "2x2_rgb_depth",
            "video_sampling": "client_downsampled_contiguous",
            "video_push_stride": REAL_DEPLOY_PUSH_STRIDE,
            "video_window_frames": REAL_DEPLOY_BUFFER_LEN,
            "cold_start_pushes": REAL_DEPLOY_COLD_START_PUSHES,
        }

    def _prepare_image_batch(
        self, images_dict: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Add a temporal axis when needed; startup contract checks own schema."""
        prepared = {}
        for client_key in self.image_key_map:
            image = np.asarray(images_dict[client_key])
            prepared[client_key] = image[np.newaxis, ...] if image.ndim == 3 else image
        return prepared

    @staticmethod
    def _coerce_state_sequence(state_raw: np.ndarray) -> np.ndarray:
        state = np.asarray(state_raw, dtype=np.float32)
        return state[np.newaxis, :] if state.ndim == 1 else state

    def push_frames(
        self, images_dict: dict, session_id: str = "default"
    ) -> dict[str, int]:
        """Append client-format images to a real-deploy-style ring buffer.

        Images may be single frames shaped (H, W, 3) or temporal batches shaped
        (T, H, W, 3). The buffer stores raw client camera keys and keeps only
        the latest 8 client-downsampled ticks.
        """
        prepared = self._prepare_image_batch(images_dict)
        buf = self._get_frame_buffer(session_id)
        for client_key, frames in prepared.items():
            for frame in frames:
                buf[client_key].append(frame.astype(np.uint8, copy=False))
                if len(buf[client_key]) > REAL_DEPLOY_BUFFER_LEN:
                    del buf[client_key][
                        0 : len(buf[client_key]) - REAL_DEPLOY_BUFFER_LEN
                    ]
        lens = {k: len(v) for k, v in buf.items()}
        logger.info("Pushed frames (session=%s) buffer_lens=%s", session_id, lens)
        return lens

    def _sample_real_deploy_frames(
        self, session_id: str = "default"
    ) -> dict[str, np.ndarray]:
        """Sample 8 contiguous, client-downsampled frames.

        If the buffer is not full yet, left-pad by reusing the earliest frame.
        The RoboTwin client normally
        primes seven copies of the initial frame, then /infer appends the eighth.
        """
        buf = self._get_frame_buffer(session_id)
        sampled: dict[str, np.ndarray] = {}
        for client_key in self.image_key_map:
            ring = buf[client_key]
            if not ring:
                raise RuntimeError(
                    f"Frame buffer for {client_key!r} is empty in session "
                    f"{session_id!r}; call /push or include images in /infer first"
                )
            latest_idx = len(ring) - 1
            indices = [max(0, latest_idx + off) for off in REAL_DEPLOY_VIDEO_OFFSETS]
            sampled[client_key] = np.stack([ring[i] for i in indices], axis=0)
        return sampled

    @staticmethod
    def _read_video_resolutions(
        ckpt_path: str, embodiment_tag: str
    ) -> dict[str, tuple[int, int]]:
        """Read per-camera video resolutions from experiment_cfg/metadata.json."""
        metadata_path = os.path.join(ckpt_path, "experiment_cfg", "metadata.json")
        with open(metadata_path) as f:
            meta = json.load(f)
        video = meta[embodiment_tag]["modalities"]["video"]
        return {
            k: tuple(v["resolution"]) for k, v in video.items() if "resolution" in v
        }

    def _restore_video_resolutions(
        self, resolutions: dict[str, tuple[int, int]]
    ) -> None:
        """Restore the dataset video resolutions.

        WorldActionPolicy sets the transform metadata to the DiT target canvas;
        RoboTwin clients send the raw per-view resolution and the VideoCrop /
        canvas assembly inside the transform chain produce the canvas.
        """
        if not resolutions:
            return

        first_res = next(iter(resolutions.values()))
        orig_w, orig_h = first_res

        for transform in self.policy.eval_transform.transforms:
            if isinstance(transform, VideoToTensor):
                orig = transform.original_resolutions
                for key, res in resolutions.items():
                    full_key = f"video.{key}"
                    if full_key in orig:
                        orig[full_key] = res

                video_meta = transform.dataset_metadata.modalities.video
                for key, res in resolutions.items():
                    if key in video_meta:
                        video_meta[key].resolution = res

            elif isinstance(transform, VideoCrop):
                transform.height = orig_h
                transform.width = orig_w
                for key, res in resolutions.items():
                    full_key = f"video.{key}"
                    if full_key in transform.original_resolutions:
                        transform.original_resolutions[full_key] = res
                transform.eval_transform = transform.get_transform("eval")

        logger.info("Restored video resolutions: %s", resolutions)

    def _convert_obs(
        self, images_dict: dict, state_raw: np.ndarray, task_text: str
    ) -> tuple[dict, dict | None]:
        """Convert client-format observation to the model's internal format.

        Client sends the current state as (16,) or (T, 16) in quat format. State
        is an infer-time condition for the current block, not a history buffer;
        if a temporal axis is present we keep only the latest state needed by
        the patched query state horizon.

        When the model is conditioned on cross-arm-relative state, action
        unapply still needs the original world-frame state as its SE(3)
        anchor. Return that separate observation through ``unapply_obs`` so
        the two coordinate-system contracts cannot be conflated.
        """
        obs = {}

        # Images: map client keys → model keys, keep all frames for temporal context.
        for client_key, model_key in self.image_key_map.items():
            image = np.asarray(images_dict[client_key])
            obs[model_key] = image[np.newaxis, ...] if image.ndim == 3 else image

        state_raw = self._coerce_state_sequence(state_raw)
        if state_raw.shape[0] > self.state_horizon:
            state_raw = state_raw[-self.state_horizon :]
        elif state_raw.shape[0] < self.state_horizon:
            pad = np.repeat(
                state_raw[:1], self.state_horizon - state_raw.shape[0], axis=0
            )
            state_raw = np.concatenate([pad, state_raw], axis=0)

        absolute_state_20d = _state_quat_to_rot6d(state_raw)
        model_state_20d = absolute_state_20d
        if self.relative_state:
            model_state_20d = _apply_relative_state_rot6d(
                absolute_state_20d, self.relative_state_keys
            )

        unapply_obs = {} if self.relative_state else None
        offset = 0
        for key, dim in STATE_SPLIT_SPEC:
            obs[key] = model_state_20d[:, offset : offset + dim].astype(np.float32)
            if unapply_obs is not None:
                unapply_obs[key] = absolute_state_20d[
                    :, offset : offset + dim
                ].astype(np.float32)
            offset += dim

        obs[LANGUAGE_KEY] = task_text or ""
        return obs, unapply_obs

    @staticmethod
    def _convert_action(action_dict: dict, n_exec_steps: int) -> np.ndarray:
        """Concatenate model rot6d action keys → (n_exec_steps, 16) quat."""
        parts = []
        for key in ACTION_CONCAT_KEYS:
            arr = np.squeeze(action_dict[key])  # remove batch dim if present
            if arr.ndim == 1:
                arr = arr[:n_exec_steps, np.newaxis]
            else:
                arr = arr[:n_exec_steps]
            parts.append(arr)
        action_20d = np.concatenate(parts, axis=-1).astype(np.float32)  # (steps, 20)
        return _action_rot6d_to_quat(action_20d)  # (steps, 16)

    @torch.no_grad()
    def _run_inference(
        self,
        images_dict: dict,
        state: np.ndarray,
        task_text: str,
        session_id: str,
    ) -> np.ndarray:
        """Run model inference and return actions as (n_exec_steps, 16) float32.

        The model manages its own streaming state: on the first call
        ah.language is None → full init (CLIP, VAE, prefill).  Subsequent
        calls inherit ah.language / current_start_frame → incremental VAE
        path automatically.
        """
        num_image_frames = self._num_image_frames(images_dict)
        obs, unapply_obs = self._convert_obs(images_dict, state, task_text)

        action_dict = self.policy.infer(obs, unapply_obs=unapply_obs)
        actions = self._convert_action(action_dict, self.n_exec_steps)

        self.request_count += 1
        logger.info(
            "Request %d: session=%s frames=%d state_shape=%s actions=%s "
            "current_start_frame=%s",
            self.request_count,
            session_id,
            num_image_frames,
            tuple(state.shape),
            tuple(actions.shape),
            self.policy.model.current_start_frame,
        )
        return actions

    @torch.no_grad()
    def infer_from_buffer(
        self,
        images_dict: dict | None,
        state: np.ndarray,
        task_text: str,
        session_id: str = "default",
    ) -> np.ndarray:
        """Real-deploy-style inference.

        /infer appends the current frame(s), samples the model input from the
        ring buffer, and then runs model inference. It never resets the action
        head; use /reset at episode boundaries.
        """
        self._activate_model_stream(session_id)

        if images_dict:
            self.push_frames(images_dict, session_id=session_id)
        sampled_images = self._sample_real_deploy_frames(session_id=session_id)
        logger.info(
            "Infer from ring buffer (session=%s) offsets=%s sampled_frames=%d state_shape=%s state_horizon=%d",
            session_id,
            REAL_DEPLOY_VIDEO_OFFSETS,
            self._num_image_frames(sampled_images),
            tuple(np.asarray(state).shape),
            self.state_horizon,
        )
        try:
            actions = self._run_inference(
                sampled_images, state, task_text, session_id=session_id
            )
        except Exception:
            # A half-finished step leaves the KV cache inconsistent; drop it so
            # the session restarts from a clean stream.
            self._drop_model_stream(session_id)
            raise
        self.model_stream_states[session_id] = self._capture_model_stream()
        return actions


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


def make_handler(server: RoboTwinPolicyServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logger.info(fmt % args)

        def _send_json(self, status: int, payload: dict) -> None:
            resp = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length))

        def do_POST(self):
            if self.path == "/contract":
                try:
                    self._send_json(200, server.contract_payload())
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    logger.warning("Client disconnected during /contract handshake")
                return

            if self.path == "/reset":
                try:
                    req = self._read_json()
                    session_id = req.get("session_id")
                    server.reset_stream("HTTP /reset", session_id=session_id)
                    try:
                        self._send_json(200, {"ok": True})
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        logger.warning(
                            "Client disconnected before /reset response was sent "
                            "(session=%s)",
                            session_id,
                        )
                except Exception:
                    logger.exception("Reset error")
                    try:
                        self._send_json(500, {"error": str(sys.exc_info()[1])})
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        logger.warning("Client disconnected before /reset error response was sent")
                return

            if self.path == "/push":
                try:
                    req = self._read_json()
                    images_dict = {
                        k: _b64_to_npy(v) for k, v in req.get("images", {}).items()
                    }
                    session_id = str(req.get("session_id", "default"))
                    buffer_lens = server.push_frames(images_dict, session_id=session_id)
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "session_id": session_id,
                            "buffer_lens": buffer_lens,
                        },
                    )
                except Exception:
                    logger.exception("Push error")
                    self._send_json(500, {"error": str(sys.exc_info()[1])})
                return

            if self.path == "/infer":
                try:
                    req = self._read_json()
                    images_dict = {
                        k: _b64_to_npy(v) for k, v in req.get("images", {}).items()
                    }
                    state = _b64_to_npy(req["state"])
                    task_text = req.get("task_text", "")
                    session_id = str(req.get("session_id", "default"))
                    actions = server.infer_from_buffer(
                        images_dict, state, task_text, session_id=session_id
                    )
                    self._send_json(
                        200,
                        {
                            "actions": _npy_to_b64(actions),
                            "session_id": session_id,
                            "video_offsets": list(REAL_DEPLOY_VIDEO_OFFSETS),
                            "buffer_lens": {
                                k: len(v)
                                for k, v in server._get_frame_buffer(session_id).items()
                            },
                        },
                    )
                except Exception:
                    logger.exception("Infer error")
                    self._send_json(500, {"error": str(sys.exc_info()[1])})
                return

            self.send_error(404)

    return Handler


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = ArgumentParser(description="RoboTwin-UMI policy server")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint directory")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument(
        "--embodiment_tag",
        default="robotwin",
        help="Checkpoint/model embodiment tag for metadata and transform selection. "
        "Use umi for UMI pretrained zero-shot checkpoints; default robotwin for "
        "RoboTwin fine-tuned checkpoints.",
    )
    parser.add_argument(
        "--n_exec_steps",
        type=int,
        default=24,
        help="Number of action steps to return per request. "
        "Must match num_action_per_block (24 = 1 block). "
        "Each request predicts one block: client sends 8 frames sampled every "
        "3 action knots (~10Hz) → VAE → 2 latent → 24 actions.",
    )
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help="Override the tokenizer path baked into the checkpoint's conf.yaml "
        "(e.g. google/umt5-xxl when the training machine's path does not exist here).",
    )
    args = parser.parse_args()

    policy_server = RoboTwinPolicyServer(
        ckpt_path=args.ckpt,
        embodiment_tag=args.embodiment_tag,
        n_exec_steps=args.n_exec_steps,
        tokenizer_path=args.tokenizer_path,
    )

    handler = make_handler(policy_server)
    httpd = HTTPServer(("0.0.0.0", args.port), handler)
    logger.info("Serving on port %d", args.port)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
