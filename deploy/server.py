"""WebSocket policy server for UMI real-robot deployment.

See deploy/reference_client.py for a minimal matching client.

Quick recap of the protocol (dispatched on the `endpoint` field of each
msgpack request frame):

  reset { session_id }
    Drop the session's pixel buffer and reset the action head KV cache.
    Returns: { status: "ok" }

  push  { session_id, every camera key advertised by the handshake }
    Append a frame to the per-session ring buffer; no inference. Use this
    for every camera frame the client captures (typically 30Hz) when no
    new action chunk is needed yet.
    Returns: { buffer_lens: { cam_key: int, ... } }

  infer { session_id, every advertised camera key,
          state.left_pose, state.left_gripper,
          state.right_pose, state.right_gripper, prompt }
    Append the frame, then extract 8 stride-3 frames at offsets
    (-21,-18,-15,-12,-9,-6,-3,0) relative to the latest frame in the
    buffer (left-pad with the earliest frame when the buffer is shorter
    than 22). Run the causal model — action-head KV / clip_feas are
    NOT reset here; they carry over across infer calls within a session so
    `current_start_frame` advances and the model streams the same way as
    in training. The `reset` endpoint is the only place state is wiped.
    The state.* fields are the robot's current proprio at chunk anchor;
    the server uses them inside unapply to compose model-predicted SE(3)
    deltas back into absolute target poses (relative_action checkpoint).
    Returns: dict mapping each action sub-key to a (24, D) float32 array
    of absolute poses (left/right) and gripper positions, plus
    `<pose key>_rel`: the (24, 9) relative pose the model predicted.
"""

import asyncio
import dataclasses
import logging
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import tyro
import websockets.asyncio.server
import websockets.frames
from openpi_client import msgpack_numpy

# Avoid FailOnRecompileLimitHit when serving: the flow scheduler's
# torch.compile'd multistep_uni_p_bh_update recompiles under varying shapes.
# The attribute names differ across torch versions, hence the hasattr checks.
_dynamo = torch._dynamo.config
if hasattr(_dynamo, "cache_size_limit"):
    _dynamo.cache_size_limit = 1000
if hasattr(_dynamo, "recompile_limit"):
    _dynamo.recompile_limit = 800
if hasattr(_dynamo, "accumulated_cache_size_limit"):
    _dynamo.accumulated_cache_size_limit = 1000
if hasattr(_dynamo, "accumulated_recompile_limit"):
    _dynamo.accumulated_recompile_limit = 2000

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_umi_dream.data.schema import EmbodimentTag
from spatial_umi_dream.data.transform.se3_utils import relative_pose9
from spatial_umi_dream.inference.policy import WorldActionPolicy
from spatial_umi_dream.data.canvas import VIDEO_KEYS, validate_camera_frame, validate_video_keys

# Cross-arm partner map for `relative_state=True` checkpoints. Mirrors
# ShardedLeRobotSE3DeltaSingleDataset._RELATIVE_STATE_PARTNER so the
# server-side preprocessing matches the dataset's `get_state` math
# (T_this_in_partner = inv(T_partner) @ T_this).
_RELATIVE_STATE_PARTNER: dict[str, str] = {
    "state.left_pose": "state.right_pose",
    "state.right_pose": "state.left_pose",
}
# Identity rot6d pose, used as the warmup placeholder so the relative
# computation does not see degenerate rot6d=(0,0,0,0,0,0) (Gram-Schmidt
# would produce a singular matrix).
_IDENTITY_POSE9 = np.array(
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server config dataclass shipped to the client on connect.
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class UmiPolicyServerConfig:
    fps: int = 30
    action_horizon: int = 24
    # Offsets from the latest frame for the 8 stride-3 input frames.
    # -21 = oldest, 0 = latest; the action chunk starts at the latest frame. Same stride-3
    # spacing as the training video frames.
    video_offsets: tuple[int, ...] = (-21, -18, -15, -12, -9, -6, -3, 0)
    # [robot0 RGB, robot1 RGB, robot0 depth, robot1 depth], from the checkpoint.
    camera_keys: tuple[str, ...] = VIDEO_KEYS
    # Pure single-frame conditioning: ignore the session ring buffer entirely
    # and build every video window from the LATEST frame only (repeated at
    # all video_offsets). No history, no left-padding path, deterministic.
    single_frame: bool = False
    # State the client must send at every infer call. Parallel tuples:
    # state_keys[i] is shipped as shape (1, state_dims[i]) float64. Required
    # because the checkpoint is relative_action=True — the server composes
    # T_abs = T_state @ T_rel inside unapply, so a wrong (or zero) state
    # yields garbage absolute pose output.
    state_keys: tuple[str, ...] = (
        "state.left_pose",
        "state.left_gripper",
        "state.right_pose",
        "state.right_gripper",
    )
    state_dims: tuple[int, ...] = (9, 1, 9, 1)
    action_keys: tuple[str, ...] = (
        "action.left_pose",
        "action.left_gripper",
        "action.right_pose",
        "action.right_gripper",
    )
    # (height, width) the server expects frames at. Client should resize
    # before sending; the server will also resize defensively.
    image_height: int = 160
    image_width: int = 320
    endpoints: tuple[str, ...] = ("reset", "push", "infer")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DEFAULT_IMAGE_HEIGHT = 160
DEFAULT_IMAGE_WIDTH = 320


def _get_expected_video_resolution(policy: WorldActionPolicy) -> tuple[int, int]:
    """Read (height, width) from the first video transform's metadata."""
    for t in policy.eval_transform.transforms:
        res = getattr(t, "original_resolutions", None)
        if res:
            # original_resolutions values are (width, height).
            w, h = next(iter(res.values()))
            return (int(h), int(w))
    return (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)


def _get_checkpoint_camera_keys(policy: WorldActionPolicy) -> tuple[str, ...]:
    """Read and validate the four-stream camera contract saved by training."""
    if "video" not in policy.modality_configs:
        raise RuntimeError("Checkpoint policy has no UMI video modality config")
    return validate_video_keys(list(policy.modality_configs.video.modality_keys))


# ---------------------------------------------------------------------------
# Policy wrapper: per-session frame buffer + inference.
# ---------------------------------------------------------------------------
class UmiPolicy:
    """Wraps a WorldActionPolicy (UMI) with per-session pixel ring buffers.

    The model is not reentrant — callers must serialize calls externally
    (the websocket server does this implicitly by awaiting one request at
    a time).
    """

    def __init__(self, policy: WorldActionPolicy, config: UmiPolicyServerConfig):
        self._policy = policy
        self._cfg = config
        validate_video_keys(config.camera_keys)
        # session_id -> { camera_key: list[np.ndarray (H, W, 3) uint8] }
        self._buffers: dict[str, dict[str, list[np.ndarray]]] = {}
        # Max frames kept per camera: enough to satisfy the oldest offset.
        self._buffer_cap = abs(min(config.video_offsets)) + 1

        # State register / relative state flags. Read off the checkpoint's
        # saved hydra config (no command-line knob); when relative_state is
        # true the server must mirror the dataset's get_state cross-arm SE(3)
        # math because the dataloader path does not run at inference.
        train_cfg = policy.train_cfg
        self._relative_state: bool = bool(train_cfg.get("relative_state", False))
        self._relative_state_keys: list[str] = list(
            train_cfg.get("relative_state_keys", []) or []
        )
        if self._relative_state:
            logger.info(
                "Checkpoint requests relative_state=True; cross-arm SE(3) "
                "preprocessing enabled for keys %s",
                self._relative_state_keys,
            )
        # Sanity: every key listed in relative_state_keys must have a partner
        # mapping defined here. Catch typos at startup, not at first infer.
        for k in self._relative_state_keys:
            full = f"state.{k}"
            if self._relative_state and full not in _RELATIVE_STATE_PARTNER:
                raise RuntimeError(
                    f"relative_state_keys contains {k!r} but no partner "
                    f"mapping is defined for {full!r}. Add it to "
                    f"_RELATIVE_STATE_PARTNER in server.py."
                )

    def _maybe_apply_relative_state(
        self,
        raw_states: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Return the state dict to feed into the model.

        When `relative_state=False` this is a shallow alias of `raw_states`
        (absolute world-frame pose). When `relative_state=True`, the pose
        sub-keys listed in `self._relative_state_keys` are replaced with
        `T_this_in_partner = inv(T_partner) @ T_this`, matching
        ShardedLeRobotSE3DeltaSingleDataset.get_state on the training side.
        Gripper / other sub-keys pass through unchanged.

        `raw_states` is NOT mutated; callers retain it for the absolute-anchor
        unapply step.
        """
        if not self._relative_state:
            return raw_states
        model_states: dict[str, np.ndarray] = {}
        for sk, value in raw_states.items():
            bare = sk[len("state."):] if sk.startswith("state.") else sk
            if bare not in self._relative_state_keys:
                model_states[sk] = value
                continue
            partner = _RELATIVE_STATE_PARTNER.get(sk)
            assert partner is not None, f"missing partner for {sk}"
            this_pose = np.asarray(value, dtype=np.float64)
            partner_pose = np.asarray(raw_states[partner], dtype=np.float64)
            assert this_pose.shape == partner_pose.shape, (
                f"shape mismatch {sk}={this_pose.shape} vs {partner}={partner_pose.shape}"
            )
            assert this_pose.shape[-1] == 9, (
                f"relative_state expects 9D pose for {sk}, got {this_pose.shape}"
            )
            # relative_pose9(chunk, base) -> inv(T_base) @ T_chunk = T_chunk_in_base.
            # Passing (this, partner) yields T_this_in_partner. Preserve the
            # input dtype on the way out so downstream q99 bypass / tensor
            # conversion sees the same type the parquet path would have given.
            model_states[sk] = relative_pose9(this_pose, partner_pose).astype(
                np.asarray(value).dtype, copy=False
            )
        return model_states

    def _get_buffer(self, session_id: str) -> dict[str, list[np.ndarray]]:
        if session_id not in self._buffers:
            self._buffers[session_id] = {k: [] for k in self._cfg.camera_keys}
        return self._buffers[session_id]

    def _resize_if_needed(self, frame: np.ndarray) -> np.ndarray:
        h, w = self._cfg.image_height, self._cfg.image_width
        if frame.shape[0] == h and frame.shape[1] == w:
            return frame
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

    def _append_frames(self, session_id: str, obs: dict) -> dict[str, int]:
        buf = self._get_buffer(session_id)
        for cam_key in self._cfg.camera_keys:
            frame = obs.get(cam_key)
            if frame is None:
                raise KeyError(
                    f"request is missing required camera key {cam_key!r}; "
                    f"expected ordered keys {self._cfg.camera_keys}"
                )
            frame = validate_camera_frame(frame, camera_key=cam_key)
            frame = self._resize_if_needed(frame)
            buf[cam_key].append(frame)
            if len(buf[cam_key]) > self._buffer_cap:
                buf[cam_key] = buf[cam_key][-self._buffer_cap :]
        return {k: len(buf[k]) for k in self._cfg.camera_keys}

    def push(self, obs: dict) -> dict:
        session_id = str(obs.get("session_id", "default"))
        buffer_lens = self._append_frames(session_id, obs)
        return {"buffer_lens": buffer_lens}

    def _sample_chunk(self, session_id: str) -> dict[str, np.ndarray]:
        """Return { cam_key: (T=8, H, W, 3) uint8 } sampled from the buffer
        using video_offsets relative to the latest frame. Left-pad with the
        earliest frame when the buffer is shorter than |min_offset| + 1.
        """
        buf = self._get_buffer(session_id)
        chunks: dict[str, np.ndarray] = {}
        for cam_key in self._cfg.camera_keys:
            ring = buf[cam_key]
            if not ring:
                raise RuntimeError(
                    f"Buffer for {cam_key} (session={session_id}) is empty; "
                    f"push a frame before calling infer"
                )
            n = len(ring)
            latest_idx = n - 1
            if self._cfg.single_frame:
                indices = [latest_idx] * len(self._cfg.video_offsets)
            else:
                indices = [max(0, latest_idx + off) for off in self._cfg.video_offsets]
            chunks[cam_key] = np.stack([ring[i] for i in indices], axis=0)
        return chunks

    def infer(self, obs: dict) -> dict:
        session_id = str(obs.get("session_id", "default"))
        self._append_frames(session_id, obs)
        chunks = self._sample_chunk(session_id)

        # Build the model obs. State is provided by the client (relative_action
        # checkpoint -> unapply does T_abs = T_state @ T_rel so zero state would
        # yield garbage absolute pose). Each state.X must be shape (1, D) float64.
        model_obs: dict = {}
        for cam_key, video in chunks.items():
            model_obs[cam_key] = video
        # `raw_states` always holds the absolute world-frame proprio shipped by
        # the client. When relative_state=True we hand the model `model_states`
        # (cross-arm relative) but the unapply step still composes
        # T_abs = T_state @ T_rel against the world-frame values in raw_states.
        raw_states: dict[str, np.ndarray] = {}
        for sk, dim in zip(self._cfg.state_keys, self._cfg.state_dims):
            if sk not in obs:
                raise KeyError(
                    f"infer payload missing required state key {sk!r} "
                    f"(expected shape (1, {dim}) or ({dim},) float64)"
                )
            s = np.asarray(obs[sk], dtype=np.float64)
            if s.ndim == 1:
                s = s.reshape(1, -1)
            if s.shape != (1, dim):
                raise ValueError(
                    f"{sk}: expected shape (1, {dim}), got {tuple(s.shape)}"
                )
            raw_states[sk] = s
        model_states = self._maybe_apply_relative_state(raw_states)
        for sk in self._cfg.state_keys:
            model_obs[sk] = model_states[sk]
        prompt = str(obs.get("prompt", ""))
        model_obs["annotation.task"] = prompt

        # Do NOT reset KV / clip_feas here. Keeping them across infer
        # calls in a session lets `current_start_frame` advance and the model
        # stream the same way as in training. The very first infer in a
        # session still hits the first-call branch because `reset` (called at
        # episode boundaries) clears `language` to None. To explicitly reset
        # mid-episode (e.g. on prompt change), the client issues `reset`.
        # When relative_state=True the unapply step needs the world-frame
        # state (raw_states), NOT the relative version placed into model_obs.
        unapply_obs: dict | None = None
        if self._relative_state:
            unapply_obs = {sk: raw_states[sk] for sk in self._cfg.state_keys}
        t0 = time.perf_counter()
        with torch.inference_mode():
            actions = self._policy.infer(model_obs, unapply_obs=unapply_obs)
        logger.info("inference took %.3fs (session=%s)", time.perf_counter() - t0, session_id)

        action_dict: dict = {}
        for k in self._cfg.action_keys:
            if k not in actions:
                continue
            v = np.asarray(actions[k], dtype=np.float32)
            if v.ndim == 1:
                v = v.reshape(-1, 1)
            action_dict[k] = v

        # Also expose the relative (delta) poses alongside the absolute ones.
        # unapply did T_abs = T_state @ T_rel; invert it per pose key to
        # recover the exact relative the model predicted:
        #   T_rel = inv(T_state) @ T_abs   (== relative_pose9(abs, anchor)).
        # Anchor is the world-frame state the client sent. Grippers are
        # scalar and have no SE(3) delta, so only *_pose keys get a _rel.
        for k in list(action_dict.keys()):
            if not k.endswith("_pose"):
                continue
            state_key = k.replace("action.", "state.", 1)
            anchor = raw_states.get(state_key)
            if anchor is None:
                continue
            abs_pose9 = action_dict[k]                       # (H, 9)
            rel = relative_pose9(abs_pose9, np.asarray(anchor).reshape(-1))
            action_dict[k + "_rel"] = rel.astype(np.float32)

        return action_dict

    def reset(self, reset_info: dict) -> dict:
        session_id = str(reset_info.get("session_id", "default"))
        self._buffers.pop(session_id, None)
        self._policy.model.reset_stream()
        return {"status": "ok"}

    def warmup(self, iterations: int) -> None:
        """Pre-run the model `iterations` times with dummy zero frames so the
        first real client request does not pay torch.compile + lazy CUDA
        init costs. Mirrors the infer path exactly (same shapes, same KV
        cache reset semantics) so the compiled code paths are real. Uses
        zero state — output is meaningless but shapes/dtypes match."""
        if iterations <= 0:
            return
        h, w = self._cfg.image_height, self._cfg.image_width
        chunk_len = len(self._cfg.video_offsets)
        dummy_obs: dict = {}
        for cam_key in self._cfg.camera_keys:
            dummy_obs[cam_key] = np.zeros((chunk_len, h, w, 3), dtype=np.uint8)
        # Use identity-pose placeholder for *_pose keys instead of all-zeros so
        # the optional cross-arm relative_state preprocessing does not see a
        # degenerate rot6d=(0,0,0,0,0,0) (Gram-Schmidt would produce a
        # singular rotation). Grippers stay at zero.
        for sk, dim in zip(self._cfg.state_keys, self._cfg.state_dims):
            if sk.endswith("_pose") and dim == _IDENTITY_POSE9.shape[0]:
                dummy_obs[sk] = _IDENTITY_POSE9.copy().reshape(1, dim)
            else:
                dummy_obs[sk] = np.zeros((1, dim), dtype=np.float64)
        dummy_obs["annotation.task"] = "warmup"

        # Same relative_state preprocessing as the real infer path so the
        # warmup actually exercises the code the production path will run.
        if self._relative_state:
            raw_states = {sk: dummy_obs[sk] for sk in self._cfg.state_keys}
            model_states = self._maybe_apply_relative_state(raw_states)
            for sk in self._cfg.state_keys:
                dummy_obs[sk] = model_states[sk]
            unapply_obs = {sk: raw_states[sk] for sk in self._cfg.state_keys}
        else:
            unapply_obs = None

        for i in range(iterations):
            self._policy.model.reset_stream()
            t0 = time.perf_counter()
            with torch.inference_mode():
                self._policy.infer(dummy_obs, unapply_obs=unapply_obs)
            dt = time.perf_counter() - t0
            logger.info("warmup iter %d/%d: %.3fs", i + 1, iterations, dt)
        # Final state reset so the first real session starts clean.
        self._policy.model.reset_stream()


# ---------------------------------------------------------------------------
# WebSocket server with reset/push/infer dispatch.
# ---------------------------------------------------------------------------
class UmiWebsocketPolicyServer:
    """WebSocket server speaking msgpack_numpy.

    On connect: sends a msgpack frame with the UmiPolicyServerConfig fields
    so the client knows expected resolution, video offsets, action keys, etc.

    Subsequent frames: msgpack dict with an `endpoint` field in
    {"reset", "push", "infer"}; the server dispatches to the matching
    UmiPolicy method. Errors are sent back as a text frame followed
    by a connection close.
    """

    def __init__(
        self,
        policy: UmiPolicy,
        server_config: UmiPolicyServerConfig,
        host: str = "0.0.0.0",
        port: int = 18000,
    ):
        self._policy = policy
        self._cfg = server_config
        self._host = host
        self._port = port
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, ws: websockets.asyncio.server.ServerConnection) -> None:
        logger.info("Connection from %s opened", ws.remote_address)
        packer = msgpack_numpy.Packer()

        await ws.send(packer.pack(dataclasses.asdict(self._cfg)))

        while True:
            try:
                req = msgpack_numpy.unpackb(await ws.recv())
                endpoint = req.pop("endpoint", "infer")
                if endpoint == "reset":
                    reply = self._policy.reset(req)
                elif endpoint == "push":
                    reply = self._policy.push(req)
                elif endpoint == "infer":
                    reply = self._policy.infer(req)
                else:
                    raise ValueError(f"Unknown endpoint: {endpoint!r}")
                await ws.send(packer.pack(reply))
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", ws.remote_address)
                break
            except Exception:
                await ws.send(traceback.format_exc())
                await ws.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


# ---------------------------------------------------------------------------
# CLI entrypoint.
# ---------------------------------------------------------------------------
def main(
    model_path: str,
    tokenizer_path: str | None = None,
    port: int = 18000,
    host: str = "0.0.0.0",
    image_height: int | None = None,
    image_width: int | None = None,
    warmup_iters: int = 12,
    warmup_compile: bool = False,
    single_frame: bool = False,
) -> None:
    """Serve a UMI checkpoint over WebSocket.

    Args:
        model_path: Checkpoint directory.
        tokenizer_path: Override the tokenizer path baked into conf.yaml.
        image_height: Frame height the client must send (default: from checkpoint).
        image_width: Frame width the client must send (default: from checkpoint).
        warmup_iters: Dummy inference rounds before serving (0 to skip).
        warmup_compile: Run model.warmup_compile instead of warmup_iters.
        single_frame: Condition every chunk on the latest frame only.
    """
    logging.basicConfig(level=logging.INFO, force=True)

    logger.info("Loading UMI policy from %s", model_path)
    policy = WorldActionPolicy(
        model_path=model_path,
        embodiment_tag=EmbodimentTag.UMI,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tokenizer_path=tokenizer_path,
    )

    if image_height is not None and image_width is not None:
        h, w = image_height, image_width
        logger.info("Using CLI image resolution: %dx%d (HxW)", h, w)
    else:
        h, w = _get_expected_video_resolution(policy)
        logger.info("Using checkpoint image resolution: %dx%d (HxW)", h, w)

    cfg = UmiPolicyServerConfig(
        image_height=h,
        image_width=w,
        camera_keys=_get_checkpoint_camera_keys(policy),
        single_frame=single_frame,
    )
    wrapper = UmiPolicy(policy=policy, config=cfg)
    if single_frame:
        logger.info("SINGLE-FRAME mode: every video window = latest frame x %d (ring buffer ignored)",
                    len(cfg.video_offsets))

    # Two flavors of warmup. warmup_compile is the slower, more thorough one:
    # it runs the replay inference path end-to-end across every distinct
    # KV-cache shape so that all torch.compile recompilations happen up front.
    # warmup_iters is the dummy-zero-frame loop and is much cheaper; use it
    # when you don't need compile coverage.
    if warmup_compile:
        logger.info("Running torch.compile warmup at %dx%d (this may take 1-2 minutes)", h, w)
        policy.model.warmup_compile(height=h, width=w)
        logger.info("Compile warmup complete")
    elif warmup_iters > 0:
        logger.info("Running %d warmup iterations before serving", warmup_iters)
        wrapper.warmup(warmup_iters)
        logger.info("Warmup complete")

    logger.info(
        "Starting UmiWebsocketPolicyServer on %s:%d (UMI, %dx%d, endpoints=%s)",
        host, port, h, w, cfg.endpoints,
    )
    server = UmiWebsocketPolicyServer(
        policy=wrapper,
        server_config=cfg,
        host=host,
        port=port,
    )
    server.serve_forever()


if __name__ == "__main__":
    tyro.cli(main)
