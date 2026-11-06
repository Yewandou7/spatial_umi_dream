"""Reference client for the UMI policy server.

This file is meant to be READ when writing your real robot client. It is
not the production client — your robot loop will have camera capture, IK,
joint commanding, safety, etc. that are outside the scope of this repo.

What this reference shows:

  1. How to open a websocket connection and read the UmiPolicyServerConfig
     the server sends on connect. The config tells you image resolution,
     fps, action_horizon, video_offsets, camera_keys, and action_keys —
     so client and server agree on shapes without hardcoding.

  2. When to send each endpoint (the "decision rule" in the demo loop).

  3. Exactly what payload to put on the wire (numpy dtype/shape/key names).

  4. What format the response is and how to consume it.

================================================================================
WIRE PROTOCOL SUMMARY
================================================================================

On connect:
    server -> client:  msgpack-encoded dict of UmiPolicyServerConfig.
    Read it once, use its fields to size your frames.

Per request:  msgpack-encoded dict, must include `endpoint` field.

  endpoint = "reset"
    payload:
        endpoint:    "reset"
        session_id:  str
    server response:
        {"status": "ok"}
    when to call:
        - at the start of every new episode
        - if the prompt changes
        - on any unrecoverable error (then reopen the websocket)

  endpoint = "push"
    payload:
        endpoint:    "push"
        session_id:  str  (same across one episode)
        every key in camera_keys:
            np.uint8 array, shape (H, W, 3), RGB or depth visualization
    server response:
        {"buffer_lens": {camera_key: int, ...}}
    when to call:
        - for every frame your camera captures (e.g. 30 Hz tick)
        - whenever you do NOT need a new action chunk yet
        - keeps the server's per-session ring buffer up-to-date so the
          next infer call sees a fresh 0.7 s window of past pixels

  endpoint = "infer"
    payload:
        endpoint:    "infer"
        session_id:  str
        every key in camera_keys:
            np.uint8 array, shape (H, W, 3), RGB or depth visualization
        state.left_pose:        np.float64 array, shape (9,) or (1, 9)
                                — current left-arm absolute pose as
                                (xyz(3), rot6d(6))
        state.left_gripper:     np.float64 array, shape (1,) or (1, 1)
                                — current left gripper opening
        state.right_pose:       np.float64 array, shape (9,) or (1, 9)
                                — current right-arm absolute pose
        state.right_gripper:    np.float64 array, shape (1,) or (1, 1)
                                — current right gripper opening
        prompt:                 str  (natural-language task instruction)
    server response (each value is float32):
        {"action.left_pose":     np.ndarray shape (24, 9),
         "action.left_gripper":  np.ndarray shape (24, 1),
         "action.right_pose":    np.ndarray shape (24, 9),
         "action.right_gripper": np.ndarray shape (24, 1),
         "action.left_pose_rel":  np.ndarray shape (24, 9),
         "action.right_pose_rel": np.ndarray shape (24, 9)}
        The *_rel entries are the relative SE(3) deltas the model predicted
        (T_rel = inv(T_state) @ T_abs), for logging / inspection only.
        Poses are ABSOLUTE (already composed with the state you passed in)
        and ready to feed straight to IK. The checkpoint was trained with
        relative_action=True so the server runs T_abs = T_state @ T_rel
        inside unapply; sending the wrong state yields the wrong absolute
        pose, sending zero state yields garbage (rot6d=0 is degenerate).
    when to call:
        - once per action chunk: after the previous 24-step chunk has been
          fully executed. Mid-chunk re-planning is NOT supported because
          the (state, video) it would feed back was never in the training
          distribution (training only saw state(t) + video[t..t+21] paired
          with the full 24-step horizon, not mid-trajectory anchors).
        - the call also pushes the current frame, so it is a strict
          superset of "push" — never call push and infer for the same frame
        - on cold start, push for ~22 frames (~0.7 s @30Hz) before calling
          infer, otherwise the first chunk's pixel window is left-padded
          with stale frames
        - state is read AT the moment of inference. The 24 returned poses
          are all anchored on that one state snapshot, so sample state at
          the same instant you sample the camera frame you send

Action chunk semantics:
    Each row of action.left_pose is the left arm's target pose at a step
    in the future, encoded as (xyz, rot6d) = (3 + 6) = 9 floats. The
    right_pose mirrors. The grippers are scalar (1 float per row).
    Stride between rows is 1 / fps = 1 / 30 s. Run the IK in the client
    and command the robot at the same 30 Hz cadence. Stride is implicit:
    the i-th row is the action at t0 + i / fps.

Frame format details:
    - dtype must be np.uint8. The server will reject other dtypes after
      cast attempts and likely produce garbage if you pre-normalize.
    - layout is RGB (matches cv2.cvtColor(BGR2RGB) output).
    - shape must be exactly (image_height, image_width, 3). The server
      will resize if you send a different resolution. Send every key in
      camera_keys: the two RGB streams followed by their two synchronized
      depth-visualization streams.
    - msgpack_numpy handles the array zerocopy; no manual encode needed.

================================================================================
"""

import argparse
import logging
import time
import uuid
from typing import Dict

import numpy as np
import websockets.sync.client
from openpi_client import msgpack_numpy

logger = logging.getLogger(__name__)

# Match server.py: websocket pings can be slow under load, give them room.
PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600


class UmiReferenceClient:
    """Reference websocket client. Single connection, single session at a
    time. The robot client should keep one of these around for an entire
    deployment session (or longer, if it can reconnect cleanly)."""

    def __init__(self, host: str = "localhost", port: int = 18000):
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        logger.info("Connecting to %s", self._uri)
        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            ping_interval=PING_INTERVAL_SECS,
            ping_timeout=PING_TIMEOUT_SECS,
        )
        # First message from the server is the UmiPolicyServerConfig.
        self._config: Dict = msgpack_numpy.unpackb(self._ws.recv())
        logger.info("Server config: %s", self._config)

    # ------------------------------------------------------------------
    # Config accessors. Use these instead of hardcoding values in your
    # robot client — if the server is reconfigured (e.g. retrained at a
    # different resolution), you should pick that up automatically.
    # ------------------------------------------------------------------
    def config(self) -> Dict:
        return self._config

    def image_hw(self) -> tuple[int, int]:
        return int(self._config["image_height"]), int(self._config["image_width"])

    def fps(self) -> int:
        return int(self._config["fps"])

    def action_horizon(self) -> int:
        return int(self._config["action_horizon"])

    def camera_keys(self) -> list[str]:
        return list(self._config["camera_keys"])

    def state_keys(self) -> list[str]:
        return list(self._config["state_keys"])

    def state_dims(self) -> list[int]:
        return list(self._config["state_dims"])

    def action_keys(self) -> list[str]:
        return list(self._config["action_keys"])

    # ------------------------------------------------------------------
    # Three protocol endpoints.
    # ------------------------------------------------------------------
    def _send_and_recv(self, payload: Dict):
        self._ws.send(self._packer.pack(payload))
        response = self._ws.recv()
        if isinstance(response, str):
            # Server returned text -> traceback. Raise to surface upstream.
            raise RuntimeError(f"Server error:\n{response}")
        return msgpack_numpy.unpackb(response)

    def push(self, frames: Dict[str, np.ndarray], session_id: str) -> Dict:
        """Buffer-only update. Call this for every captured frame at 30 Hz
        when you do not need a new action chunk yet.

        Args:
            frames: dict { camera_key: np.uint8 (H, W, 3) RGB }. Must
                contain every key from .camera_keys(); pre-resize to
                .image_hw() before sending (the server will resize if you
                don't, but you save bandwidth by doing it locally).
            session_id: stable id for the current episode.

        Returns:
            { "buffer_lens": { camera_key: int } } indicating how many
            frames the server is now holding for each camera.
        """
        for k in self.camera_keys():
            if k not in frames:
                raise KeyError(f"frames is missing required camera key {k!r}")
        payload = {"endpoint": "push", "session_id": session_id}
        payload.update(frames)
        return self._send_and_recv(payload)

    def infer(
        self,
        frames: Dict[str, np.ndarray],
        state: Dict[str, np.ndarray],
        prompt: str,
        session_id: str,
    ) -> Dict[str, np.ndarray]:
        """Push the current frame then run inference; returns the absolute
        action chunk as a dict of subkey -> (action_horizon, D) float32 array.

        Args:
            frames: same shape contract as push().
            state: dict { state_key: np.float64 (D,) or (1, D) } — robot's
                current proprio at chunk anchor. Required because the
                checkpoint is relative_action=True; the server composes
                T_abs = T_state @ T_rel inside unapply. Must contain every
                key listed by .state_keys() with dims matching .state_dims().
                The 24 returned poses are all anchored on this single state.
            prompt: natural-language task instruction.
            session_id: stable id for the current episode.

        Returns:
            dict mapping each entry in .action_keys() to a (T, D) array
            of ABSOLUTE poses where T = .action_horizon() (24 by default).
        """
        for k in self.camera_keys():
            if k not in frames:
                raise KeyError(f"frames is missing required camera key {k!r}")
        for k, dim in zip(self.state_keys(), self.state_dims()):
            if k not in state:
                raise KeyError(f"state is missing required key {k!r} (expected dim {dim})")
        payload = {"endpoint": "infer", "session_id": session_id, "prompt": prompt}
        payload.update(frames)
        payload.update(state)
        return self._send_and_recv(payload)

    def reset(self, session_id: str) -> Dict:
        """Drop the session's buffer + reset the action head KV cache.
        Call at episode boundaries."""
        return self._send_and_recv({"endpoint": "reset", "session_id": session_id})

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass


# ============================================================================
# Demo loop showing the push/infer cadence. Mirrors the training distribution
# exactly: one infer per action_horizon (24) ticks, executed straight through;
# NO mid-chunk re-planning. Mid-chunk replan would feed the model a (state,
# video) at t+k that the training set never contained, so we deliberately
# don't offer it. Replace the dummy camera + dummy IK with your real robot
# integration.
# ============================================================================
def demo_loop(
    host: str,
    port: int,
    prompt: str,
    num_seconds: float,
) -> None:
    """Simulates a 30 Hz robot loop with chunked open-loop execution:

      - Every tick we capture a frame.
      - We `push` it to the server so the ring buffer stays warm.
      - Once per `action_horizon` ticks (24 at 30 Hz = every 0.8 s) we
        instead `infer`, which both pushes the frame AND returns a fresh
        (24, D) action chunk.
      - We then execute that chunk one row per tick until it's consumed,
        and only then call `infer` again. There is no mid-chunk replan.

    This cadence matches how the model was trained: state(t) + video[t,
    t+3, ..., t+21] → action[t..t+23]. Re-inferring at t+k (k>0) with the
    new mid-trajectory state was not in the training distribution.
    """
    logging.basicConfig(level=logging.INFO, force=True)
    client = UmiReferenceClient(host=host, port=port)

    image_h, image_w = client.image_hw()
    camera_keys = client.camera_keys()
    state_keys = client.state_keys()
    state_dims = client.state_dims()
    action_keys = client.action_keys()
    action_horizon = client.action_horizon()
    period_s = 1.0 / client.fps()

    session_id = str(uuid.uuid4())
    client.reset(session_id)

    pending_chunk: Dict[str, np.ndarray] | None = None
    chunk_idx: int = 0
    total_ticks = int(num_seconds * client.fps())

    # Cold-start: push N frames so the server has a populated buffer before
    # the first infer call. Without this, the first chunk's pixel context
    # is mostly the same frame repeated.
    cold_start_ticks = abs(min(client.config()["video_offsets"]))  # 21
    logger.info("Cold start: pushing %d frames before first infer", cold_start_ticks)

    for tick in range(total_ticks):
        tick_start = time.monotonic()

        # 1. Capture frames. Replace this with real camera reads.
        frames = {
            k: np.zeros((image_h, image_w, 3), dtype=np.uint8) for k in camera_keys
        }

        # 2. Decide push vs infer: only call infer once the previous chunk
        # is fully consumed (chunk_idx >= action_horizon), and never before
        # the buffer is fully primed.
        need_chunk_now = (
            tick >= cold_start_ticks
            and (pending_chunk is None or chunk_idx >= action_horizon)
        )

        if need_chunk_now:
            # Real robot client: read proprio from the robot at this exact
            # tick. The demo just sends zeros, which yields meaningless
            # action output, but the protocol round-trip is exercised.
            state = {
                k: np.zeros((d,), dtype=np.float64)
                for k, d in zip(state_keys, state_dims)
            }
            t0 = time.monotonic()
            pending_chunk = client.infer(
                frames, state=state, prompt=prompt, session_id=session_id,
            )
            logger.info(
                "tick %d: infer took %.3fs, got chunks %s",
                tick, time.monotonic() - t0,
                {k: v.shape for k, v in pending_chunk.items()},
            )
            chunk_idx = 0
        else:
            ack = client.push(frames, session_id=session_id)
            if tick % 30 == 0:
                logger.info("tick %d: push, buffer_lens=%s", tick, ack["buffer_lens"])

        # 3. Apply one row of the current chunk to the robot. Each chunk
        # is consumed strictly in order; the next infer fires only after
        # chunk_idx reaches action_horizon.
        if pending_chunk is not None:
            current_action = {
                k: pending_chunk[k][chunk_idx]
                for k in action_keys
                if k in pending_chunk
            }
            # ── plug your IK + joint command here ───────────────────────
            # current_action["action.left_pose"]    shape (9,)
            # current_action["action.left_gripper"] shape (1,)
            # current_action["action.right_pose"]   shape (9,)
            # current_action["action.right_gripper"]shape (1,)
            chunk_idx += 1

        # 4. Hold the 30 Hz cadence.
        elapsed = time.monotonic() - tick_start
        if elapsed < period_s:
            time.sleep(period_s - elapsed)
        else:
            logger.warning(
                "tick %d ran %.3fs (> period %.3fs) — falling behind real time",
                tick, elapsed, period_s,
            )

    client.reset(session_id)
    client.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--prompt", default="pick up the object")
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()
    demo_loop(
        host=args.host,
        port=args.port,
        prompt=args.prompt,
        num_seconds=args.seconds,
    )
