"""Video frame decoding for the LeRobot loaders (decord backend)."""

import numpy as np

try:
    import decord

    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


def get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    video_backend: str = "decord",
    video_backend_kwargs: dict | None = None,
) -> np.ndarray:
    """Decode the frames closest to `timestamps` (seconds). Returns (T, H, W, C) uint8."""
    if video_backend != "decord":
        raise NotImplementedError(f"video_backend={video_backend!r} is not supported; use 'decord'")
    if not DECORD_AVAILABLE:
        raise ImportError("decord is not available. Install it with: pip install decord")
    vr = decord.VideoReader(video_path, **(video_backend_kwargs or {}))
    num_frames = len(vr)
    # (num_frames, 2) start/end seconds per frame; match each request to the closest frame start.
    frame_ts: np.ndarray = vr.get_frame_timestamp(range(num_frames))
    indices = np.abs(frame_ts[:, :1] - timestamps).argmin(axis=0)
    frames = vr.get_batch(indices)
    return frames.asnumpy()
