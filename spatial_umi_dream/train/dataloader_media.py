"""Small helpers for logging the exact video batch consumed by training."""

from __future__ import annotations

from os import PathLike

import imageio_ffmpeg
import numpy as np
import torch


def optimizer_step_for_current_batch(completed_steps: int) -> int:
    """Map Trainer's completed-step counter to the current batch's update."""

    if completed_steps < 0:
        raise ValueError(f"completed_steps must be non-negative, got {completed_steps}")
    return completed_steps + 1


def should_log_dataloader_video(
    *, step: int, every_steps: int, last_logged_step: int | None
) -> bool:
    """Return whether this optimizer step needs one media upload."""

    return (
        every_steps > 0
        and step > 0
        and step % every_steps == 0
        and step != last_logged_step
    )


def prepare_dataloader_wandb_video(images: torch.Tensor) -> np.ndarray:
    """Convert one collated sample from BTHWC uint8 to W&B's TCHW layout."""

    if not isinstance(images, torch.Tensor):
        raise TypeError(f"images must be a torch.Tensor, got {type(images).__name__}")
    if images.ndim != 5:
        raise ValueError(f"images must have BTHWC shape, got {tuple(images.shape)}")
    if images.shape[0] < 1 or images.shape[1] < 1:
        raise ValueError(f"images must contain a sample and a frame, got {tuple(images.shape)}")
    if images.shape[-1] != 3:
        raise ValueError(f"images must have three RGB channels, got {tuple(images.shape)}")
    if images.dtype != torch.uint8:
        raise TypeError(f"images must be uint8 at the dataloader boundary, got {images.dtype}")

    return (
        images[0]
        .detach()
        .cpu()
        .permute(0, 3, 1, 2)
        .contiguous()
        .numpy()
    )


def write_dataloader_video_mp4(
    video: np.ndarray,
    output_path: str | PathLike[str],
    *,
    fps: int,
) -> None:
    """Encode a TCHW uint8 RGB video without W&B's optional moviepy extra."""

    if video.ndim != 4 or video.shape[1] != 3:
        raise ValueError(f"video must have TCHW RGB shape, got {video.shape}")
    if video.dtype != np.uint8:
        raise TypeError(f"video must be uint8, got {video.dtype}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    _, _, height, width = video.shape
    writer = imageio_ffmpeg.write_frames(
        str(output_path),
        size=(width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
    )
    writer.send(None)
    try:
        for frame in video:
            writer.send(np.ascontiguousarray(frame.transpose(1, 2, 0)))
    finally:
        writer.close()
