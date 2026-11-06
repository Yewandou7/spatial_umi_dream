"""RGB + depth video canvas and layout-aware prompts, shared by training and inference.

Every sample (UMI and RoboTwin) carries four three-channel uint8 streams in the order
``[robot0 RGB, robot1 RGB, robot0 depth, robot1 depth]``. Depth streams are already-rendered
colormap videos, so they follow the ordinary Wan video path. The streams are tiled into one
eye-paired canvas::

    [robot0 RGB | robot0 depth]
    [robot1 RGB | robot1 depth]
"""

import ast
from collections.abc import Sequence

import numpy as np


RGB_VIDEO_KEYS: tuple[str, str] = (
    "video.robot0_camera",
    "video.robot1_camera",
)
DEPTH_VIDEO_KEYS: tuple[str, str] = (
    "video.robot0_depth",
    "video.robot1_depth",
)
VIDEO_KEYS: tuple[str, str, str, str] = (*RGB_VIDEO_KEYS, *DEPTH_VIDEO_KEYS)


def validate_camera_frame(frame: object, *, camera_key: str) -> np.ndarray:
    """Return one raw camera frame without performing a lossy dtype cast."""

    array = np.asarray(frame)
    if array.dtype != np.uint8:
        raise ValueError(
            f"{camera_key} must be raw uint8 pixels; got dtype {array.dtype}. "
            "Refusing to cast non-uint8 camera data because normalized float "
            "RGB/depth values would be corrupted. Render depth visualizations "
            "as three-channel uint8 before deployment."
        )
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(
            f"{camera_key} must be (H, W, 3) uint8; got shape {array.shape}"
        )
    return array


def validate_video_keys(video_keys: Sequence[str]) -> tuple[str, ...]:
    """Validate the ordered four-stream camera contract and return it as a tuple.

    Physical dataset names may differ from the defaults above, so validation checks structure
    (two RGB views, then their paired depth views) rather than hard-coding names.
    """

    keys = tuple(video_keys)
    if len(keys) != 4:
        raise ValueError(
            f"RGB+depth expects exactly 4 ordered video keys, got {len(keys)}: {keys}"
        )
    if len(set(keys)) != len(keys):
        raise ValueError(f"video keys must be unique, got: {keys}")
    invalid = [key for key in keys if not key.startswith("video.")]
    if invalid:
        raise ValueError(f"video keys must start with 'video.': {invalid}")
    return keys


def assemble_rgbd_canvas(
    video: np.ndarray,
    *,
    target_height: int | None = None,
    target_width: int | None = None,
) -> np.ndarray:
    """Assemble ``[T, 4, H, W, C]`` views into the ``[T, 2H, 2W, C]`` canvas.

    Each panel must already have its final geometry; the completed canvas is never resized.
    ``target_height``/``target_width`` optionally assert the final canvas size.
    """

    if not isinstance(video, np.ndarray) or video.ndim != 5:
        shape = getattr(video, "shape", None)
        raise ValueError(f"video must be a numpy [T,V,H,W,C] array, got {shape}")
    _, view_count, _, _, channels = video.shape
    if view_count != 4:
        raise ValueError(
            f"RGB+depth canvas expects exactly 4 views, got shape {video.shape}"
        )
    if channels != 3:
        raise ValueError(
            "RGB and depth-visualization streams must be three-channel; "
            f"got shape {video.shape}"
        )

    top = np.concatenate((video[:, 0], video[:, 2]), axis=2)
    bottom = np.concatenate((video[:, 1], video[:, 3]), axis=2)
    canvas = np.concatenate((top, bottom), axis=1)

    if (target_height is None) != (target_width is None):
        raise ValueError("target_height and target_width must be provided together")
    if target_height is not None:
        actual = canvas.shape[1:3]
        expected = (int(target_height), int(target_width))
        if actual != expected:
            raise ValueError(
                "canvas does not match the configured Wan input geometry: "
                f"got HxW={actual}, expected HxW={expected}. Resize/crop each "
                "view to the intended panel geometry before canvas assembly."
            )
    return canvas


UMI_PROMPT_PREFIX = "A four-panel RGB-and-depth video"
ROBOTWIN_PROMPT_PREFIX = "A RobotWin four-panel RGB-and-depth video"


def umi_prompt_with_layout(task: str) -> str:
    """Attach the UMI canvas description used by the text encoder (idempotent)."""

    raw_task = str(task)
    if raw_task.lower().startswith(UMI_PROMPT_PREFIX.lower()):
        return raw_task
    task = raw_task.lower()
    layout = (
        "The video is a two-by-two grid: the top-left panel is the left "
        "wrist RGB view, the top-right panel is its depth visualization, "
        "the bottom-left panel is the right wrist RGB view, and the "
        "bottom-right panel is its depth visualization."
    )
    return (
        f"{UMI_PROMPT_PREFIX} shows that a person operating a handheld bimanual gripper "
        f"system {task} {layout} The person {task}"
    )


def robotwin_prompt_with_layout(task: str) -> str:
    """Attach the RoboTwin canvas description used by train and inference (idempotent)."""

    raw_task = str(task)
    if raw_task.lower().startswith(ROBOTWIN_PROMPT_PREFIX.lower()):
        return raw_task

    # Some LeRobot language columns arrive as the string representation of a
    # one-element list. Normalize that representation before adding the shared
    # prompt so training and raw server task_text have identical semantics.
    try:
        parsed_task = ast.literal_eval(raw_task)
    except (ValueError, SyntaxError, TypeError):
        parsed_task = raw_task
    if isinstance(parsed_task, (list, tuple)) and parsed_task:
        parsed_task = parsed_task[0]
    task_text = str(parsed_task).lower()
    layout = (
        "The video is a two-by-two grid: the top-left panel is robot0's "
        "left-wrist RGB view, the top-right panel is its depth "
        "visualization, the bottom-left panel is robot1's right-wrist RGB "
        "view, and the bottom-right panel is its depth visualization."
    )
    return (
        f"{ROBOTWIN_PROMPT_PREFIX} shows a bimanual robot that must {task_text}. "
        f"{layout} The robot must {task_text}."
    )
