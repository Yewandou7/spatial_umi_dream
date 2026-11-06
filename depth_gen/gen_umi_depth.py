"""Generate cam0-fisheye Turbo depth videos for a UMI LeRobot v2.1 dataset.

Consumes a UMI recording (videos/chunk-XXX/<key>/episode_*.mp4 layout with
observation.gripper.{left,right}.cam{0,1} streams) and produces
a new dataset that ADDS depth while keeping every source stream:

    videos/chunk-XXX/<every source video key>/         (copied, untouched)
    videos/chunk-XXX/observation.gripper.left.depth/   (generated)
    videos/chunk-XXX/observation.gripper.right.depth/  (generated)
    depths/chunk-XXX/episode_*.npz                     (lossless uint16 mm)
    data/  meta/                                       (copied; info.json
                                                        gains the depth keys)

The depth path is the deployment sidecar's, imported from pipeline/ (the same
code that runs in pipeline.depth_sidecar at deployment):

    cam0+cam1 -> stereo-rectify (auto-K, balance=0)
    -> Fast-FoundationStereo (batch-2, pytorch1) -> rectified cam0 depth
    -> static same-camera redistort into the cam0 KB4 fisheye plane
    -> spatial_pixel_depth Turbo render

The only difference from the sidecar: the fisheye output grid is the native
640x480 cam0 grid (the sidecar renders at 448x336 and the client resizes).
The contract requires the depth video resolution to match the paired same-eye
RGB video exactly, and rendering natively avoids resampling palette colors.

No fallbacks: any inconsistency (frame-count mismatch, missing stream,
unexpected metadata) aborts the episode run with an error.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline.common import (  # noqa: E402
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    SOURCE_SIZE,
    kb4_pixels_to_rays,
)
from pipeline.depth_common import (  # noqa: E402
    BatchedDualStereo,
    _HandRig,
    render_spatial_pixel_depth,
    valid_fraction,
)

SOURCE_WIDTH, SOURCE_HEIGHT = SOURCE_SIZE  # 640, 480
HANDS = ("left", "right")
RGB_KEYS = {hand: f"observation.gripper.{hand}.cam0" for hand in HANDS}
CAM1_KEYS = {hand: f"observation.gripper.{hand}.cam1" for hand in HANDS}
DEPTH_KEYS = {hand: f"observation.gripper.{hand}.depth" for hand in HANDS}

DEPTH_ENCODING = {
    "encoding": "spatial_pixel_depth",
    "colormap": "turbo",
    "channels": 3,
    "dtype": "uint8",
    "units": "millimetre",
    "clip_min_mm": 10.0,
    "clip_max_mm": 1500.0,
    "index_formula": "round(255 * (1500 - depth_mm) / 1490)",
    "near_is": 255,
    "far_is": 0,
    "invalid_pixel": [0, 0, 0],
    "video_codec": "h264",
    "pix_fmt": "yuv420p",
    "keys": [DEPTH_KEYS["left"], DEPTH_KEYS["right"]],
    "raw_depth": {
        "path": "depths/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npz",
        "arrays": ["left", "right"],
        "shape": "[T, 480, 640]",
        "dtype": "uint16",
        "units": "millimetre",
        "invalid_value": 0,
        "note": "native-resolution raw depth, same timeline as the depth videos",
    },
    "generation": {
        "source": "cam0+cam1 same-hand stereo, pair01",
        "model": "Fast-FoundationStereo 23-36-37",
        "geometry": (
            "stereo-rectified to per-hand auto-estimated K (balance=0), "
            "statically redistorted to the native 640x480 cam0 KB4 fisheye "
            "grid; depth is cam0 camera-frame Z in metres"
        ),
        "code": "depth_gen/ (same pipeline as the deploy sidecar)",
    },
}


class NativeHandRig(_HandRig):
    """_HandRig whose fisheye output grid is the native 640x480 cam0 grid.

    The rectified stereo space stays 448x336 (the model's trained input);
    only the redistort target changes, so every depth pixel lands exactly on
    the corresponding raw cam0 RGB pixel with no post-render resize.
    """

    def _build_fisheye_maps(self) -> None:
        cam0 = self.calibrations[0]  # native 640x480 calibration, unscaled
        vv, uu = np.indices((SOURCE_HEIGHT, SOURCE_WIDTH), dtype=np.float64)
        rays_orig = kb4_pixels_to_rays(uu, vv, cam0)
        dir_rect = np.einsum("ij,hwj->hwi", self.R_cam0, rays_orig)
        zr = dir_rect[..., 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            ur = self.new_K[0, 0] * dir_rect[..., 0] / zr + self.new_K[0, 2]
            vr = self.new_K[1, 1] * dir_rect[..., 1] / zr + self.new_K[1, 2]
            scale = rays_orig[..., 2] / zr
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


class NativeBatchedDualStereo(BatchedDualStereo):
    def __init__(self, backend: str = "pytorch1") -> None:
        super().__init__(backend)
        self.left = NativeHandRig("left")
        self.right = NativeHandRig("right")

    def metric_depths(self, l0, l1, r0, r1):
        """Four raw fisheye RGB frames -> (left, right) native cam0 depth (m)."""
        lc0, lc1 = self.left.rectify(l0, l1)
        rc0, rc1 = self.right.rectify(r0, r1)
        left_disp, right_disp = self._disparities(lc0, lc1, rc0, rc1)
        return (
            self.left.depth_cam0_fisheye(left_disp),
            self.right.depth_cam0_fisheye(right_disp),
        )


class VideoReader:
    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {path}")

    def read_rgb(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if not ok:
            return None
        if frame.shape != (SOURCE_HEIGHT, SOURCE_WIDTH, 3):
            raise RuntimeError(
                f"{self.path}: frame shape {frame.shape}, "
                f"expected {(SOURCE_HEIGHT, SOURCE_WIDTH, 3)}"
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()


class VideoWriter:
    """Raw RGB frames -> H.264/yuv420p via an ffmpeg pipe."""

    def __init__(self, path: Path, fps: int, size: tuple[int, int]):
        path.parent.mkdir(parents=True, exist_ok=True)
        width, height = size
        self.path = path
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "18", "-preset", "veryfast",
                "-r", str(fps),
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame_rgb: np.ndarray) -> None:
        self.proc.stdin.write(frame_rgb.tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path}")


def probe_frames(path: Path) -> int:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def load_info(root: Path) -> dict:
    with (root / "meta" / "info.json").open() as f:
        return json.load(f)


def episode_video(root: Path, info: dict, key: str, episode: int) -> Path:
    chunk = episode // int(info["chunks_size"])
    return root / info["video_path"].format(
        episode_chunk=chunk, video_key=key, episode_index=episode
    )


def rewrite_info(info: dict) -> dict:
    """Add the two depth video features; every source feature is kept."""
    out = json.loads(json.dumps(info))
    features = out["features"]
    template = features[RGB_KEYS["left"]]
    for hand in HANDS:
        depth_feature = json.loads(json.dumps(template))
        features[DEPTH_KEYS[hand]] = depth_feature
    kept_video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    if "total_videos" in out:
        out["total_videos"] = out["total_episodes"] * len(kept_video_keys)
    return out


def process_episode(
    stereo: NativeBatchedDualStereo,
    src: Path,
    dst: Path,
    info: dict,
    episode: int,
) -> dict:
    fps = int(info["fps"])
    chunk = episode // int(info["chunks_size"])
    readers = {}
    for hand in HANDS:
        for key_map in (RGB_KEYS, CAM1_KEYS):
            path = episode_video(src, info, key_map[hand], episode)
            if not path.is_file():
                raise FileNotFoundError(path)
            readers[key_map[hand]] = VideoReader(path)

    writers = {
        hand: VideoWriter(
            episode_video(dst, info, DEPTH_KEYS[hand], episode),
            fps, (SOURCE_WIDTH, SOURCE_HEIGHT),
        )
        for hand in HANDS
    }
    depth_mm = {hand: [] for hand in HANDS}
    fractions = {hand: [] for hand in HANDS}

    frames = 0
    while True:
        got = {key: reader.read_rgb() for key, reader in readers.items()}
        ended = [key for key, frame in got.items() if frame is None]
        if ended:
            if len(ended) != len(readers):
                raise RuntimeError(
                    f"episode {episode}: streams ended at different frame "
                    f"counts (frame {frames}, ended: {sorted(ended)})"
                )
            break
        left_m, right_m = stereo.metric_depths(
            got[RGB_KEYS["left"]], got[CAM1_KEYS["left"]],
            got[RGB_KEYS["right"]], got[CAM1_KEYS["right"]],
        )
        for hand, metric in (("left", left_m), ("right", right_m)):
            writers[hand].write(render_spatial_pixel_depth(metric))
            valid = np.isfinite(metric) & (metric > 0)
            mm = np.zeros(metric.shape, dtype=np.uint16)
            mm[valid] = np.clip(
                np.rint(metric[valid] * 1000.0), 1, 65535
            ).astype(np.uint16)
            depth_mm[hand].append(mm)
            fractions[hand].append(valid_fraction(metric))
        frames += 1

    for reader in readers.values():
        reader.close()
    for writer in writers.values():
        writer.close()

    npz_path = dst / "depths" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        left=np.stack(depth_mm["left"]),
        right=np.stack(depth_mm["right"]),
    )

    # Copy every source video stream verbatim.
    for key, feature in info["features"].items():
        if feature.get("dtype") != "video":
            continue
        src_video = episode_video(src, info, key, episode)
        if not src_video.is_file():
            raise FileNotFoundError(src_video)
        dst_video = episode_video(dst, info, key, episode)
        dst_video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_video, dst_video)

    for hand in HANDS:
        rgb_frames = probe_frames(episode_video(dst, info, RGB_KEYS[hand], episode))
        depth_frames = probe_frames(episode_video(dst, info, DEPTH_KEYS[hand], episode))
        if not frames == rgb_frames == depth_frames:
            raise RuntimeError(
                f"episode {episode} {hand}: processed {frames} frames but "
                f"rgb={rgb_frames} depth={depth_frames}"
            )

    return {
        "episode": episode,
        "frames": frames,
        "left_valid_fraction": float(np.mean(fractions["left"])),
        "right_valid_fraction": float(np.mean(fractions["right"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backend", default="pytorch1", choices=("pytorch1", "tensorrt"))
    parser.add_argument("--episodes", type=int, nargs="*")
    args = parser.parse_args()

    src = args.dataset_root.resolve()
    dst = args.output_root.resolve()
    if dst == src:
        raise ValueError("output-root must differ from dataset-root")
    info = load_info(src)
    if info["codebase_version"] != "v2.1":
        raise RuntimeError(f"unexpected codebase_version {info['codebase_version']}")
    for hand in HANDS:
        for key_map in (RGB_KEYS, CAM1_KEYS):
            if key_map[hand] not in info["features"]:
                raise RuntimeError(f"dataset missing stream {key_map[hand]}")

    episodes = args.episodes or list(range(int(info["total_episodes"])))
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src / "meta", dst / "meta", dirs_exist_ok=True)
    shutil.copytree(src / "data", dst / "data", dirs_exist_ok=True)
    with (dst / "meta" / "info.json").open("w") as f:
        json.dump(rewrite_info(info), f, indent=2)
    encoding = dict(DEPTH_ENCODING)
    encoding["fps"] = int(info["fps"])
    with (dst / "meta" / "depth_encoding.json").open("w") as f:
        json.dump(encoding, f, indent=2)

    stereo = NativeBatchedDualStereo(backend=args.backend)
    stats = []
    for episode in episodes:
        result = process_episode(stereo, src, dst, info, episode)
        stats.append(result)
        print(
            f"[episode {result['episode']:06d}] frames={result['frames']} "
            f"valid L={result['left_valid_fraction']:.3f} "
            f"R={result['right_valid_fraction']:.3f}",
            flush=True,
        )

    with (dst / "meta" / "depth_generation_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"DONE episodes={len(stats)} frames={sum(s['frames'] for s in stats)}")


if __name__ == "__main__":
    main()
