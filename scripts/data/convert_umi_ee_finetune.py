#!/usr/bin/env python3
"""Convert depth-augmented UMI end-effector recordings into one merged v3.0 UMI
finetune dataset.

Sources are "converted" LeRobot v2.1 UMI recordings after depth_gen: they
already carry per-frame EE pose9 (`observation.ee.{left,right}`,
`action.ee.{left,right}`, computed when the recordings are converted), relabeled
gripper actions, and the two generated depth streams
(`observation.gripper.{left,right}.depth`).

This does NOT run FK: the frame of the precomputed EE columns is the frame the
pretrain data and the deploy runner use, so the columns are consumed as-is.

Output layout and conventions follow the depth pretrain datasets:

  state  = [obs.ee.left(9),  obs.gripper.left.distance(1),
            obs.ee.right(9), obs.gripper.right.distance(1)]
  action = [act.ee.left(9),  act.gripper.left(1),
            act.ee.right(9), act.gripper.right(1)]

  videos/observation.images.robot0_camera/  <- observation.gripper.left.cam0
  videos/observation.images.robot0_depth/   <- observation.gripper.left.depth
  videos/observation.images.robot1_camera/  <- observation.gripper.right.cam0
  videos/observation.images.robot1_depth/   <- observation.gripper.right.depth

  meta/: info.json (v3.0), modality.json (annotation -> task_index),
  tasks.jsonl (merged, real instructions preserved), episodes.jsonl,
  stats.json, depth_encoding.json (keys rewritten).

Task texts come from each source's own tasks.jsonl via its per-episode
task_index. Videos are symlinked, never re-encoded. Per-episode frame counts
are verified against all four videos with ffprobe. No fallbacks.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

VIDEO_KEY_MAP = {
    "observation.gripper.left.cam0": "observation.images.robot0_camera",
    "observation.gripper.left.depth": "observation.images.robot0_depth",
    "observation.gripper.right.cam0": "observation.images.robot1_camera",
    "observation.gripper.right.depth": "observation.images.robot1_depth",
}

STATE_COLS = [
    ("observation.ee.left", 9),
    ("observation.gripper.left.distance", 1),
    ("observation.ee.right", 9),
    ("observation.gripper.right.distance", 1),
]
ACTION_COLS = [
    ("action.ee.left", 9),
    ("action.gripper.left", 1),
    ("action.ee.right", 9),
    ("action.gripper.right", 1),
]

FEATURE_NAMES = [
    "robot0_x", "robot0_y", "robot0_z",
    "robot0_r1x", "robot0_r1y", "robot0_r1z",
    "robot0_r2x", "robot0_r2y", "robot0_r2z",
    "robot0_gripper",
    "robot1_x", "robot1_y", "robot1_z",
    "robot1_r1x", "robot1_r1y", "robot1_r1z",
    "robot1_r2x", "robot1_r2y", "robot1_r2z",
    "robot1_gripper",
]

MODALITY_JSON = {
    "action": {
        "left_pose": {"start": 0, "end": 9},
        "left_gripper": {"start": 9, "end": 10},
        "right_pose": {"start": 10, "end": 19},
        "right_gripper": {"start": 19, "end": 20},
    },
    "state": {
        "left_pose": {"start": 0, "end": 9},
        "left_gripper": {"start": 9, "end": 10},
        "right_pose": {"start": 10, "end": 19},
        "right_gripper": {"start": 19, "end": 20},
    },
    "video": {
        "robot0_camera": {"original_key": "observation.images.robot0_camera"},
        "robot1_camera": {"original_key": "observation.images.robot1_camera"},
        "robot0_depth": {"original_key": "observation.images.robot0_depth"},
        "robot1_depth": {"original_key": "observation.images.robot1_depth"},
    },
    "annotation": {
        "task": {"original_key": "task_index"},
    },
}


def stack_cols(df: pd.DataFrame, spec: list[tuple[str, int]]) -> np.ndarray:
    parts = []
    for col, width in spec:
        if col not in df.columns:
            raise KeyError(f"missing column {col}")
        arr = np.stack(df[col].values).astype(np.float32).reshape(len(df), -1)
        if arr.shape[1] != width:
            raise ValueError(f"{col}: width {arr.shape[1]}, expected {width}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{col}: non-finite values")
        parts.append(arr)
    return np.concatenate(parts, axis=1)


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


def load_source(root: Path) -> dict:
    with (root / "meta" / "info.json").open() as f:
        info = json.load(f)
    tasks = {}
    with (root / "meta" / "tasks.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            tasks[int(row["task_index"])] = row["task"]
    return {"root": root, "info": info, "tasks": tasks}


def source_video(root: Path, info: dict, key: str, episode: int) -> Path:
    chunk = episode // int(info["chunks_size"])
    return root / info["video_path"].format(
        episode_chunk=chunk, video_key=key, episode_index=episode
    )


def build_features(fps: int) -> dict:
    video_feature = {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }
    features: dict = {}
    for key in VIDEO_KEY_MAP.values():
        features[key] = json.loads(json.dumps(video_feature))
    for name in ("observation.state", "action"):
        features[name] = {
            "dtype": "float32", "shape": [20], "names": FEATURE_NAMES,
        }
    for col, dtype in [
        ("timestamp", "float32"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
    ]:
        features[col] = {"dtype": dtype, "shape": [1], "names": None}
    return features


def stats_entry(arr: np.ndarray) -> dict:
    return {
        "max": np.max(arr, axis=0).tolist(),
        "min": np.min(arr, axis=0).tolist(),
        "mean": np.mean(arr, axis=0).tolist(),
        "std": np.std(arr, axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, action="append", required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--chunks-size", type=int, default=1000)
    args = parser.parse_args()

    dst = args.dst.resolve()
    if dst.exists():
        raise FileExistsError(f"{dst} already exists; delete it to redo")

    sources = [load_source(p.resolve()) for p in args.src]
    fps = {int(s["info"]["fps"]) for s in sources}
    if len(fps) != 1:
        raise ValueError(f"sources disagree on fps: {fps}")
    fps = fps.pop()

    merged_tasks: list[str] = []
    task_lookup: dict[str, int] = {}

    episode_lengths: list[int] = []
    all_state, all_action, all_ts = [], [], []
    new_ep = 0
    running_index = 0
    for source in sources:
        root, info = source["root"], source["info"]
        for old_ep in range(int(info["total_episodes"])):
            chunk_src = old_ep // int(info["chunks_size"])
            parquet = root / info["data_path"].format(
                episode_chunk=chunk_src, episode_index=old_ep
            )
            df = pd.read_parquet(parquet)
            n = len(df)
            state = stack_cols(df, STATE_COLS)
            action = stack_cols(df, ACTION_COLS)

            src_task_indices = set(df["task_index"].astype(int).tolist())
            if len(src_task_indices) != 1:
                raise ValueError(f"{parquet}: multiple task_index values")
            task_text = source["tasks"][src_task_indices.pop()]
            if task_text not in task_lookup:
                task_lookup[task_text] = len(merged_tasks)
                merged_tasks.append(task_text)
            task_idx = task_lookup[task_text]

            ts = df["timestamp"].astype(np.float64).values
            ts = ts - ts[0]

            new_chunk = new_ep // args.chunks_size
            out = pd.DataFrame({
                "index": np.arange(running_index, running_index + n, dtype=np.int64),
                "episode_index": np.full(n, new_ep, dtype=np.int64),
                "frame_index": df["frame_index"].astype(np.int64).values,
                "timestamp": ts.astype(np.float32),
                "task_index": np.full(n, task_idx, dtype=np.int64),
                "observation.state": list(state),
                "action": list(action),
            })
            out_parquet = dst / "data" / f"chunk-{new_chunk:03d}" / f"file-{new_ep:03d}.parquet"
            out_parquet.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(out_parquet, index=False)

            for src_key, dst_key in VIDEO_KEY_MAP.items():
                src_video = source_video(root, info, src_key, old_ep)
                if not src_video.is_file():
                    raise FileNotFoundError(src_video)
                dst_video = (
                    dst / "videos" / dst_key / f"chunk-{new_chunk:03d}"
                    / f"file-{new_ep:03d}.mp4"
                )
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(src_video.resolve(), dst_video)
                frames = probe_frames(dst_video)
                if frames != n:
                    raise RuntimeError(
                        f"episode {new_ep} ({src_video.name} {src_key}): "
                        f"video has {frames} frames, parquet has {n}"
                    )

            all_state.append(state)
            all_action.append(action)
            all_ts.append(ts.astype(np.float32))
            episode_lengths.append(n)
            running_index += n
            print(f"[{root.name} ep{old_ep:03d} -> {new_ep:03d}] frames={n} task={task_idx}", flush=True)
            new_ep += 1

    meta = dst / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    info_out = {
        "codebase_version": "v3.0",
        "robot_type": "umi",
        "total_episodes": new_ep,
        "total_frames": int(sum(episode_lengths)),
        "total_tasks": len(merged_tasks),
        "chunks_size": args.chunks_size,
        "fps": fps,
        "splits": {"train": f"0:{new_ep}"},
        "data_path": "data/chunk-{episode_chunk:03d}/file-{episode_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/file-{episode_index:03d}.mp4",
        "features": build_features(fps),
    }
    (meta / "info.json").write_text(json.dumps(info_out, indent=2))
    with (meta / "episodes.jsonl").open("w") as f:
        for ep, length in enumerate(episode_lengths):
            f.write(json.dumps({"episode_index": ep, "length": length}) + "\n")
    with (meta / "tasks.jsonl").open("w") as f:
        for i, t in enumerate(merged_tasks):
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")
    (meta / "modality.json").write_text(json.dumps(MODALITY_JSON, indent=2))

    stats = {
        "observation.state": stats_entry(np.concatenate(all_state)),
        "action": stats_entry(np.concatenate(all_action)),
        "timestamp": stats_entry(np.concatenate(all_ts).reshape(-1, 1)),
    }
    (meta / "stats.json").write_text(json.dumps(stats, indent=2))

    with (sources[0]["root"] / "meta" / "depth_encoding.json").open() as f:
        encoding = json.load(f)
    encoding["keys"] = [
        "observation.images.robot0_depth",
        "observation.images.robot1_depth",
    ]
    encoding.pop("raw_depth", None)
    (meta / "depth_encoding.json").write_text(json.dumps(encoding, indent=2))

    print(
        f"DONE episodes={new_ep} frames={sum(episode_lengths)} "
        f"tasks={len(merged_tasks)} dst={dst}"
    )


if __name__ == "__main__":
    main()
