"""Sharded LeRobot loader used for training.

A shard is a run of whole trajectories (about `num_steps_per_shard` steps); its videos are
decoded once into memory in a background thread while the previous shard is consumed.
Every sample is a fixed window of `max_chunk_size` consecutive 24-step chunks starting at
a sampled anchor: video frames every 3 steps (plus the final frame), one state per chunk
anchor, and 24 actions per chunk, all under a single language annotation.
"""

from concurrent.futures import Future, ThreadPoolExecutor
import copy
import glob
import importlib
import json
from collections import defaultdict
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import pandas as pd
import torch.distributed as dist
from pydantic import BaseModel, Field, field_validator
from torch.utils.data import IterableDataset, get_worker_info
from tqdm import tqdm
import yaml

from spatial_umi_dream.data.schema import DatasetMetadata, EmbodimentTag
from spatial_umi_dream.data.transform import ComposedModalityTransform
from spatial_umi_dream.data.transform.se3_utils import relative_pose9
from spatial_umi_dream.data.video_io import get_frames_by_timestamps

from .lerobot import LeRobotSingleDataset, ModalityConfig


CHUNK_SIZE = 24
VIDEO_FRAME_OFFSETS = np.arange(0, CHUNK_SIZE, 3, dtype=np.int64)


def _last_full_forward_anchor(trajectory_length: int, max_chunks: int) -> int:
    """Last anchor that fits fixed forward chunks plus the final video frame."""
    return trajectory_length - CHUNK_SIZE * max_chunks - 1


def _plan_forward_chunk_anchors(
    first_idx: int,
    trajectory_length: int,
    max_chunks: int,
    language_annotations: np.ndarray | None = None,
) -> np.ndarray:
    """Chunk anchors `first_idx + 24*k` for k < max_chunks, or empty if the window does not fit.

    The window must stay inside the trajectory (including the final video frame at
    `last_anchor + 24`) and keep the language annotation of `first_idx` at every anchor.
    """
    if max_chunks <= 0:
        raise ValueError(f"max_chunks must be positive, got {max_chunks}")
    if not 0 <= first_idx < trajectory_length:
        return np.empty(0, dtype=np.int64)

    target_language = (
        language_annotations[first_idx] if language_annotations is not None else None
    )
    anchors: list[int] = []
    for chunk_idx in range(max_chunks):
        anchor = first_idx + CHUNK_SIZE * chunk_idx
        if anchor + CHUNK_SIZE >= trajectory_length:
            break
        if (
            language_annotations is not None
            and language_annotations[anchor] != target_language
        ):
            break
        anchors.append(anchor)

    if len(anchors) != max_chunks:
        return np.empty(0, dtype=np.int64)
    return np.asarray(anchors, dtype=np.int64)


class ShardedLeRobotSE3DeltaSingleDataset(LeRobotSingleDataset):
    """Sharded single dataset with multi-chunk windows and SE(3) relative actions.

    Sub-key contract (set in `meta/modality.json` of the LeRobot dataset):
      - `*_pose` (9D = `[xyz(3), rot6d(6)]`): per-chunk SE(3) composition
        delta `inv(T_anchor) @ T_action`, output stays in 9D rot6d format.
      - any other sub-key (e.g. `*_gripper`): additive subtraction of the anchor
        state, only when the sub-key is listed in `relative_action_keys`.

    Optional cross-arm relative state (bimanual): when `relative_state=True` and a
    pose sub-key is listed in `relative_state_keys`, `get_state` returns the pose
    expressed in the partner arm's frame instead of the world frame.

      state.left_pose  -> T_left_in_right  = inv(T_right) @ T_left
      state.right_pose -> T_right_in_left  = inv(T_left)  @ T_right

    Gripper sub-keys (and any subkey not in `relative_state_keys`) pass through
    unchanged.
    """

    _RELATIVE_STATE_PARTNER: dict[str, str] = {
        "left_pose": "right_pose",
        "right_pose": "left_pose",
    }

    def __init__(
        self,
        *args,
        num_steps_per_shard: int = int(1e4),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_steps_per_shard = num_steps_per_shard
        self.all_video_paths = self.get_all_video_paths()
        self.all_parquet_paths = self.get_all_parquet_paths()
        self.sharded_trajectories, self.shard_lengths = self.generate_shards()

        # Shard cache, filled by `start_cache_shard` / `finish_cache_shard`.
        self.shard_start_indices: dict[int, int] | None = None
        self.cached_shard: dict[str, np.ndarray] | None = None
        self.cached_df: pd.DataFrame | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._cache_job: Future | None = None

    def _get_step_filter(self) -> dict[int, np.ndarray]:
        """Drop anchors that cannot fit the fixed forward chunk window.

        Filtering happens at the step_filter layer (not `_get_all_steps`) because
        `ShardedLeRobotMixtureDataset.__iter__` builds its per-shard anchor list
        directly from `dataset.step_filter[trajectory_id]`.
        """
        step_filter = super()._get_step_filter()
        filtered: dict[int, np.ndarray] = {}
        for trajectory_id, indices in step_filter.items():
            trajectory_index = self.get_trajectory_index(trajectory_id)
            traj_length = self.trajectory_lengths[trajectory_index].item()
            cutoff = _last_full_forward_anchor(traj_length, self.max_chunk_size)
            candidates = indices[indices <= cutoff]
            language_key = self._get_language_key()
            language_annotations = pd.read_parquet(
                self.get_parquet_path(trajectory_id), columns=[language_key]
            )[language_key].to_numpy()
            valid = [
                _plan_forward_chunk_anchors(
                    first_idx=int(anchor),
                    trajectory_length=int(traj_length),
                    max_chunks=self.max_chunk_size,
                    language_annotations=language_annotations,
                ).size
                == self.max_chunk_size
                for anchor in candidates
            ]
            filtered[trajectory_id] = candidates[np.asarray(valid, dtype=bool)]
        return filtered

    # ------------------------------------------------------------------ shards

    def get_all_video_paths(self) -> dict[int, dict[str, Path]]:
        """trajectory_id -> {video key -> path}."""
        video_paths = {}
        for trajectory_id in self.trajectory_ids:
            if isinstance(trajectory_id, np.integer):
                trajectory_id = trajectory_id.item()
            assert isinstance(
                trajectory_id, int
            ), f"trajectory_id must be an integer, got {type(trajectory_id)}"
            video_paths[trajectory_id] = {}
            for key in self.modality_keys["video"]:
                assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
                video_paths[trajectory_id][key] = self.get_video_path(
                    trajectory_id, key.replace("video.", "")
                )
        return video_paths

    def get_all_parquet_paths(self) -> dict[int, Path]:
        return {
            trajectory_id: self.get_parquet_path(trajectory_id)
            for trajectory_id in self.trajectory_ids
        }

    def generate_shards(self) -> tuple[list[list[int]], np.ndarray]:
        """Split trajectories into shards of about `num_steps_per_shard` sampleable steps.

        Returns the trajectory ids per shard and the number of steps per shard.
        """
        sharded_trajectories = [[]]
        curr_num_steps = 0
        curr_shard_index = 0
        discarded_episode_indices = []
        trajectory_ids = self.trajectory_ids
        if self.discard_bad_trajectories:
            discarded_episode_indices = self._lerobot_info_meta.get("discarded_episode_indices", [])
            trajectory_ids = [
                trajectory_id
                for trajectory_id in trajectory_ids
                if trajectory_id not in discarded_episode_indices
            ]

        assert len(trajectory_ids) > 0, "No valid trajectories found for dataset"
        total_steps = np.sum(
            [len(self.step_filter[trajectory_id]) for trajectory_id in trajectory_ids]
        ).astype(int)
        if total_steps == 0:
            raise ValueError(
                f"No valid sampling anchors found for dataset {self.dataset_path}; "
                f"max_chunk_size={self.max_chunk_size}. The trajectories may be "
                "too short for the requested fixed forward chunk window."
            )
        num_shards = np.ceil(total_steps / self.num_steps_per_shard).astype(int)
        cutoffs = np.linspace(0, total_steps, num_shards + 1)[1:]  # Exclude the first cutoff (0)
        shard_lengths = []
        last_num_steps = 0
        for trajectory_id in trajectory_ids:
            sharded_trajectories[-1].append(trajectory_id)
            curr_num_steps += len(self.step_filter[trajectory_id])
            if curr_num_steps > cutoffs[curr_shard_index]:
                sharded_trajectories.append([])
                curr_shard_index += 1
                shard_lengths.append(curr_num_steps - last_num_steps)
                last_num_steps = curr_num_steps
        shard_lengths.append(curr_num_steps - last_num_steps)
        assert (
            curr_num_steps == total_steps
        ), "Total steps not equal to the sum of trajectory lengths"
        assert (
            len(shard_lengths) == num_shards
        ), "Number of shards not equal to the number of cutoffs"
        assert (
            len(sharded_trajectories) == num_shards
        ), "Number of shards not equal to the number of cutoffs"
        print(f"Generated {len(sharded_trajectories)} shards for dataset {self.dataset_path}")
        return sharded_trajectories, np.array(shard_lengths)

    @staticmethod
    def get_shard(
        trajectory_ids: list[int] | np.ndarray,
        modality_keys: dict,
        video_paths: dict[int, dict[str, Path]],
        parquet_paths: dict[int, Path],
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[int, int], pd.DataFrame]:
        """Decode every frame of every video in the shard and concatenate the parquets.

        Static so it can run in the background thread. Returns the frames per video key,
        the start row of each trajectory in the concatenated arrays, and the dataframe.
        """
        print("Caching shard")
        start_time = time.time()
        assert "video" in modality_keys, "No video modality found. No need to use caching."
        cached_frames = {}
        trajectory_start_indices = {}
        curr_step_index = 0
        cached_df = None
        for trajectory_id in trajectory_ids:
            trajectory_start_indices[trajectory_id] = curr_step_index
            parquet_path = parquet_paths[trajectory_id]
            parquet_df = pd.read_parquet(parquet_path)
            parquet_timestamps = parquet_df["timestamp"].to_numpy()
            trajectory_length = len(parquet_timestamps)
            if isinstance(trajectory_id, np.integer):
                trajectory_id = trajectory_id.item()
            assert isinstance(
                trajectory_id, int
            ), f"trajectory_id must be an integer, got {type(trajectory_id)}"
            for key in modality_keys["video"]:
                assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
                if key not in cached_frames:
                    cached_frames[key] = []
                frames = get_frames_by_timestamps(
                    video_paths[trajectory_id][key].as_posix(),
                    timestamps=parquet_timestamps,
                    video_backend=video_backend,
                    video_backend_kwargs=video_backend_kwargs,
                )
                cached_frames[key].append(frames)
            if cached_df is None:
                cached_df = parquet_df
            else:
                cached_df = pd.concat([cached_df, parquet_df])
            curr_step_index += trajectory_length

        for key in cached_frames:
            cached_frames[key] = np.concatenate(cached_frames[key], axis=0)
        end_time = time.time()
        print(f"Cached shard in {end_time - start_time:.2f} seconds")
        assert cached_df is not None, "Cached dataframe is None"
        # Some dataset formats omit the global "index" column.
        if "index" not in cached_df.columns:
            cached_df = cached_df.reset_index(drop=True)
            cached_df["index"] = cached_df.index
        return cached_frames, trajectory_start_indices, cached_df

    def start_cache_shard(self, shard_index: int) -> None:
        """Start caching a shard in a background thread."""
        self._cache_job = self._executor.submit(
            self.get_shard,
            self.sharded_trajectories[shard_index],
            self.modality_keys,
            self.all_video_paths,
            self.all_parquet_paths,
            self.video_backend,
            self.video_backend_kwargs,
        )

    def finish_cache_shard(self):
        """Wait for the background job and install the shard as the current cache."""
        assert self._cache_job is not None
        self.cached_shard, self.shard_start_indices, self.cached_df = self._cache_job.result()
        self._cache_job = None  # Let the future's memory go.

    def delete_cached_shard(self):
        del self.cached_shard
        del self.shard_start_indices
        del self.cached_df

    def get_trajectories_in_shard(self) -> list[int]:
        assert self.shard_start_indices is not None
        return list(self.shard_start_indices.keys())

    # ----------------------------------------------------------------- samples

    def get_step_data(self, trajectory_id: int, indices: dict[str, np.ndarray]) -> dict | None:
        """RAW data for one sample; None if the forward chunk window does not fit."""
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        for modality in self.modality_keys:
            for key in self.modality_keys[modality]:
                if key in indices:
                    data[key] = self.get_data_by_modality(
                        trajectory_id, modality, key, indices[key]
                    )
                    if len(data[key]) == 0:
                        return None
        return data

    def _get_language_key(self) -> str:
        """Parquet column holding the language annotation (first `annotation.*` key)."""
        for modality_config in self.modality_configs.values():
            for modality_key in modality_config.modality_keys:
                if not modality_key.startswith("annotation."):
                    continue
                subkey = modality_key.removeprefix("annotation.")
                language_key = self.lerobot_modality_meta.annotation[
                    subkey
                ].original_key
                return language_key or subkey
        raise ValueError("Language annotation key not found in dataset modalities")

    def _get_language_annotations(self, traj_data: pd.DataFrame) -> np.ndarray:
        language_key = self._get_language_key()
        if language_key not in traj_data.columns:
            raise ValueError(f"Language column {language_key!r} not found in trajectory")
        return traj_data[language_key].values

    def _plan_chunk_anchors(self, trajectory_id: int, step_indices: np.ndarray) -> np.ndarray:
        """Forward chunk anchors for a sample whose first requested step is `step_indices[0]`."""
        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = self.trajectory_lengths[trajectory_index]
        traj_data = self.get_trajectory_data(trajectory_id)
        language_annotations = self._get_language_annotations(traj_data)
        first_idx = int(step_indices[0]) if len(step_indices) > 0 else -1
        return _plan_forward_chunk_anchors(
            first_idx=first_idx,
            trajectory_length=int(trajectory_length),
            max_chunks=self.max_chunk_size,
            language_annotations=language_annotations,
        )

    def get_video(self, trajectory_id: int, key: str, step_indices: np.ndarray) -> np.ndarray:
        """Video frames of the forward window from the cached shard, shape (T, H, W, C).

        T = 8 frames per chunk (stride 3) plus the final frame at `last_anchor + 24`.
        """
        chunk_anchors = self._plan_chunk_anchors(trajectory_id, step_indices)
        if chunk_anchors.size == 0:
            sampled_indices = np.empty(0, dtype=np.int64)
        else:
            sampled_indices = (
                chunk_anchors[:, None] + VIDEO_FRAME_OFFSETS[None, :]
            ).reshape(-1)
            final_frame = chunk_anchors[-1] + CHUNK_SIZE
            sampled_indices = np.append(sampled_indices, final_frame)

        assert (
            self.shard_start_indices is not None
            and self.cached_shard is not None
            and trajectory_id in self.shard_start_indices
        ), "Shard not cached. Please call `cache_next_shard` and `use_next_shard` first."
        indices_in_shard = self.shard_start_indices[trajectory_id] + sampled_indices
        return self.cached_shard[key][indices_in_shard]

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray | list[str]:
        if modality == "video":
            return self.get_video(trajectory_id, key, step_indices)
        elif modality == "state":
            return self.get_state(trajectory_id, modality, key, step_indices)
        elif modality == "action":
            return self.get_action(trajectory_id, modality, key, step_indices)
        elif modality == "language":
            return self.get_language(trajectory_id, key, step_indices)
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def _get_absolute_state(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """World-frame state at each chunk anchor, shape (max_chunk_size, D)."""
        data_array, max_length, padding_strategy = self._get_state_or_action_array(
            trajectory_id, modality, key
        )
        sampled_indices = self._plan_chunk_anchors(trajectory_id, step_indices)
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=sampled_indices,
            max_length=max_length,
            padding_strategy=padding_strategy,
        )

    def get_state(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """State at each chunk anchor; pose sub-keys in `relative_state_keys` are expressed
        in the partner arm's frame."""
        abs_state = self._get_absolute_state(trajectory_id, modality, key, step_indices)

        if not self.relative_state:
            return abs_state

        subkey = key[len(modality) + 1:]
        if subkey not in self.relative_state_keys:
            return abs_state

        partner_subkey = self._RELATIVE_STATE_PARTNER.get(subkey)
        assert partner_subkey is not None, (
            f"relative_state requested for sub-key '{subkey}' but no partner "
            f"mapping is defined (known pairs: {list(self._RELATIVE_STATE_PARTNER)})"
        )
        partner_key = f"{modality}.{partner_subkey}"
        partner_state = self._get_absolute_state(trajectory_id, modality, partner_key, step_indices)

        assert abs_state.shape == partner_state.shape, (
            f"relative_state shape mismatch: this={abs_state.shape} vs "
            f"partner={partner_state.shape} for {key=} {partner_key=}"
        )
        assert abs_state.shape[-1] == 9, (
            f"_pose relative_state expects 9D (xyz+rot6d), got "
            f"{abs_state.shape[-1]} for sub-key '{subkey}'"
        )

        # relative_pose9(chunk, base) = inv(T_base) @ T_chunk = T_chunk_in_base.
        # Passing (this, partner) yields T_this_in_partner.
        return relative_pose9(abs_state, partner_state).astype(abs_state.dtype, copy=False)

    def get_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """24 consecutive actions per chunk anchor, shape (max_chunk_size * 24, D).

        Sub-keys in `relative_action_keys` are made relative to the state at their
        chunk anchor (see `_compute_relative_chunk`).
        """
        data_array, max_length, padding_strategy = self._get_state_or_action_array(
            trajectory_id, modality, key
        )
        chunk_anchors = self._plan_chunk_anchors(trajectory_id, step_indices)
        if chunk_anchors.size == 0:
            sampled_indices = np.empty(0, dtype=np.int64)
        else:
            sampled_indices = (
                chunk_anchors[:, None]
                + np.arange(CHUNK_SIZE, dtype=np.int64)[None, :]
            ).reshape(-1)

        action_data = self.retrieve_data_and_pad(
            array=data_array,
            step_indices=sampled_indices,
            max_length=max_length,
            padding_strategy=padding_strategy,
        )
        subkey = key.replace("action.", "")
        should_convert_to_relative = (
            self.relative_action
            and len(sampled_indices) > 0
            and (self.relative_action_keys is None or subkey in self.relative_action_keys)
        )
        if should_convert_to_relative:
            action_data = self._convert_to_relative_action(
                action_data=action_data,
                action_key=key,
                sampled_indices=sampled_indices,
                trajectory_id=trajectory_id,
            )

        return action_data

    def _convert_to_relative_action(
        self,
        action_data: np.ndarray,
        action_key: str,
        sampled_indices: np.ndarray,
        trajectory_id: int,
    ) -> np.ndarray:
        """Make each 24-step chunk relative to the state (same sub-key) at its anchor step."""
        subkey = action_key.replace("action.", "")

        traj_data = self.get_trajectory_data(trajectory_id)
        le_state_cfg = self.lerobot_modality_meta.state

        if subkey not in le_state_cfg:
            return action_data

        le_state_key = le_state_cfg[subkey].original_key
        if le_state_key is None:
            le_state_key = subkey

        if le_state_key not in traj_data.columns:
            return action_data

        state_array: np.ndarray = np.stack(traj_data[le_state_key])
        if state_array.ndim == 1:
            state_array = state_array.reshape(-1, 1)

        le_indices = np.arange(
            le_state_cfg[subkey].start,
            le_state_cfg[subkey].end,
        )
        state_array = state_array[:, le_indices]

        relative_action_data = action_data.copy()
        num_chunks = len(sampled_indices) // CHUNK_SIZE

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * CHUNK_SIZE
            chunk_end = chunk_start + CHUNK_SIZE
            anchor_idx = sampled_indices[chunk_start]

            if anchor_idx < len(state_array):
                relative_action_data[chunk_start:chunk_end] = self._compute_relative_chunk(
                    action_chunk=action_data[chunk_start:chunk_end],
                    anchor_state=state_array[anchor_idx],
                    action_subkey=subkey,
                )

        return relative_action_data

    def _compute_relative_chunk(
        self,
        action_chunk: np.ndarray,
        anchor_state: np.ndarray,
        action_subkey: str,
    ) -> np.ndarray:
        """Per-chunk delta: SE(3) composition for `_pose` sub-keys, additive otherwise.

        Must match the offline stats pass in `LeRobotSingleDataset._calculate_relative_stats_for_key`.
        Inputs:
            action_chunk: shape (CHUNK_SIZE, sub_D)
            anchor_state: shape (sub_D,)
        """
        if action_subkey.endswith("_pose"):
            assert action_chunk.shape[-1] == 9, (
                f"_pose sub-key expects 9D (xyz+rot6d), got "
                f"{action_chunk.shape[-1]} for sub-key '{action_subkey}'"
            )
            assert anchor_state.shape[-1] == 9, (
                f"_pose anchor state expects 9D, got {anchor_state.shape[-1]} "
                f"for sub-key '{action_subkey}'"
            )
            return relative_pose9(action_chunk, anchor_state)
        return action_chunk - anchor_state

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Rows of one trajectory from the cached shard dataframe."""
        assert self.cached_df is not None, "Cached dataframe is None"
        if self.cached_df.empty:
            raise ValueError("cached_df is completely empty!")

        available_episodes = self.cached_df["episode_index"].unique()
        if trajectory_id not in available_episodes:
            raise ValueError(
                f"trajectory_id {trajectory_id} not found in cached_df. "
                f"Available episodes: {sorted(available_episodes)}"
            )

        traj_data = self.cached_df.loc[self.cached_df["episode_index"] == trajectory_id]
        trajectory_index = self.get_trajectory_index(trajectory_id)
        trajectory_length = self.trajectory_lengths[trajectory_index]
        assert (
            len(traj_data) == trajectory_length
        ), f"Trajectory length mismatch: {len(traj_data)} != {trajectory_length} for {trajectory_id=} in {self}"
        indices = traj_data["index"].to_numpy()
        if len(indices) > 0:
            start_index = indices[0]
            expected_indices = np.arange(start_index, start_index + len(indices))
            assert np.array_equal(
                indices, expected_indices
            ), f"[{self}] Index sequence mismatch in trajectory data, {trajectory_id=}"
        return traj_data


class MixtureSpecElement(BaseModel):
    """One entry of a mixture: dataset paths per embodiment tag and the mixture weight."""

    dataset_path: dict[str, list[Path] | Path] = Field(..., description="The path to the dataset.")
    dataset_weight: float = Field(..., description="The weight of the dataset in the mixture.")
    distribute_weights: bool = Field(
        default=False,
        description="Whether to distribute the weights of the dataset across all the paths. If True, the weights will be evenly distributed across all the paths.",
    )

    @field_validator("dataset_path", mode="after")
    def validate_dataset_path_keys(cls, v: dict[str, list[Path] | Path]) -> dict[str, list[Path]]:
        """Check the embodiment tags and expand glob patterns in the paths."""
        all_globbed_paths: dict[str, list[Path]] = {}
        for embodiment_tag, paths in v.items():
            try:
                _ = EmbodimentTag(embodiment_tag)
            except ValueError:
                raise ValueError(f"Invalid embodiment tag: {embodiment_tag}")
            if isinstance(paths, Path):
                paths = [paths]
            globbed_paths = []
            for path in paths:
                globbed_paths.extend(glob.glob(str(path)))
            all_globbed_paths[embodiment_tag] = globbed_paths
        return all_globbed_paths


class ShardedLeRobotMixtureDataset(IterableDataset):
    """Weighted mixture of sharded datasets, iterated shard by shard.

    A schedule of (dataset, shard) pairs is drawn once from the shard weights with
    `seed`; each rank/worker takes its own slice of that schedule, so every worker
    sees distinct shards. Within a shard the anchors are shuffled and the first
    `num_steps_per_shard * shard_sampling_rate` of them are yielded.
    """

    def __init__(
        self,
        data_mixture: list[tuple[ShardedLeRobotSE3DeltaSingleDataset, float]],
        training: bool,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        shard_sampling_rate: float = 0.5,
        num_shards_to_sample: int = 2**20,
    ):
        """
        Args:
            data_mixture: Datasets and their corresponding weights.
            training: If True, shards are drawn randomly and anchors shuffled; otherwise
                shards cycle in order and anchors keep their order.
            balance_dataset_weights: Multiply each dataset weight by its number of steps.
            balance_trajectory_weights: Must be True (trajectories are weighted by length).
            seed: Random seed for sampling.
            shard_sampling_rate: Fraction of `num_steps_per_shard` yielded per shard, in [0, 1].
            num_shards_to_sample: Length of the shard schedule.
        """
        datasets: list[ShardedLeRobotSE3DeltaSingleDataset] = []
        dataset_sampling_weights: list[float] = []
        for dataset, weight in data_mixture:
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)
        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.seed = seed
        self.training = training

        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])
        self._dataset_sampling_weights = np.array(dataset_sampling_weights)
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths
        self._dataset_sampling_weights /= self._dataset_sampling_weights.sum()

        # Merged per-embodiment metadata; the trainer dumps it next to the checkpoint so
        # inference can normalize/unnormalize the same way.
        self.update_metadata(percentile_mixing_method="min_max")

        if self.training:
            for dataset in self.datasets:
                dataset.transforms.train()
        else:
            for dataset in self.datasets:
                dataset.transforms.eval()

        self.shard_sampling_rate = shard_sampling_rate
        self.num_shards_to_sample = num_shards_to_sample

        # Shard weight = dataset weight x share of the dataset's steps in that shard.
        all_shard_sampling_weights = []
        all_shards = []
        for dataset_id, (dataset, weight) in enumerate(
            zip(self.datasets, self._dataset_sampling_weights)
        ):
            shard_sampling_weights = dataset.shard_lengths / dataset.shard_lengths.sum()
            all_shard_sampling_weights.append(shard_sampling_weights * weight)
            all_shards.extend(
                [(dataset_id, shard_idx) for shard_idx in range(shard_sampling_weights.shape[0])]
            )
        all_shard_sampling_weights = np.concatenate(all_shard_sampling_weights)
        all_shard_sampling_weights /= all_shard_sampling_weights.sum()
        self._shard_sampling_weights = all_shard_sampling_weights
        self._all_shards = all_shards

        # One schedule for all ranks and workers; each takes its slice in `__iter__`.
        self._shards_sample_schedule = self.generate_shards_sample_schedule()

        assert 0 <= shard_sampling_rate <= 1, "Shard sampling rate must be between 0 and 1"

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.worker_id = None
        self.num_workers = None

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        return self._dataset_sampling_weights

    @property
    def shard_sampling_weights(self) -> np.ndarray:
        return self._shard_sampling_weights

    @property
    def all_shards(self) -> list[tuple[int, int]]:
        """All (dataset_index, shard_index) pairs."""
        return self._all_shards

    @property
    def shards_sample_schedule(self) -> list[tuple[int, int]]:
        """(dataset_index, shard_index) pairs in iteration order."""
        assert self._shards_sample_schedule is not None, "Shards sample schedule not set."
        return self._shards_sample_schedule

    @classmethod
    def from_mixture_spec(
        cls,
        mixture_spec: Sequence[MixtureSpecElement | dict],
        dataset_class: type[LeRobotSingleDataset] | str,
        all_modality_configs: dict[str, dict[str, ModalityConfig]],
        all_transforms: dict[str, ComposedModalityTransform],
        metadata_versions: dict[str, str],
        fps: dict[str, float] | None = None,
        dataset_kwargs: dict | None = None,
        mixture_kwargs: dict | None = None,
    ) -> "ShardedLeRobotMixtureDataset":
        """Build the mixture from a spec (the `train_dataset` block of the data configs).

        Args:
            mixture_spec: One `MixtureSpecElement` (or dict) per weighted group of datasets.
            dataset_class: Single-dataset class or its dotted path.
            all_modality_configs / all_transforms / metadata_versions / fps: Keyed by embodiment tag.
            dataset_kwargs: Extra keyword arguments for every single dataset.
            mixture_kwargs: Keyword arguments for the mixture itself.
        """
        if isinstance(dataset_class, str):
            module_name, class_name = dataset_class.rsplit(".", 1)
            module = importlib.import_module(module_name)
            dataset_class = getattr(module, class_name)
        assert not isinstance(dataset_class, str), f"{dataset_class} is a string"
        assert issubclass(
            dataset_class, LeRobotSingleDataset
        ), f"{dataset_class} is not a subclass of LeRobotSingleDataset"
        data_mixture = []

        for dataset_spec in tqdm(
            mixture_spec,
            total=len(mixture_spec),
            desc="Initializing datasets",
        ):
            start_time = time.time()
            if isinstance(dataset_spec, dict):
                dataset_spec = MixtureSpecElement.model_validate(dataset_spec)
            datasets = []
            for embodiment_tag, paths in dataset_spec.dataset_path.items():
                if isinstance(paths, Path):
                    paths = [paths]
                for dataset_path in paths:
                    # Glob expansion may match stray files next to the dataset directories.
                    if '.sh' in dataset_path or '.json' in dataset_path:
                        continue
                    assert (
                        embodiment_tag in all_modality_configs
                    ), f"{embodiment_tag} not in modality_configs: {all_modality_configs.keys()}"
                    assert (
                        embodiment_tag in all_transforms
                    ), f"{embodiment_tag} not in transforms: {all_transforms.keys()}"
                    dataset = dataset_class(
                        dataset_path=dataset_path,
                        embodiment_tag=EmbodimentTag(embodiment_tag),
                        modality_configs=copy.copy(all_modality_configs[embodiment_tag]),
                        transforms=copy.copy(all_transforms[embodiment_tag]),
                        metadata_version=metadata_versions[embodiment_tag],
                        fps=fps[embodiment_tag] if embodiment_tag in fps else None,
                        **(dataset_kwargs if dataset_kwargs is not None else {}),
                    )
                    datasets.append(dataset)
            dataset_lengths = np.array([len(dataset) for dataset in datasets])
            dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            for dataset, relative_length in zip(datasets, dataset_relative_lengths):
                if dataset_spec.distribute_weights:
                    weight = relative_length * dataset_spec.dataset_weight
                else:
                    weight = dataset_spec.dataset_weight
                data_mixture.append((dataset, weight))

            print(
                f"Time taken to initialize {len(datasets)} datasets: {time.time() - start_time:.2f} seconds"
            )

        return cls(
            data_mixture=data_mixture,
            **(mixture_kwargs if mixture_kwargs is not None else {}),
        )

    # ---------------------------------------------------------------- metadata

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        """Combine per-dataset statistics ({sub-key: {min, max, mean, std, q01, q99}}) using the
        sampling weights: exact weighted mean/std, min/max over datasets, and q01/q99 either
        weighted-averaged or taken as min(q01)/max(q99).
        """
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        overall_stats: dict[str, dict[str, list[float]]] = {}

        modality_keys = per_task_stats[0].keys()

        for modality in modality_keys:
            first_mean = np.array(per_task_stats[0][modality]["mean"])
            num_dims = len(first_mean)

            weighted_means = np.zeros(num_dims)
            weighted_squares = np.zeros(num_dims)

            min_list = []
            max_list = []
            q01_list = []
            q99_list = []

            for task_idx, task_stats in enumerate(per_task_stats):
                w_i = normalized_weights[task_idx]
                stats = task_stats[modality]
                means = np.array(stats["mean"])
                stds = np.array(stats["std"])

                weighted_means += w_i * means
                weighted_squares += w_i * (stds**2 + means**2)

                min_list.append(stats["min"])
                max_list.append(stats["max"])
                q01_list.append(stats["q01"])
                q99_list.append(stats["q99"])

            overall_mean = weighted_means.tolist()

            overall_variance = weighted_squares - weighted_means**2
            overall_std = np.sqrt(np.maximum(overall_variance, 0)).tolist()

            overall_min = np.min(np.array(min_list), axis=0).tolist()
            overall_max = np.max(np.array(max_list), axis=0).tolist()

            q01_array = np.array(q01_list)
            q99_array = np.array(q99_list)
            if percentile_mixing_method == "weighted_average":
                weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()
            elif percentile_mixing_method == "min_max":
                weighted_q01 = np.min(q01_array, axis=0).tolist()
                weighted_q99 = np.max(q99_array, axis=0).tolist()
            else:
                raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
    ) -> DatasetMetadata:
        """Merge the metadata of several datasets of the same embodiment into one."""
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        merged_metadata = {}

        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        dataset_statistics = {}
        dataset_statistics["state"] = ShardedLeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = ShardedLeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_metadata["statistics"] = dataset_statistics

        # The modality definitions must agree across datasets of one embodiment.
        modality_configs = defaultdict(set)
        for metadata in metadata_dicts:
            for modality, configs in metadata["modalities"].items():
                modality_configs[modality].add(json.dumps(configs))
        merged_metadata["modalities"] = {}
        for modality, configs in modality_configs.items():
            assert (
                len(configs) == 1
            ), f"Multiple modality configs for modality {modality}: {list(configs)}"
            merged_metadata["modalities"][modality] = json.loads(configs.pop())

        return DatasetMetadata.model_validate(merged_metadata)

    def update_metadata(self, percentile_mixing_method: str) -> None:
        """Merge metadata per embodiment tag and hand it to every dataset's transforms.

        percentile_mixing_method: "weighted_average" (by sampling weight) or "min_max"
        (min of q01, max of q99) for the percentiles.
        """
        self.merged_metadata: dict[str, DatasetMetadata] = {}
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        for dataset in self.datasets:
            if dataset.tag.value not in all_metadatas:
                all_metadatas[dataset.tag.value] = []
            all_metadatas[dataset.tag.value].append(dataset.metadata)
        for tag, metadatas in all_metadatas.items():
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=self.dataset_sampling_weights.tolist(),
                percentile_mixing_method=percentile_mixing_method,
            )
        for dataset in self.datasets:
            dataset.set_transforms_metadata(self.merged_metadata[dataset.tag.value])

    # ---------------------------------------------------------------- sampling

    def reset_seed(self, seed: int):
        """Re-draw the shard schedule; the trainer calls this when resuming mid-run."""
        self.seed = seed
        self._shards_sample_schedule = self.generate_shards_sample_schedule()

    def generate_shards_sample_schedule(self):
        if self.training:
            rng = np.random.default_rng(self.seed)
            sampled_shard_ids = rng.choice(
                len(self.all_shards), size=self.num_shards_to_sample, p=self.shard_sampling_weights
            )
            shards_sample_schedule = [self.all_shards[i] for i in sampled_shard_ids]
            rng.shuffle(shards_sample_schedule)
        else:
            shards_sample_schedule = [
                self.all_shards[i % len(self.all_shards)] for i in range(self.num_shards_to_sample)
            ]
        return shards_sample_schedule

    def filter_shards_sample_schedule(self):
        """This rank/worker's slice of the schedule (round-robin over rank x worker)."""
        filtered_schedule = []
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        if self.worker_id is None:
            assert self.num_workers is None
            self.worker_id = worker_id
            self.num_workers = num_workers
        else:
            assert (
                self.worker_id == worker_id and self.num_workers == num_workers
            ), "Worker ID or number of workers has been changed since it was set. This is not allowed."

        for i, shard in enumerate(self.shards_sample_schedule):
            if i % (self.world_size * num_workers) == self.rank * num_workers + worker_id:
                filtered_schedule.append(shard)
        return filtered_schedule

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            shard_lengths = dataset.shard_lengths
            assert len(shard_lengths.shape) == 1, "Shard lengths must be a 1D array"
            num_shards = shard_lengths.shape[0]
            max_shard_length = int(shard_lengths.max())
            min_shard_length = int(shard_lengths.min())
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
                "Num shards": num_shards,
                "Max shard length": max_shard_length,
                "Min shard length": min_shard_length,
            }
            dataset_descriptions.append(dataset_description)
        return yaml.dump(
            {
                "Mixture dataset": dataset_descriptions,
                "Rank": self.rank,
                "World size": self.world_size,
            }
        )

    def __iter__(self):
        if not self.balance_trajectory_weights:
            raise NotImplementedError(
                "balance_trajectory_weights=False is not supported. Please use balance_dataset_weights=True instead."
            )

        self._shards_sample_schedule = self.filter_shards_sample_schedule()
        self.curr_shard_index = -1
        self.cache_next_shard()
        rng = np.random.default_rng(self.seed)
        for i, (dataset_index, shard_index) in enumerate(self.shards_sample_schedule):
            self.curr_shard_index += 1
            assert (
                i == self.curr_shard_index
            ), f"Shard index mismatch: {i} != {self.curr_shard_index}"
            dataset = self.datasets[dataset_index]
            wait_start = time.time()
            dataset.finish_cache_shard()
            wait_end = time.time()
            print(
                f"Rank {self.rank}, Worker {self.worker_id}: Wait for shard {shard_index} in dataset {dataset_index} in {wait_end - wait_start:.2f} seconds"
            )
            # Start caching the next shard immediately
            self.cache_next_shard()
            all_steps: list[tuple[int, int]] = []
            for trajectory_id in dataset.get_trajectories_in_shard():
                trajectory_index = dataset.get_trajectory_index(trajectory_id)
                max_delta_index = dataset.max_delta_index
                trajectory_length = dataset.trajectory_lengths[trajectory_index]
                allowed_length = trajectory_length - max_delta_index
                allowed_indices = dataset.step_filter[trajectory_id]
                allowed_indices = allowed_indices[allowed_indices <= allowed_length]
                for i in allowed_indices:
                    all_steps.append((trajectory_id, i))
            if self.training:
                rng.shuffle(all_steps)
            sampled_steps = all_steps[: int(dataset.num_steps_per_shard * self.shard_sampling_rate)]
            for trajectory_id, step_index in sampled_steps:
                indices = {
                    key: delta_indices + step_index
                    for key, delta_indices in dataset.delta_indices.items()
                }
                step_data = dataset.get_step_data(trajectory_id, indices)
                # None when the forward chunk window does not fit (should not happen after
                # the step filter, kept as a guard).
                if step_data is not None:
                    yield dataset.transforms(step_data)

            dataset.delete_cached_shard()

    def cache_next_shard(self):
        """Start caching the next shard of the schedule in a background thread."""
        next_dataset_idx, next_shard_idx = self.shards_sample_schedule[self.curr_shard_index + 1]
        self.datasets[next_dataset_idx].start_cache_shard(next_shard_idx)

    def __len__(self) -> int:
        total_length = 0
        for dataset_idx, _ in self.shards_sample_schedule:
            dataset = self.datasets[dataset_idx]
            total_length += int(dataset.num_steps_per_shard * self.shard_sampling_rate)
        return total_length
