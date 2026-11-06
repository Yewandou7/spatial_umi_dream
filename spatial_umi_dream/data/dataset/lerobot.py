from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm

from spatial_umi_dream.data.schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    EmbodimentTag,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)
from spatial_umi_dream.data.transform import ComposedModalityTransform
from spatial_umi_dream.data.transform.se3_utils import relative_pose9
from spatial_umi_dream.data.video_io import get_frames_by_timestamps

LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
STEP_FILTER_FILENAME = "meta/step_filter.jsonl"
LEROBOT_RELATIVE_STATS_FILE_NAME = "meta/relative_stats.json"


def calculate_dataset_statistics(
    parquet_paths: list[Path], features: list[str] | None = None
) -> dict[str, DatasetStatisticalValues]:
    """Per-column statistics (mean/std/min/max/q01/q99) over a list of parquet files."""
    all_low_dim_data_list = []
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        all_low_dim_data_list.append(pd.read_parquet(parquet_path))
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    dataset_statistics = {}
    if features is None:
        features = list(all_low_dim_data.columns)
    for le_modality in features:
        print(f"Computing statistics for {le_modality}...")
        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = DatasetStatisticalValues(
            mean=np.mean(np_data, axis=0).tolist(),
            std=np.std(np_data, axis=0).tolist(),
            min=np.min(np_data, axis=0).tolist(),
            max=np.max(np_data, axis=0).tolist(),
            q01=np.quantile(np_data, 0.01, axis=0).tolist(),
            q99=np.quantile(np_data, 0.99, axis=0).tolist(),
        )
    return dataset_statistics


class ModalityConfig(BaseModel):
    """Which keys to load for a modality and which indices (relative to the sampled base index)."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    eval_delta_indices: list[int] | None = None
    """Delta indices to sample relative to the current index for evaluation. If None, uses the same indices as delta_indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        if self.eval_delta_indices is None:
            self.eval_delta_indices = self.delta_indices


class LeRobotSingleDataset(Dataset):
    """A single LeRobot-format dataset, read directly from parquet + mp4 files.

    The base of the sharded training loader.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        use_global_metadata: bool = False,
        metadata_version: str | None = None,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        discard_bad_trajectories: bool = True,
        fps: float = None,
        max_chunk_size: int = None,
        relative_action: bool = False,
        relative_action_keys: list[str] | None = None,
        relative_action_per_horizon: bool = False,
        relative_state: bool = False,
        relative_state_keys: list[str] | None = None,
        max_trajectories: int | None = None,
    ):
        """
        Args:
            dataset_path: Root of the LeRobot dataset (contains `meta/`, `data/`, `videos/`).
            modality_configs: Per-modality keys and delta indices. See `ModalityConfig`.
            embodiment_tag: Embodiment of this dataset.
            use_global_metadata / metadata_version: Legacy; only the per-dataset metadata
                under `meta/` is supported.
            video_backend / video_backend_kwargs: Passed to `spatial_umi_dream.data.video_io`.
            transforms: Transforms applied on top of the raw step data.
            discard_bad_trajectories: Skip episodes listed in `info.json["discarded_episode_indices"]`.
            max_chunk_size: Number of 24-step forward chunks per sample (sharded loader).
            relative_action: Normalize actions with relative-action stats, loaded from (or
                computed and cached to) `meta/relative_stats.json`.
            relative_action_keys: Action sub-keys converted to relative actions; None means
                every action key without "gripper" in its name.
            relative_action_per_horizon: Legacy; per-horizon relative stats are not supported.
            relative_state / relative_state_keys: Cross-arm relative state, handled by the
                sharded subclass which knows the partner-arm mapping.
            max_trajectories: Keep only the first N episodes of `meta/episodes.jsonl`.
        """
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
        if use_global_metadata:
            raise ValueError("use_global_metadata=True is not supported; per-dataset meta/ is used.")
        if relative_action_per_horizon:
            raise ValueError("relative_action_per_horizon=True is not supported.")

        self.modality_configs = modality_configs
        self.use_global_metadata = use_global_metadata
        self.metadata_version = metadata_version
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.fps = fps
        self.max_chunk_size = max_chunk_size
        self.transforms = (
            transforms if transforms is not None else ComposedModalityTransform(transforms=[])
        )
        self.discard_bad_trajectories = discard_bad_trajectories
        self.relative_action = relative_action
        self.relative_action_keys = relative_action_keys
        self.relative_state = relative_state
        self.relative_state_keys = list(relative_state_keys) if relative_state_keys else []
        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name
        self.tag = EmbodimentTag(embodiment_tag)
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        # Discarded trajectories are intentionally included in the stats for larger state coverage.
        self._lerobot_stats_meta = self._get_lerobot_stats_meta()

        self._max_trajectories = max_trajectories

        # Trajectory info and chunk size come first: relative stats computation needs them.
        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self._data_path_pattern = self._get_data_path_pattern()
        self._chunk_size = self._get_chunk_size()

        if self.relative_action and self.relative_action_keys is None:
            action_keys = self.modality_configs.get("action", ModalityConfig(delta_indices=[0], modality_keys=[])).modality_keys
            self.relative_action_keys = [
                k.replace("action.", "") for k in action_keys
                if "gripper" not in k.lower()
            ]
            print(f"Relative action will be applied to keys: {self.relative_action_keys}")
        self._lerobot_relative_stats_meta = self._get_lerobot_relative_stats_meta() if self.relative_action else {}
        self._metadata = self._get_metadata()
        self._step_filter = self._get_step_filter()
        self._all_steps = self._get_all_steps()
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self._max_delta_index = self._get_max_delta_index()

        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        print(f"Initialized dataset {self.dataset_name} with {embodiment_tag}")

        self._video_path_pattern = self._get_video_path_pattern()
        self._tasks = self._get_tasks()
        self.curr_traj_data = None
        self.curr_traj_id = None

        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """Simplified metadata (modalities + normalization statistics) built from `meta/`."""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """Trajectory lengths, in the same order as `trajectory_ids`."""
        return self._trajectory_lengths

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """(trajectory_id, base_index) for every sampleable step."""
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        """Modality name -> list of keys, e.g. {"video": ["video.robot0_camera", ...], ...}."""
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """modality.key -> delta indices."""
        return self._delta_indices

    def _get_max_delta_index(self) -> int:
        max_delta_index = 0
        for delta_index in self.delta_indices.values():
            max_delta_index = max(max_delta_index, delta_index.max())
        return max_delta_index

    @property
    def max_delta_index(self) -> int:
        return self._max_delta_index

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        return self._lerobot_info_meta

    @property
    def lerobot_stats_meta(self) -> dict[str, DatasetStatisticalValues]:
        return self._lerobot_stats_meta

    @property
    def step_filter(self) -> dict[int, np.ndarray]:
        """trajectory_id -> sampleable base indices."""
        return self._step_filter

    @property
    def data_path_pattern(self) -> str:
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """Episodes per `data/chunk-XXX` directory (LeRobot `chunks_size`)."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        return self._tasks

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        with open(modality_meta_path, "r") as f:
            modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        return modality_meta

    def _get_lerobot_info_meta(self) -> dict:
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_lerobot_stats_meta(self) -> dict[str, DatasetStatisticalValues]:
        """Load `meta/stats.json`; if missing or malformed, compute it from the parquet files."""
        stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        try:
            with open(stats_path, "r") as f:
                stats: dict = json.load(f)
            for name in ["num_trajectories", "total_trajectory_length"]:
                stats.pop(name, None)
            for name, stat in stats.items():
                stats[name] = DatasetStatisticalValues.model_validate(stat)
            return stats
        except (FileNotFoundError, ValidationError) as e:
            print(f"Failed to load dataset statistics: {e}")
            print(f"Calculating dataset statistics for {self.dataset_name}")
            parquet_files = list((self.dataset_path).glob(LE_ROBOT_DATA_FILENAME))
            lowdim_features = []
            le_features = self.lerobot_info_meta["features"]
            for feature in le_features:
                if "float" in le_features[feature]["dtype"]:
                    lowdim_features.append(feature)

            stats = calculate_dataset_statistics(parquet_files, lowdim_features)
            stats_serialized = {k: v.model_dump(mode="json") for k, v in stats.items()}
            with open(stats_path, "w") as f:
                json.dump(stats_serialized, f, indent=4)
            return stats

    def _get_lerobot_relative_stats_meta(self) -> dict[str, DatasetStatisticalValues]:
        """Load `meta/relative_stats.json`; if missing, compute and cache it.

        `scripts/data/compute_umi_relative_stats.sh` relies on the compute path.
        """
        stats_path = self.dataset_path / LEROBOT_RELATIVE_STATS_FILE_NAME

        if stats_path.exists():
            print(f"Loading relative action stats from {stats_path}")
            with open(stats_path, "r") as f:
                stats: dict = json.load(f)
            for name, stat in stats.items():
                stats[name] = DatasetStatisticalValues.model_validate(stat)
            return stats

        print(f"Relative stats file not found at {stats_path}")
        print(f"Calculating relative action stats for {self.dataset_name}")

        all_action_keys = self.modality_configs.get("action", ModalityConfig(delta_indices=[0], modality_keys=[])).modality_keys
        if not all_action_keys:
            print("No action keys found in modality configs, skipping relative stats calculation")
            return {}

        action_keys_to_process = []
        for key in all_action_keys:
            subkey = key.replace("action.", "")
            if self.relative_action_keys is None or subkey in self.relative_action_keys:
                action_keys_to_process.append(subkey)

        if not action_keys_to_process:
            print("No action keys to process for relative stats")
            return {}

        print(f"Will calculate relative stats for: {action_keys_to_process}")

        stats = {}
        for action_key in action_keys_to_process:
            print(f"Calculating relative stats for action key: {action_key}")
            stats[action_key] = self._calculate_relative_stats_for_key(action_key)

        if stats:
            stats_serialized = {k: v.model_dump(mode="json") for k, v in stats.items()}
            print(f"Saving relative action stats to {stats_path}")
            with open(stats_path, "w") as f:
                json.dump(stats_serialized, f, indent=4)

        return stats

    def _calculate_relative_stats_for_key(self, action_key: str) -> DatasetStatisticalValues:
        """Statistics of the relative action for one sub-key, pooled over all horizon steps.

        The reference is the state at the last state delta index; `_pose` keys use the
        SE(3) delta, which must match `ShardedLeRobotSE3DeltaSingleDataset._compute_relative_chunk`.
        """
        state_key = action_key  # The state sub-key is assumed to match the action sub-key.

        state_meta = self.lerobot_modality_meta.state.get(state_key)
        action_meta = self.lerobot_modality_meta.action.get(action_key)

        if state_meta is None:
            raise ValueError(f"State key '{state_key}' not found in modality metadata")
        if action_meta is None:
            raise ValueError(f"Action key '{action_key}' not found in modality metadata")

        state_original_key = state_meta.original_key
        action_original_key = action_meta.original_key

        state_start, state_end = state_meta.start, state_meta.end
        action_start, action_end = action_meta.start, action_meta.end

        state_delta_indices = self.modality_configs.get("state", ModalityConfig(delta_indices=[0], modality_keys=[])).delta_indices
        action_delta_indices = self.modality_configs["action"].delta_indices

        print(f"Calculating relative stats for {action_key}:")
        print(f"  State: column='{state_original_key}', indices=[{state_start}:{state_end}]")
        print(f"  Action: column='{action_original_key}', indices=[{action_start}:{action_end}]")

        max_trajs_for_stats = 10000
        traj_ids_to_process = self.trajectory_ids
        if len(traj_ids_to_process) > max_trajs_for_stats:
            rng = np.random.default_rng(seed=42)
            sampled_indices = rng.choice(len(traj_ids_to_process), size=max_trajs_for_stats, replace=False)
            traj_ids_to_process = traj_ids_to_process[sampled_indices]
            print(f"Sampling {max_trajs_for_stats} trajectories out of {len(self.trajectory_ids)} for stats calculation")

        all_relative_actions = []

        for traj_id in tqdm(traj_ids_to_process, desc=f"Calculating relative stats for {action_key}"):
            traj_data = self._load_trajectory_data(traj_id)
            if state_original_key not in traj_data.columns or action_original_key not in traj_data.columns:
                raise ValueError(
                    f"Missing columns in trajectory {traj_id}: "
                    f"state='{state_original_key}' exists={state_original_key in traj_data.columns}, "
                    f"action='{action_original_key}' exists={action_original_key in traj_data.columns}"
                )

            full_state_data = np.stack(traj_data[state_original_key].values)
            full_action_data = np.stack(traj_data[action_original_key].values)
            state_data = full_state_data[:, state_start:state_end]
            action_data = full_action_data[:, action_start:action_end]

            usable_length = len(traj_data) - max(action_delta_indices)
            for i in range(usable_length):
                ref_state_idx = state_delta_indices[-1] + i
                if ref_state_idx >= len(state_data):
                    continue
                ref_state = state_data[ref_state_idx]
                action_indices = [idx + i for idx in action_delta_indices]
                if max(action_indices) >= len(action_data):
                    continue
                actions = action_data[action_indices]
                if action_key.endswith("_pose"):
                    relative_actions = relative_pose9(actions, ref_state)
                else:
                    relative_actions = actions - ref_state
                all_relative_actions.extend(relative_actions)

        if not all_relative_actions:
            raise ValueError(f"No relative actions calculated for {action_key}")

        all_relative_actions = np.array(all_relative_actions)
        print(f"Collected {len(all_relative_actions)} relative action samples for {action_key}")

        return DatasetStatisticalValues(
            max=np.max(all_relative_actions, axis=0).tolist(),
            min=np.min(all_relative_actions, axis=0).tolist(),
            mean=np.mean(all_relative_actions, axis=0).tolist(),
            std=np.std(all_relative_actions, axis=0).tolist(),
            q01=np.quantile(all_relative_actions, 0.01, axis=0).tolist(),
            q99=np.quantile(all_relative_actions, 0.99, axis=0).tolist(),
        )

    def _load_trajectory_data(self, traj_id: int) -> pd.DataFrame:
        """Parquet of one trajectory, for the relative-stats pass (fixed `file-XXX` layout)."""
        chunk_index = traj_id // self.chunk_size
        parquet_path = self.dataset_path / f"data/chunk-{chunk_index:03d}/file-{traj_id:03d}.parquet"
        return pd.read_parquet(parquet_path)

    def _get_step_filter(self) -> dict[int, np.ndarray]:
        """Sampleable base indices per trajectory; `meta/step_filter.jsonl` lists indices to drop."""
        step_filter_path = self.dataset_path / STEP_FILTER_FILENAME
        step_filter = {}
        kept_trajectory_ids = set(int(tid) for tid in self.trajectory_ids)
        if step_filter_path.exists():
            with open(step_filter_path, "r") as f:
                for line in f:
                    episode_step_filter = json.loads(line)
                    trajectory_id = episode_step_filter["episode_index"]
                    if trajectory_id not in kept_trajectory_ids:
                        continue
                    all_indices = np.arange(self.trajectory_lengths[trajectory_id].item())
                    indices_to_filter = np.array(episode_step_filter["step_indices"])
                    step_filter[trajectory_id] = np.setdiff1d(all_indices, indices_to_filter)
        else:
            for trajectory_id in self.trajectory_ids:
                step_filter[trajectory_id] = np.arange(
                    self.trajectory_lengths[trajectory_id].item()
                )
        return step_filter

    def _get_metadata(self) -> DatasetMetadata:
        """Build the simplified `DatasetMetadata` (modalities + per-sub-key statistics)."""
        # 1. Modalities
        simplified_modality_meta: dict[str, dict] = {}
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
                self.lerobot_modality_meta, modality
            )
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [
                        le_state_action_meta[subkey].end - le_state_action_meta[subkey].start
                    ],
                    "continuous": continuous,
                }

        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert (
            le_info_path.exists()
        ), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in self.lerobot_modality_meta.video:
            original_key = self.lerobot_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # LeRobot dataset versions differ in where channels / fps live.
            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
                fps = le_video_meta["video_info"]["video.fps"]
            except (ValueError, KeyError):
                channels = le_video_meta["info"]["video.channels"]
                fps = le_video_meta["info"]["video.fps"]
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        # 2. Statistics: slice the per-column stats to each sub-key; relative-action keys
        #    use the relative stats instead.
        dataset_statistics = {}
        le_statistics = {k: v.model_dump() for k, v in self.lerobot_stats_meta.items()}
        relative_stats = {}
        if self.relative_action:
            relative_stats = {k: v.model_dump() for k, v in self._lerobot_relative_stats_meta.items()}

        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = self.lerobot_modality_meta.get_key_meta(
                    f"{our_modality}.{subkey}"
                )
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)

                should_use_relative = (
                    our_modality == "action"
                    and self.relative_action
                    and subkey in relative_stats
                    and (self.relative_action_keys is None or subkey in self.relative_action_keys)
                )

                if should_use_relative:
                    for stat_name in relative_stats[subkey]:
                        dataset_statistics[our_modality][subkey][stat_name] = relative_stats[subkey][stat_name]
                    print(f"Using relative stats for {subkey}: {dataset_statistics[our_modality][subkey]}")
                else:
                    le_modality = state_action_meta.original_key
                    for stat_name in le_statistics[le_modality]:
                        indices = np.arange(
                            state_action_meta.start,
                            state_action_meta.end,
                        )
                        stat = np.array(le_statistics[le_modality][stat_name])
                        dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=self.tag,
        )

        return metadata

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Trajectory ids and lengths from `meta/episodes.jsonl`, truncated to `max_trajectories`."""
        episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]
        trajectory_ids = []
        trajectory_lengths = []
        for episode in episode_metadata:
            trajectory_ids.append(episode["episode_index"])
            trajectory_lengths.append(episode["length"])
        trajectory_ids = np.array(trajectory_ids)
        trajectory_lengths = np.array(trajectory_lengths)
        if self._max_trajectories is not None:
            assert self._max_trajectories > 0, "max_trajectories must be positive"
            assert self._max_trajectories <= len(trajectory_ids), (
                f"max_trajectories={self._max_trajectories} exceeds dataset size "
                f"{len(trajectory_ids)} for {self.dataset_path}"
            )
            total_episodes = len(trajectory_ids)
            total_frames = int(trajectory_lengths.sum())
            trajectory_ids = trajectory_ids[: self._max_trajectories]
            trajectory_lengths = trajectory_lengths[: self._max_trajectories]
            kept_frames = int(trajectory_lengths.sum())
            print(
                f"max_trajectories={self._max_trajectories}: keeping first "
                f"{self._max_trajectories}/{total_episodes} episodes "
                f"({kept_frames}/{total_frames} frames) "
                f"of {self.dataset_path}"
            )
            # Picked up by the trainer and logged to W&B.
            self._truncation_info = {
                f"dataset/{self.dataset_path.name}/kept_episodes": self._max_trajectories,
                f"dataset/{self.dataset_path.name}/total_episodes": total_episodes,
                f"dataset/{self.dataset_path.name}/kept_frames": kept_frames,
                f"dataset/{self.dataset_path.name}/total_frames": total_frames,
            }
        return trajectory_ids, trajectory_lengths

    def _get_all_steps(self) -> list[tuple[int, int]]:
        """(trajectory_id, base_index) for every step allowed by the step filter.

        Discarded trajectories are dropped here for the single-dataset path; the mixture
        handles them through its sampling weights instead.
        """
        all_steps: list[tuple[int, int]] = []
        discarded_episode_indices = []
        if self.discard_bad_trajectories:
            discarded_episode_indices = self._lerobot_info_meta.get("discarded_episode_indices", [])

        for trajectory_id in self.trajectory_ids:
            if trajectory_id in discarded_episode_indices:
                continue
            for base_index in self.step_filter[trajectory_id]:
                all_steps.append((trajectory_id, base_index))
        return all_steps

    def _get_modality_keys(self) -> dict:
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Re-key the delta indices by modality.key instead of by modality."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _get_data_path_pattern(self) -> str:
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        return self.lerobot_info_meta["chunks_size"]

    def _get_tasks(self) -> pd.DataFrame:
        tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
        with open(tasks_path, "r") as f:
            tasks = [json.loads(line) for line in f]
        df = pd.DataFrame(tasks)
        return df.set_index("task_index")

    def _check_integrity(self):
        """Every configured key must exist in `meta/modality.json`."""
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(
                        ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}"
                    )

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Give the transforms the metadata they need (e.g. normalization statistics)."""
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.all_steps)

    def __str__(self) -> str:
        return f"{self.dataset_name} ({len(self)} steps)"

    def __getitem__(self, index: int) -> dict:
        trajectory_id, base_index = self.all_steps[index]
        indices = {
            key: delta_indices + base_index for key, delta_indices in self.delta_indices.items()
        }
        return self.transforms(self.get_step_data(trajectory_id, indices))

    def get_step_data(self, trajectory_id: int, indices: dict[str, np.ndarray]) -> dict:
        """RAW data for one step (no transforms): {modality.key: array of shape (T, ...)}.

        `indices` maps modality.key -> absolute step indices to load; keys absent from it
        are skipped.
        """
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        for modality in self.modality_keys:
            for key in self.modality_keys[modality]:
                if key in indices:
                    data[key] = self.get_data_by_modality(
                        trajectory_id, modality, key, indices[key]
                    )
        return data

    def get_parquet_path(self, trajectory_id: int) -> Path:
        chunk_index = self.get_episode_chunk(trajectory_id)
        return self.dataset_path / self.data_path_pattern.format(
            episode_chunk=chunk_index, episode_index=trajectory_id
        )

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        else:
            parquet_path = self.get_parquet_path(trajectory_id)
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            return pd.read_parquet(parquet_path)

    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Position of `trajectory_id` in `trajectory_ids` (indexes `trajectory_lengths`)."""
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(
                f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}"
            )
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunk_size

    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """`array[step_indices]` with out-of-range indices padded.

        "first_last" repeats the first/last step, "zero" pads with zeros.
        """
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        output = np.zeros(expected_shape)
        output[~padding_positions] = raw_data
        if padding_positions.any():
            if padding_strategy == "first_last":
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        """Video file for a trajectory and video sub-key (without the 'video.' prefix)."""
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        video_filename = self.video_path_pattern.format(
            episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
        )
        return self.dataset_path / video_filename

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """Frames at `step_indices` (clamped to the trajectory), shape (T, H, W, C)."""
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Clamping is equivalent to padding the video with its first/last frame.
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key)
        # Frames are located by the parquet timestamps of the requested steps.
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        video_timestamp = timestamp[step_indices]

        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )

    def _get_state_or_action_array(
        self, trajectory_id: int, modality: str, key: str
    ) -> tuple[np.ndarray, int, str]:
        """Full-trajectory (T, D) array for a state/action sub-key, plus its length and padding strategy.

        Absolute quantities pad with the first/last step, everything else with zeros.
        """
        trajectory_index = self.get_trajectory_index(trajectory_id)
        max_length = self.trajectory_lengths[trajectory_index]

        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        subkey = key.replace(modality + ".", "")
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[subkey].original_key
        if le_key is None:
            le_key = subkey
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        if data_array.ndim == 1:
            assert (
                data_array.shape[0] == max_length
            ), f"Expected 1D array with length {max_length}, got {data_array.shape} array"
            data_array = data_array.reshape(-1, 1)
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[subkey].start,
            le_state_or_action_cfg[subkey].end,
        )
        data_array = data_array[:, le_indices]
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[subkey]
        padding_strategy = "first_last" if state_or_action_cfg.absolute else "zero"
        return data_array, max_length, padding_strategy

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """State or action at `step_indices`, padded when out of range. Shape (T, D)."""
        data_array, max_length, padding_strategy = self._get_state_or_action_array(
            trajectory_id, modality, key
        )
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy=padding_strategy,
        )

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        step_indices: np.ndarray,
    ) -> list[str]:
        """Language annotation at `step_indices` (clamped), one string per step.

        The parquet column holds either task indices (resolved via `meta/tasks.jsonl`)
        or the strings themselves.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        trajectory_index = self.get_trajectory_index(trajectory_id)
        max_length = self.trajectory_lengths[trajectory_index]
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        assert key.startswith(
            "annotation."
        ), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")

        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
            subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        if pd.api.types.is_numeric_dtype(self.curr_traj_data[original_key]):
            task_indices: list[int] = self.curr_traj_data[original_key].iloc[step_indices].tolist()
            return self.tasks.loc[task_indices]["task"].tolist()
        else:
            return self.curr_traj_data[original_key].iloc[step_indices].astype(str).tolist()

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        step_indices: np.ndarray,
    ) -> np.ndarray | list[str]:
        if modality == "video":
            return self.get_video(trajectory_id, key, step_indices)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, step_indices)
        elif modality == "language":
            return self.get_language(trajectory_id, key, step_indices)
        else:
            raise ValueError(f"Invalid modality: {modality}")
