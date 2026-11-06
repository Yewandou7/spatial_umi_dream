"""Inference wrapper around a saved checkpoint directory.

A checkpoint directory holds the model weights plus `experiment_cfg/conf.yaml`
(the training hydra config) and `experiment_cfg/metadata.json` (normalization
statistics per embodiment). `WorldActionPolicy` rebuilds the eval-time transform
chain from the config, runs the causal video+action model and maps the
normalized prediction back to absolute robot actions.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from spatial_umi_dream.data.schema import DatasetMetadata, EmbodimentTag
from spatial_umi_dream.data.transform import ComposedModalityTransform
from spatial_umi_dream.data.transform.se3_utils import absolute_pose9
from spatial_umi_dream.model import WorldActionModel

logger = logging.getLogger(__name__)


def _update_tokenizer_path_in_config(cfg, new_path: str) -> None:
    """Update tokenizer_path in the transforms subtree only (avoids trainer.model etc.)."""
    if isinstance(cfg, DictConfig):
        if "tokenizer_path" in cfg:
            cfg.tokenizer_path = new_path
        # Only recurse via "transforms" to avoid triggering resolution of trainer, etc.
        if "transforms" not in cfg:
            return
        sub = cfg.transforms
        if isinstance(sub, DictConfig):
            for v in sub.values():
                _update_tokenizer_path_in_config(v, new_path)
        elif isinstance(sub, ListConfig):
            for v in sub:
                _update_tokenizer_path_in_config(v, new_path)
    elif isinstance(cfg, ListConfig):
        for v in cfg:
            _update_tokenizer_path_in_config(v, new_path)


def unsqueeze_dict_values(data: dict[str, Any]) -> dict[str, Any]:
    """Add a leading batch dimension of size 1 to every value."""
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.array(v)
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        elif isinstance(v, str):
            unsqueezed_data[k] = np.array([v])
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: dict[str, Any]) -> dict[str, Any]:
    """Remove the batch dimension from every value."""
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v)
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze()
        else:
            squeezed_data[k] = v
    return squeezed_data


class WorldActionPolicy:
    """Load a checkpoint and run streaming (KV-cached) inference.

    The model is not reentrant: callers must serialize `infer` calls. Streaming
    state (KV caches, CLIP features, `current_start_frame`) lives on
    `self.model` and carries over across `infer` calls until the
    caller resets it.
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: EmbodimentTag,
        device: str = "cuda",
        tokenizer_path: str | None = None,
    ):
        """
        Args:
            model_path: Checkpoint directory (weights + experiment_cfg/).
            embodiment_tag: Which embodiment's transforms/statistics to use.
            device: Device to run the model on.
            tokenizer_path: Override for the tokenizer path baked into conf.yaml
                (training machines usually store an absolute local path).
        """
        self.embodiment_tag = embodiment_tag
        self.model_path = model_path
        self.device = device
        model_dir = Path(model_path)

        exp_cfg_dir = model_dir / "experiment_cfg"
        train_cfg = OmegaConf.load(exp_cfg_dir / "conf.yaml")
        if tokenizer_path is not None:
            _update_tokenizer_path_in_config(train_cfg, tokenizer_path)
        self.train_cfg = train_cfg

        # 1. Load the model.
        logger.info("Loading model weights from %s", model_path)
        model = WorldActionModel.from_pretrained(model_path)
        model.eval()
        model.requires_grad_(False)

        self.eval_bf16 = self.train_cfg.get("eval_bf16", False)
        if self.eval_bf16:
            model = model.to(dtype=torch.bfloat16)
        model.to(device=device)

        # Post initialize, move RoPE freqs to cuda.
        model.post_initialize()
        torch.cuda.empty_cache()
        self.model = model

        # 2. Normalization statistics and eval transforms. One policy serves
        # one embodiment, so only that embodiment's metadata is loaded.
        with open(exp_cfg_dir / "metadata.json", "r") as f:
            metadatas = json.load(f)
        metadata = DatasetMetadata.model_validate(metadatas[self.embodiment_tag.value])

        # The DiT consumes video at the action head's target resolution; make
        # the transform metadata agree with it. metadata.json may carry a
        # different (dataset) resolution, e.g. 180x320.
        cfg = self.model.config
        target_h, target_w = cfg.target_video_height, cfg.target_video_width
        if target_h is not None and target_w is not None and metadata.modalities.video:
            for key in metadata.modalities.video.keys():
                metadata.modalities.video[key].resolution = (int(target_w), int(target_h))

        assert (
            self.embodiment_tag.value in train_cfg.transforms
        ), f"{self.embodiment_tag.value=}, {train_cfg.transforms.keys()=}"
        eval_transform = instantiate(train_cfg.transforms[self.embodiment_tag.value])
        assert isinstance(eval_transform, ComposedModalityTransform), f"{eval_transform=}"
        eval_transform.set_metadata(metadata)
        eval_transform.eval()
        self.eval_transform = eval_transform

        # 3. Modality configs (video/state keys and horizons the checkpoint expects).
        self.modality_configs = instantiate(
            train_cfg.modality_configs[self.embodiment_tag.value]
        )

        self.relative_action = bool(self.train_cfg.get("relative_action", False))
        self.relative_action_keys = list(self.train_cfg.get("relative_action_keys", []) or [])
        logger.info(
            "Policy ready: embodiment=%s eval_bf16=%s relative_action=%s keys=%s",
            self.embodiment_tag.value, self.eval_bf16, self.relative_action,
            self.relative_action_keys,
        )

    @staticmethod
    def _check_state_is_batched(obs: dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def _unnormalize_actions(
        self, normalized_action: torch.Tensor, obs: dict[str, Any]
    ) -> dict[str, Any]:
        """Unnormalize actions and compose relative actions onto the anchor state."""
        unnormalized_action = self.eval_transform.unapply(
            dict(action=normalized_action.cpu())
        )
        if not (self.relative_action and self.relative_action_keys):
            return unnormalized_action

        for key in self.relative_action_keys:
            action_key = f"action.{key}"
            state_key = f"state.{key}"
            if action_key not in unnormalized_action:
                continue
            if state_key not in obs:
                raise KeyError(
                    f"relative_action key {key!r} needs {state_key!r} in the observation "
                    f"to compose absolute actions; got keys {list(obs)}"
                )
            # Shape is (B, T, D) or (T, D); the anchor is the last timestep.
            last_state = obs[state_key]
            if len(last_state.shape) >= 2:
                last_state = last_state[..., -1, :]

            # Dispatch by sub-key suffix mirrors
            # ShardedLeRobotSE3DeltaSingleDataset._compute_relative_chunk:
            # `_pose` sub-keys went through SE(3) (`relative_pose9`) on the
            # dataset side, so the inverse is `absolute_pose9`
            # (T_abs = T_base @ T_rel). Other sub-keys were additive deltas.
            if key.endswith("_pose"):
                action_arr = unnormalized_action[action_key]
                assert action_arr.shape[-1] == 9, (
                    f"_pose key {action_key} expects 9D action (xyz+rot6d), "
                    f"got shape {action_arr.shape}"
                )
                # action_arr: (B, T, 9). last_state: (B, 9) or (9,).
                if action_arr.ndim != 3:
                    raise ValueError(
                        f"Unexpected action shape {action_arr.shape} for SE(3) unapply"
                    )
                unnormalized_action[action_key] = np.stack([
                    absolute_pose9(
                        action_arr[b],
                        last_state[b] if last_state.ndim >= 2 else last_state,
                    )
                    for b in range(action_arr.shape[0])
                ], axis=0)
            else:
                # Additive inverse (broadcast anchor state across the horizon).
                if len(unnormalized_action[action_key].shape) > len(last_state.shape):
                    last_state = np.expand_dims(last_state, axis=-2)
                unnormalized_action[action_key] = unnormalized_action[action_key] + last_state
        return unnormalized_action

    def infer(
        self, obs: dict[str, Any], unapply_obs: dict[str, Any] | None = None
    ) -> dict[str, np.ndarray]:
        """Run one streaming step and return unnormalized actions per sub-key.

        Args:
            obs: `video.*` uint8 arrays (T, H, W, 3), `state.*` float arrays
                (T, D) and `annotation.task` str; a leading batch dimension is
                added when absent and removed again from the output.
            unapply_obs: Optional `state.*` values that replace `obs` entries
                for the relative-action composition only. Used when the model
                is fed a transformed state (cross-arm relative pose) while
                T_abs = T_state @ T_rel must use the world-frame proprio.

        Returns:
            dict mapping `action.<key>` to an absolute action chunk array.
        """
        transform_start_time = time.perf_counter()

        # Snapshot the observation before the transforms mutate it; the
        # relative-action composition needs the raw state.
        anchor_obs = {
            k: v.copy() if isinstance(v, np.ndarray) else v.clone() if torch.is_tensor(v) else v
            for k, v in obs.items()
        }
        if unapply_obs is not None:
            for k, v in unapply_obs.items():
                anchor_obs[k] = (
                    v.copy() if isinstance(v, np.ndarray)
                    else v.clone() if torch.is_tensor(v)
                    else v
                )

        obs = dict(obs)
        is_batched = self._check_state_is_batched(obs)
        if not is_batched:
            obs = unsqueeze_dict_values(obs)
            anchor_obs = unsqueeze_dict_values(anchor_obs)

        normalized_input = self.eval_transform(obs)
        transform_time = time.perf_counter() - transform_start_time

        for k, v in normalized_input.items():
            if torch.is_tensor(v) and v.dtype == torch.float32 and self.eval_bf16:
                normalized_input[k] = v.to(dtype=torch.bfloat16)

        model_start_time = time.perf_counter()
        with torch.inference_mode():
            model_pred = self.model.predict_chunk(normalized_input)
        normalized_action = model_pred["action_pred"].float()
        model_time = time.perf_counter() - model_start_time

        untransform_start_time = time.perf_counter()
        actions = self._unnormalize_actions(normalized_action, anchor_obs)
        if not is_batched:
            actions = squeeze_dict_values(actions)
        untransform_time = time.perf_counter() - untransform_start_time

        logger.info(
            "Inference time: total %.3fs (transform %.3fs, model %.3fs, untransform %.3fs)",
            transform_time + model_time + untransform_time,
            transform_time, model_time, untransform_time,
        )
        return actions
