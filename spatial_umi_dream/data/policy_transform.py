"""Policy input transform and collator.

``PolicyInputTransform`` is the last step of the per-embodiment
``ComposedModalityTransform``: it assembles the ordered camera views into the
single video canvas the video model sees, builds the layout-aware prompt and
pads state/action to the register widths. ``PolicyCollator`` stacks the
samples and tokenizes the prompts.
"""

import ast
import html
import os
from typing import Any, Dict, List, Optional

import ftfy
import numpy as np
from pydantic import Field, PrivateAttr
import regex as re
import torch
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorMixin
import tree

from spatial_umi_dream.data.schema import DatasetMetadata, EmbodimentTag
from spatial_umi_dream.data.transform.base import InvertibleModalityTransform
from spatial_umi_dream.data.canvas import (
    assemble_rgbd_canvas,
    robotwin_prompt_with_layout,
    umi_prompt_with_layout,
)


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class HuggingfaceTokenizer:
    """umt5 tokenizer with Wan's text cleaning, padded/truncated to ``seq_len``."""

    def __init__(self, name, seq_len):
        self.name = name
        self.seq_len = seq_len
        # A local checkpoint directory must not be validated as a hub repo id.
        load_kwargs = {"local_files_only": True} if os.path.isdir(name) else {}
        self.tokenizer = AutoTokenizer.from_pretrained(name, **load_kwargs)

    def __call__(self, sequences: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(input_ids, attention_mask)`` for a list of prompts."""
        sequences = [whitespace_clean(basic_clean(u)) for u in sequences]
        ids = self.tokenizer(
            sequences,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.seq_len,
            add_special_tokens=True,
        )
        return ids.input_ids, ids.attention_mask


def _prompt_with_layout(text: str, elem: dict) -> str:
    """Wrap the task instruction with a description of the video canvas layout."""
    if bool(elem.get("is_robotwin", False)):
        return robotwin_prompt_with_layout(text)
    return umi_prompt_with_layout(text)


def collate(features: List[dict], tokenizer: HuggingfaceTokenizer) -> dict:
    batch = {}
    keys = features[0].keys()

    for key in keys:
        if key == "text":
            output_values = []
            for elem in features:
                item = elem[key]
                # Some LeRobot language columns arrive as the string representation
                # of a one-element list; unwrap it before adding the layout prompt.
                try:
                    parsed_item = ast.literal_eval(item)
                    if isinstance(parsed_item, (list, tuple)):
                        processed_item = str(parsed_item[0])
                    else:
                        processed_item = str(parsed_item)
                    output_values.append(_prompt_with_layout(processed_item, elem))
                except (ValueError, SyntaxError, TypeError):
                    output_values.append(_prompt_with_layout(str(item), elem))
            ids, mask = tokenizer(output_values)
            batch[key] = ids
            batch['text_attention_mask'] = mask
        else:
            values = [elem[key] for elem in features]
            shapes = [getattr(v, "shape", type(v).__name__) for v in values]
            if len({tuple(s) if hasattr(s, "__iter__") else s for s in shapes}) > 1:
                raise ValueError(
                    f"collate shape mismatch: key={key!r} per-sample shapes={shapes}"
                )
            batch[key] = torch.from_numpy(np.stack(values))
    return batch


class PolicyCollator(DataCollatorMixin):
    def __init__(self, tokenizer_path: str = "google/umt5-xxl", max_length: int = 512):
        super().__init__()
        self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=max_length)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate(features, self.tokenizer)


class PolicyInputTransform(InvertibleModalityTransform):

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )

    always_use_default_instruction: bool = Field(
        default=False,
        description="Whether to always use the default instruction. For studying how much the language helps.",
    )

    # Language key found in the sample, set by check_keys_and_batch_size.
    _language_key: Optional[str] = PrivateAttr(default=None)

    default_instruction: str
    max_action_dim: int
    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None
    action_horizon: int

    # State register. When use_state is False (default) the transform does not
    # emit state/state_mask and the action head + DiT run without state tokens.
    use_state: bool = False
    max_state_dim: int = 32
    state_horizon: int = 1

    # Expected size of the assembled [RGB | depth] canvas; checked, never resized to.
    canvas_height: int | None = None
    canvas_width: int | None = None

    tokenizer_path: str = Field(
        default="google/umt5-xxl",
        description="Path to the tokenizer."
    )
    _tokenizer: Optional[HuggingfaceTokenizer] = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tokenizer = HuggingfaceTokenizer(name=self.tokenizer_path, seq_len=self.max_length)

    @property
    def tokenizer(self):
        return self._tokenizer

    def set_metadata(
        self, dataset_metadata: DatasetMetadata
    ):
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def check_keys_and_batch_size(self, data):
        language_keys = [key for key in data if "annotation" in key]
        self._language_key = language_keys[0] if language_keys else None
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")
        return is_batched, batch_size

    def _prepare_video(self, data: dict) -> np.ndarray:
        """Assemble the four ordered views of data['video'] ([T, 4, H, W, C]) into one [T, H, W, C] canvas."""
        if self.embodiment_tag not in (EmbodimentTag.UMI, EmbodimentTag.ROBOTWIN):
            raise ValueError(f"Embodiment {self.embodiment_tag} is not supported.")
        # Only the UMI canvas is asserted against the configured size; the RoboTwin server packs its
        # own panel geometry and the model resizes the canvas to the target size.
        check = self.embodiment_tag == EmbodimentTag.UMI
        return assemble_rgbd_canvas(
            data["video"],
            target_height=self.canvas_height if check else None,
            target_width=self.canvas_width if check else None,
        )

    def _prepare_language(self, data: dict) -> str:
        """Return the task instruction (or default_instruction if missing)."""
        if self._language_key is None:
            language = self.default_instruction
        else:
            language = data[self._language_key]
            if isinstance(language, np.ndarray):
                language = language.item() if language.size == 1 else language[0]
            if isinstance(language, list):
                language = language[0]

        if self.always_use_default_instruction:
            language = self.default_instruction
        return language

    def _prepare_state(self, data: dict):
        """
        Gather state from data['state'], pad to max_state_dim, return masks.

        data['state'] is `(state_horizon * num_chunks, sub_D)` produced by the
        upstream ConcatTransform from the per-anchor state sub-keys. We pad the
        channel dim to max_state_dim and emit a boolean state_mask marking the
        real (non-padding) dims.
        """
        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            return state, state_mask

        state = data["state"]
        assert state.shape[0] % self.state_horizon == 0, (
            f"{state.shape=}, {self.state_horizon=}"
        )

        n_state_dims = state.shape[-1]
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            state = np.pad(
                state,
                ((0, 0), (0, self.max_state_dim - n_state_dims)),
                "constant",
            )

        state_mask = np.zeros_like(state).astype(bool)
        state_mask[:, :n_state_dims] = True
        return state, state_mask

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            return actions, actions_mask

        actions = data["action"]
        assert actions.shape[0] % self.action_horizon == 0, f"{actions.shape=}, {self.action_horizon=}"

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask

    def apply_single(self, data: dict) -> dict:
        transformed_data = {}

        # 1) Video canvas and layout-aware prompt.
        images = self._prepare_video(data).astype(np.uint8)
        language = self._prepare_language(data)
        # Format here as well as in the collator so direct eval/deployment calls receive the same
        # layout-aware prompt; the helpers are idempotent when the collator sees the string again.
        if self.embodiment_tag == EmbodimentTag.ROBOTWIN:
            language = robotwin_prompt_with_layout(language)
        else:
            language = umi_prompt_with_layout(language)

        if self.use_state:
            state, state_mask = self._prepare_state(data)
            transformed_data["state"] = state
            transformed_data["state_mask"] = state_mask

        if self.training:
            # 2) Actions
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask

        transformed_data["images"] = images
        transformed_data["text"] = language

        transformed_data["is_robotwin"] = np.array(
            self.embodiment_tag == EmbodimentTag.ROBOTWIN,
            dtype=bool,
        )

        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension, process each element, collate.
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return collate(data_split_processed, self.tokenizer)

    def apply(self, data: dict) -> dict:
        if not self.training and data["video"].ndim == 5:
            data["video"] = data["video"][None, ...]
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)
