from typing import Any

import numpy as np
from pydantic import Field, PrivateAttr
import torch

from spatial_umi_dream.data.schema import DatasetMetadata, StateActionMetadata
from spatial_umi_dream.data.transform.base import InvertibleModalityTransform


class Normalizer:
    valid_modes = ["q99", "min_max"]

    def __init__(self, mode: str, statistics: dict, identity_dims: list[int] | None = None):
        self.mode = mode
        self.statistics = statistics
        for key, value in self.statistics.items():
            self.statistics[key] = torch.tensor(value)
        # Dims listed here bypass normalization in both forward and inverse
        # (e.g. rot6d components, which are already bounded in [-1, 1]).
        self.identity_dims = identity_dims if identity_dims is not None else []
        if self.identity_dims:
            dim = next(iter(self.statistics.values())).shape[-1]
            assert all(0 <= d < dim for d in self.identity_dims), (
                f"identity_dims {self.identity_dims} out of range for stats dim {dim}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"

        # Normalize the tensor
        if self.mode == "q99":

            # Range of q99 is [-1, 1]
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)

            # In the case of q01 == q99, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = q01 != q99
            normalized = torch.zeros_like(x)

            # Normalize the values where q01 != q99
            # Formula: 2 * (x - q01) / (q99 - q01) - 1
            normalized[..., mask] = (x[..., mask] - q01[..., mask]) / (
                q99[..., mask] - q01[..., mask]
            )
            normalized[..., mask] = 2 * normalized[..., mask] - 1

            # Set the normalized values to the original values where q01 == q99
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

            # Clip the normalized values to be between -1 and 1
            normalized = torch.clamp(normalized, -1, 1)

        elif self.mode == "min_max":
            # Range of min_max is [-1, 1]
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)

            # In the case of min == max, the normalization will be undefined
            # So we set the normalized values to 0
            mask = min != max
            normalized = torch.zeros_like(x)

            # Normalize the values where min != max
            # Formula: 2 * (x - min) / (max - min) - 1
            normalized[..., mask] = (x[..., mask] - min[..., mask]) / (
                max[..., mask] - min[..., mask]
            )
            normalized[..., mask] = 2 * normalized[..., mask] - 1

            normalized[..., ~mask] = 0

            if not (torch.all(normalized <= 1 + 1e-5) and torch.all(normalized >= -1 - 1e-5)):
                print(
                    f"WARNING: normalization out of range: {x.shape=}, max: {torch.max(normalized)}, min: {torch.min(normalized)}"
                )

            # Clip the normalized values to be between -1 and 1
            normalized = torch.clamp(normalized, -1, 1)

        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")

        if self.identity_dims:
            normalized[..., self.identity_dims] = x[..., self.identity_dims]

        return normalized

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        if self.mode == "q99":
            q01 = self.statistics["q01"].to(dtype=x.dtype, device=x.device)
            q99 = self.statistics["q99"].to(dtype=x.dtype, device=x.device)
            denormalized = (x + 1) / 2 * (q99 - q01) + q01
        elif self.mode == "min_max":
            min = self.statistics["min"].to(dtype=x.dtype, device=x.device)
            max = self.statistics["max"].to(dtype=x.dtype, device=x.device)
            denormalized = (x + 1) / 2 * (max - min) + min
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")

        if self.identity_dims:
            denormalized[..., self.identity_dims] = x[..., self.identity_dims]

        return denormalized


class StateActionToTensor(InvertibleModalityTransform):
    """
    Transforms states and actions to tensors.
    """

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(
                value, np.ndarray
            ), f"Unexpected input type: {type(value)}. Expected type: {np.ndarray}"
            data[key] = torch.from_numpy(value)
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(
                value, torch.Tensor
            ), f"Unexpected input type: {type(value)}. Expected type: {torch.Tensor}"
            data[key] = value.numpy()
        return data


class StateActionTransform(InvertibleModalityTransform):
    """
    Class for state or action normalization.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        normalization_modes (dict[str, str]): The normalization modes for each state key.
            If a state key in apply_to is not present in the dictionary, it will not be normalized.
    """

    # Configurable attributes
    apply_to: list[str] = Field(..., description="The keys in the modality to load and transform.")
    normalization_modes: dict[str, str] = Field(
        default_factory=dict, description="The normalization modes for each state key."
    )
    normalization_statistics: dict[str, dict] = Field(
        default_factory=dict, description="The statistics for each state key."
    )
    modality_metadata: dict[str, StateActionMetadata] = Field(
        default_factory=dict, description="The modality metadata for each state key."
    )
    relative_state: bool = Field(
        default=False,
        description=(
            "When true, sub-keys listed in relative_state_keys bypass the "
            "configured normalization mode because their values were already "
            "computed as cross-arm relative poses in the dataloader and the "
            "absolute q01/q99 stats no longer apply."
        ),
    )
    relative_state_keys: list[str] = Field(
        default_factory=list,
        description=(
            "Bare sub-keys (e.g. 'left_pose') for which to skip normalization "
            "when relative_state=True. Matching is done on the trailing part "
            "of the full key (after '<modality>.')."
        ),
    )
    identity_normalization_dims: dict[str, list[int]] = Field(
        default_factory=dict,
        description=(
            "Per-key dims that bypass normalization in both apply and unapply "
            "(e.g. rot6d dims of pose keys, which are unit-vector components "
            "already bounded in [-1, 1])."
        ),
    )

    # Model variables
    _normalizers: dict[str, Normalizer] = PrivateAttr(default_factory=dict)
    _input_dtypes: dict[str, torch.dtype] = PrivateAttr(default_factory=dict)

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        dataset_statistics = dataset_metadata.statistics
        modality_metadata = dataset_metadata.modalities

        # Check that all state keys specified in apply_to have their modality_metadata
        for key in self.apply_to:
            split_key = key.split(".")
            assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
            if key not in self.modality_metadata:
                modality, state_key = split_key
                assert hasattr(modality_metadata, modality), f"{modality} config not found"
                assert state_key in getattr(
                    modality_metadata, modality
                ), f"{state_key} config not found in {modality}"
                self.modality_metadata[key] = getattr(modality_metadata, modality)[state_key]

        # Check that all state keys specified in normalization_modes have their statistics in state_statistics
        for key in self.normalization_modes:
            split_key = key.split(".")
            assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
            modality, state_key = split_key
            assert hasattr(dataset_statistics, modality), f"{modality} statistics not found"
            assert state_key in getattr(
                dataset_statistics, modality
            ), f"{state_key} statistics not found"
            assert (
                len(getattr(modality_metadata, modality)[state_key].shape) == 1
            ), f"{getattr(modality_metadata, modality)[state_key].shape=}"
            self.normalization_statistics[key] = getattr(dataset_statistics, modality)[
                state_key
            ].model_dump()

        # Initialize the normalizers
        for key in self.normalization_modes:
            modality, state_key = key.split(".")
            # Cross-arm relative state: skip normalizer construction for the
            # listed pose sub-keys (their absolute q01/q99 stats are not valid
            # for the relative values produced upstream).
            if self.relative_state and state_key in self.relative_state_keys:
                continue
            if not self.modality_metadata[key].continuous:
                raise ValueError(f"{key} is not continuous and cannot be normalized")
            self._normalizers[key] = Normalizer(
                mode=self.normalization_modes[key],
                statistics=self.normalization_statistics[key],
                identity_dims=self.identity_normalization_dims.get(key),
            )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                # We allow some keys to be missing in the data, and only process the keys that are present
                continue
            if key not in self._input_dtypes:
                input_dtype = data[key].dtype
                assert isinstance(
                    input_dtype, torch.dtype
                ), f"Unexpected input dtype: {input_dtype}. Expected type: {torch.dtype}"
                self._input_dtypes[key] = input_dtype
            else:
                assert (
                    data[key].dtype == self._input_dtypes[key]
                ), f"All states corresponding to the same key must be of the same dtype, input dtype: {data[key].dtype}, expected dtype: {self._input_dtypes[key]}"
            state = data[key]
            if key in self._normalizers:
                state = self._normalizers[key].forward(state)
            data[key] = state
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            state = data[key]
            assert isinstance(
                state, torch.Tensor
            ), f"Unexpected state type: {type(state)}. Expected type: {torch.Tensor}"
            if key in self._normalizers:
                state = self._normalizers[key].inverse(state)
            # Only convert back to the original dtype if it's known, i.e. `apply` was called before
            # If not, we don't know the original dtype, so we don't convert
            if key in self._input_dtypes:
                state = state.to(self._input_dtypes[key])
            data[key] = state
        return data
