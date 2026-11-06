from typing import Any, Callable, Literal

from einops import rearrange
import numpy as np
from pydantic import Field, PrivateAttr, field_validator
import torch
import torchvision.transforms.v2 as T

from spatial_umi_dream.data.schema import DatasetMetadata
from spatial_umi_dream.data.transform.base import ModalityTransform


_INTERPOLATION_MODES: dict[str, T.InterpolationMode] = {
    "nearest": T.InterpolationMode.NEAREST,
    "linear": T.InterpolationMode.BILINEAR,
    "cubic": T.InterpolationMode.BICUBIC,
    "lanczos4": T.InterpolationMode.LANCZOS,
    "nearest_exact": T.InterpolationMode.NEAREST_EXACT,
}


class VideoTransform(ModalityTransform):
    """Base class for torchvision-backed video transforms.

    Subclasses implement ``get_transform`` for the train and eval modes. The
    views listed in ``apply_to`` are concatenated along the frame axis before
    the torchvision transform runs, so random parameters (crop offset, color
    jitter factors) are shared across views and frames of one sample.
    """

    # Model variables
    _train_transform: Callable | None = PrivateAttr(default=None)
    _eval_transform: Callable | None = PrivateAttr(default=None)
    _original_resolutions: dict[str, tuple[int, int]] = PrivateAttr(default_factory=dict)

    @property
    def train_transform(self) -> Callable:
        assert (
            self._train_transform is not None
        ), "Transform is not set. Please call set_metadata() before calling apply()."
        return self._train_transform

    @train_transform.setter
    def train_transform(self, value: Callable):
        self._train_transform = value

    @property
    def eval_transform(self) -> Callable | None:
        return self._eval_transform

    @eval_transform.setter
    def eval_transform(self, value: Callable | None):
        self._eval_transform = value

    @property
    def original_resolutions(self) -> dict[str, tuple[int, int]]:
        assert (
            self._original_resolutions is not None
        ), "Original resolutions are not set. Please call set_metadata() before calling apply()."
        return self._original_resolutions

    @original_resolutions.setter
    def original_resolutions(self, value: dict[str, tuple[int, int]]):
        self._original_resolutions = value

    def check_input(self, data: dict[str, Any]):
        for key in self.apply_to:
            assert isinstance(data[key], torch.Tensor), f"Video {key} is not a torch tensor"
            assert data[key].ndim in [
                4,
                5,
            ], f"Expected video {key} to have 4 or 5 dimensions (T, C, H, W or T, B, C, H, W), got {data[key].ndim}"

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        super().set_metadata(dataset_metadata)
        self.original_resolutions = {}
        for key in self.apply_to:
            split_keys = key.split(".")
            assert len(split_keys) == 2, f"Invalid key: {key}. Expected format: modality.key"
            sub_key = split_keys[1]
            if sub_key in dataset_metadata.modalities.video:
                self.original_resolutions[key] = dataset_metadata.modalities.video[
                    sub_key
                ].resolution
            else:
                raise ValueError(
                    f"Video key {sub_key} not found in dataset metadata. Available keys: {dataset_metadata.modalities.video.keys()}"
                )
        train_transform = self.get_transform(mode="train")
        eval_transform = self.get_transform(mode="eval")
        assert train_transform is not None, "Train transform must be set"
        self.train_transform = train_transform
        self.eval_transform = eval_transform

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.training:
            transform = self.train_transform
        else:
            transform = self.eval_transform
            if transform is None:
                return data
        try:
            self.check_input(data)
        except AssertionError as e:
            raise ValueError(
                f"Input data does not match the expected format for {self.__class__.__name__}: {e}"
            ) from e

        # Concatenate views
        views = [data[key] for key in self.apply_to]
        num_views = len(views)
        is_batched = views[0].ndim == 5
        bs = views[0].shape[0] if is_batched else 1
        # VideoToTensor receives numpy uint8 frames; every later transform receives tensors.
        if isinstance(views[0], torch.Tensor):
            views = torch.cat(views, 0)
        else:
            views = np.concatenate(views, 0)
        if is_batched:
            views = rearrange(views, "(v b) t c h w -> (v b t) c h w", v=num_views, b=bs)
        views = transform(views)
        # Split views
        if is_batched:
            views = rearrange(views, "(v b t) c h w -> v b t c h w", v=num_views, b=bs)
        else:
            views = rearrange(views, "(v t) c h w -> v t c h w", v=num_views)
        for key, view in zip(self.apply_to, views):
            data[key] = view
        return data

    @classmethod
    def _validate_interpolation(cls, interpolation: str):
        if interpolation not in _INTERPOLATION_MODES:
            raise ValueError(f"Interpolation mode {interpolation} not supported")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        raise NotImplementedError(
            "set_transform is not implemented for VideoTransform. Please implement this function to set the transforms."
        )


class VideoCrop(VideoTransform):
    height: int | None = Field(default=None, description="The height of the input image")
    width: int | None = Field(default=None, description="The width of the input image")
    scale: float = Field(
        ...,
        description="The scale of the crop. The crop size is (width * scale, height * scale)",
    )

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Random crop in train mode, center crop of the same size in eval mode."""
        # 1. Check the input resolution
        assert (
            len(set(self.original_resolutions.values())) == 1
        ), f"All video keys must have the same resolution, got: {self.original_resolutions}"
        if self.height is None:
            assert self.width is None, "Height and width must be either both provided or both None"
            self.width, self.height = self.original_resolutions[self.apply_to[0]]
        else:
            assert (
                self.width is not None
            ), "Height and width must be either both provided or both None"
        # 2. Create the transform
        size = (int(self.height * self.scale), int(self.width * self.scale))
        if mode == "train":
            return T.RandomCrop(size)
        elif mode == "eval":
            return T.CenterCrop(size)
        else:
            raise ValueError(f"Crop mode {mode} not supported")

    def check_input(self, data: dict[str, Any]):
        super().check_input(data)
        # Check the input resolution
        for key in self.apply_to:
            height, width = data[key].shape[-2:]
            assert (
                height == self.height and width == self.width
            ), f"Video {key} has invalid shape {height, width}, expected {self.height, self.width}"


class VideoResize(VideoTransform):
    height: int = Field(..., description="The height of the resize")
    width: int = Field(..., description="The width of the resize")
    interpolation: str = Field(default="linear", description="The interpolation mode")
    antialias: bool = Field(default=True, description="Whether to apply antialiasing")

    @field_validator("interpolation")
    def validate_interpolation(cls, v):
        cls._validate_interpolation(v)
        return v

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Same resize for train and eval."""
        interpolation = _INTERPOLATION_MODES[self.interpolation]
        size = (self.height, self.width)
        return T.Resize(size, interpolation=interpolation, antialias=self.antialias)


class VideoColorJitter(VideoTransform):
    brightness: float | tuple[float, float] = Field(
        ..., description="The brightness of the color jitter"
    )
    contrast: float | tuple[float, float] = Field(
        ..., description="The contrast of the color jitter"
    )
    saturation: float | tuple[float, float] = Field(
        ..., description="The saturation of the color jitter"
    )
    hue: float | tuple[float, float] = Field(..., description="The hue of the color jitter")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Color jitter in train mode, no-op in eval mode."""
        if mode == "eval":
            return None
        return T.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
        )


class VideoRandomGrayscale(VideoTransform):
    p: float = Field(..., description="The probability of the grayscale transformation")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Random grayscale in train mode, no-op in eval mode."""
        if mode == "eval":
            return None
        return T.RandomGrayscale(self.p)


class VideoRandomPosterize(VideoTransform):
    bits: int = Field(..., description="The number of bits to posterize the image")
    p: float = Field(..., description="The probability of the posterize transformation")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Random posterize in train mode, no-op in eval mode."""
        if mode == "eval":
            return None
        return T.RandomPosterize(bits=self.bits, p=self.p)


class VideoToTensor(VideoTransform):
    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Same conversion for train and eval."""
        return self.__class__.to_tensor

    def check_input(self, data: dict):
        """Check if the input data has the correct shape.
        Expected video shape: [T, H, W, C], dtype np.uint8
        """
        for key in self.apply_to:
            assert key in data, f"Key {key} not found in data. Available keys: {data.keys()}"
            assert data[key].ndim in [
                4,
                5,
            ], f"Video {key} must have 4 or 5 dimensions, got {data[key].ndim}"
            assert (
                data[key].dtype == np.uint8
            ), f"Video {key} must have dtype uint8, got {data[key].dtype}"
            input_resolution = data[key].shape[-3:-1][::-1]
            if key in self.original_resolutions:
                expected_resolution = self.original_resolutions[key]
            else:
                expected_resolution = input_resolution
            assert (
                input_resolution == expected_resolution
            ), f"Video {key} has invalid resolution {input_resolution}, expected {expected_resolution}. Full shape: {data[key].shape}"

    @staticmethod
    def to_tensor(frames: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor efficiently.

        Args:
            frames: numpy array of shape [T, H, W, C] in uint8 format
        Returns:
            tensor of shape [T, C, H, W] in range [0, 1]
        """
        frames = torch.from_numpy(frames)
        frames = frames.to(torch.float32) / 255.0
        return frames.permute(0, 3, 1, 2)  # [T, C, H, W]


class VideoToNumpy(VideoTransform):
    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Same conversion for train and eval."""
        return self.__class__.to_numpy

    @staticmethod
    def to_numpy(frames: torch.Tensor) -> np.ndarray:
        """Convert tensor back to numpy array efficiently.

        Args:
            frames: tensor of shape [T, C, H, W] in range [0, 1]
        Returns:
            numpy array of shape [T, H, W, C] in uint8 format
        """
        frames = (frames.permute(0, 2, 3, 1) * 255).to(torch.uint8)
        return frames.cpu().numpy()


class VideoNormalize(VideoTransform):
    mean: list[float] = Field(..., description="Mean for normalization")
    std: list[float] = Field(..., description="Standard deviation for normalization")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Same normalization for train and eval."""
        return T.Normalize(mean=self.mean, std=self.std)

    def check_input(self, data: dict):
        for key in self.apply_to:
            assert key in data, f"Key {key} not found in data"
            assert isinstance(data[key], torch.Tensor), f"Video {key} is not a torch tensor"
            assert data[key].ndim in [4, 5], f"Video {key} must have 4 or 5 dimensions, got {data[key].ndim}"
            assert data[key].dtype == torch.float32, f"Video {key} must be float32, got {data[key].dtype}"
            assert data[key].min() >= 0.0 and data[key].max() <= 1.0, (
                f"Video {key} must be in [0,1] range before normalization"
            )
