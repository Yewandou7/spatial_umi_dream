from __future__ import annotations

import os
from pathlib import Path

import torch


# Engines are specific to the GPU architecture and TensorRT version. Prefer an sm_120 build if
# present; FFS_TRT_ENGINE overrides the path.
_ENGINE_DIR = (
    Path(__file__).resolve().parents[1]
    / "weights"
    / "fast_foundation_stereo_trt_448x352"
)
_ENV_ENGINE = os.environ.get("FFS_TRT_ENGINE")
if _ENV_ENGINE:
    FAST_TRT_ENGINE = Path(_ENV_ENGINE)
elif (_ENGINE_DIR / "fast_foundationstereo_sm120.engine").is_file():
    FAST_TRT_ENGINE = _ENGINE_DIR / "fast_foundationstereo_sm120.engine"
else:
    FAST_TRT_ENGINE = _ENGINE_DIR / "fast_foundationstereo.engine"


def _trt_to_torch_dtype(trt, dtype) -> torch.dtype:
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT8: torch.int8,
        trt.DataType.BOOL: torch.bool,
    }
    if dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]


class FastFoundationStereoTensorRT:
    """Fixed-shape runner for the repository's single FP16 TensorRT engine."""

    def __init__(self, engine_path: Path = FAST_TRT_ENGINE):
        import tensorrt as trt

        if not engine_path.is_file():
            raise FileNotFoundError(engine_path)
        self.trt = trt
        self.engine_path = engine_path
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(
            engine_path.read_bytes()
        )
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        self.input_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
            == trt.TensorIOMode.INPUT
        ]
        output_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
            == trt.TensorIOMode.OUTPUT
        ]
        if self.input_names != ["left_image", "right_image"] or output_names != [
            "disparity"
        ]:
            raise RuntimeError(
                f"Unexpected engine tensors: inputs={self.input_names}, "
                f"outputs={output_names}"
            )
        self.input_shapes = {
            name: tuple(self.engine.get_tensor_shape(name))
            for name in self.input_names
        }
        self.output_name = output_names[0]
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        output_dtype = _trt_to_torch_dtype(
            trt, self.engine.get_tensor_dtype(self.output_name)
        )
        self.output = torch.empty(
            output_shape, device="cuda", dtype=output_dtype
        )
        self.context.set_tensor_address(
            self.output_name, int(self.output.data_ptr())
        )
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406], device="cuda", dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225], device="cuda", dtype=torch.float32
        ).view(1, 3, 1, 1)

    def _bind_input(self, name: str, tensor: torch.Tensor) -> None:
        expected_shape = self.input_shapes[name]
        expected_dtype = _trt_to_torch_dtype(
            self.trt, self.engine.get_tensor_dtype(name)
        )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} expects {expected_shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != expected_dtype:
            raise TypeError(f"{name} expects {expected_dtype}, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        self.context.set_tensor_address(name, int(tensor.data_ptr()))

    @torch.inference_mode()
    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left = ((left / 255.0 - self.mean) / self.std).contiguous()
        right = ((right / 255.0 - self.mean) / self.std).contiguous()
        self._bind_input("left_image", left)
        self._bind_input("right_image", right)

        caller_stream = torch.cuda.current_stream()
        left.record_stream(self.stream)
        right.record_stream(self.stream)
        self.stream.wait_stream(caller_stream)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        caller_stream.wait_stream(self.stream)
        return self.output

    @property
    def settings(self) -> dict[str, object]:
        return {
            "backend": "tensorrt",
            "engine": str(self.engine_path),
            "tensorrt": self.trt.__version__,
            "engine_input_shape": list(self.input_shapes["left_image"]),
            "precision": "FP16 allowed",
            "external_normalization": "ImageNet",
        }
