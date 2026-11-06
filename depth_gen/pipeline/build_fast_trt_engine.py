from __future__ import annotations

import argparse
from pathlib import Path

import tensorrt as trt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the repository's single-ONNX Fast-FoundationStereo engine."
    )
    parser.add_argument("onnx", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--workspace-gb", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    args.engine.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(args.onnx)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {args.onnx}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, args.workspace_gb * 1024**3
    )
    if not builder.platform_has_fast_fp16:
        raise RuntimeError("Selected GPU does not advertise fast FP16 support")
    config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    args.engine.write_bytes(bytes(serialized))
    print(f"Built FP16 engine: {args.engine}")


if __name__ == "__main__":
    main()
