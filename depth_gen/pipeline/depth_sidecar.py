"""UMI depth sidecar server.

Runs in the Fast-FoundationStereo env, separate from the policy env. Loads both hands' stereo
once and serves colored absolute-depth frames over a ZMQ REP socket.

Wire protocol (ZMQ REQ/REP, multipart). Dependency-light: only ``zmq`` +
``msgpack`` + ``numpy`` on both ends (no msgpack_numpy), arrays travel as raw
C-contiguous bytes so no per-frame array (de)serialization cost.

  REQUEST  (5 parts):
    part 0: msgpack header {
        "frame_id": int, "capture_timestamp": float,
        "shape": [480, 640, 3], "dtype": "uint8",
        "order": ["left_cam0", "left_cam1", "right_cam0", "right_cam1"] }
    parts 1..4: raw uint8 RGB bytes for each cam in header["order"].

  REPLY (ok, 3 parts):
    part 0: msgpack header {
        "frame_id": int, "ok": true,
        "shape": [336, 448, 3], "dtype": "uint8",
        "order": ["left_depth_rgb", "right_depth_rgb"],
        "left_valid_fraction", "right_valid_fraction",
        "stereo_ms", "render_ms", "total_ms", "server_ms" }
    parts 1..2: raw uint8 RGB bytes for each depth in header["order"].
  REPLY (error, 1 part): msgpack header {"frame_id", "ok": false, "error": str}

Run (from depth_gen/; UMI_CALIB_SET selects the calibration dir under data/):
    UMI_CALIB_SET=cam_params python -m pipeline.depth_sidecar --port 5701 --backend pytorch1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import zmq
import msgpack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE_HW = (480, 640, 3)
REQ_ORDER = ("left_cam0", "left_cam1", "right_cam0", "right_cam1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5701)
    ap.add_argument("--backend", default="pytorch1", choices=["pytorch1", "tensorrt"])
    ap.add_argument("--bind", default="tcp://0.0.0.0")
    args = ap.parse_args()
    # The calibration must match the grippers mounted on the robot; a silent default would
    # produce plausible-looking but wrong depth, so the deploy path requires an explicit choice.
    if not os.environ.get("UMI_CALIB_SET"):
        ap.error("set UMI_CALIB_SET to the calibration dir under data/ for the mounted grippers")

    from pipeline.depth_common import BatchedDualStereo

    print(f"[depth_sidecar] loading stereo (backend={args.backend})...", flush=True)
    stereo = BatchedDualStereo(backend=args.backend)
    dummy = np.zeros(SOURCE_HW, dtype=np.uint8)
    for _ in range(3):  # warm the engine off the request path
        stereo.process(dummy, dummy, dummy, dummy)
    print("[depth_sidecar] warmed up.", flush=True)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"{args.bind}:{args.port}")
    print(f"[depth_sidecar] REP bound on {args.bind}:{args.port}", flush=True)

    n = 0
    while True:
        parts = sock.recv_multipart()
        t_recv = time.perf_counter()
        fid = -1
        try:
            header = msgpack.unpackb(parts[0], raw=False)
            fid = int(header.get("frame_id", -1))
            shape = tuple(header["shape"])
            if header.get("dtype") != "uint8" or shape != SOURCE_HW:
                raise ValueError(f"bad header shape/dtype: {header.get('dtype')} {shape}")
            if len(parts) != 1 + len(REQ_ORDER):
                raise ValueError(f"expected {1 + len(REQ_ORDER)} parts, got {len(parts)}")
            order = header.get("order", list(REQ_ORDER))
            imgs = {
                name: np.frombuffer(buf, dtype=np.uint8).reshape(shape)
                for name, buf in zip(order, parts[1:])
            }
            left_rgb, right_rgb, meta = stereo.process(
                imgs["left_cam0"], imgs["left_cam1"],
                imgs["right_cam0"], imgs["right_cam1"],
            )
            out_header = {
                "frame_id": fid, "ok": True,
                "shape": list(left_rgb.shape), "dtype": "uint8",
                "order": ["left_depth_rgb", "right_depth_rgb"],
                "server_ms": (time.perf_counter() - t_recv) * 1000.0,
                **meta,
            }
            sock.send_multipart([
                msgpack.packb(out_header, use_bin_type=True),
                np.ascontiguousarray(left_rgb).tobytes(),
                np.ascontiguousarray(right_rgb).tobytes(),
            ])
            n += 1
            if n % 100 == 1:
                print(f"[depth_sidecar] frame {fid}: total={meta['total_ms']:.1f}ms "
                      f"valid L/R={meta['left_valid_fraction']:.2f}/"
                      f"{meta['right_valid_fraction']:.2f}", flush=True)
        except Exception as e:
            print(f"[depth_sidecar] ERROR frame {fid}: {e!r}", flush=True)
            sock.send_multipart([
                msgpack.packb({"frame_id": fid, "ok": False, "error": repr(e)},
                              use_bin_type=True)
            ])


if __name__ == "__main__":
    main()
