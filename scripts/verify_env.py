#!/usr/bin/env python3
"""Verify that the PaintFusion environment can import and run the pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PAPER_REF_ROOT = Path(__file__).resolve().parents[1]
if str(PAPER_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_REF_ROOT))


def _check(label: str, fn) -> None:
    try:
        fn()
    except Exception as exc:
        raise SystemExit(f"[FAIL] {label}: {exc}") from exc
    print(f"[OK] {label}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cuda:0 or cpu")
    parser.add_argument("--run-pipeline", action="store_true", help="Run one fast-pipeline frame")
    args = parser.parse_args()

    import numpy as np  # noqa: F401
    import torch

    from src.mmdet3d_paths import resolve_mmdet3d_root

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    print(f"NumPy: {np.__version__}")

    if int(np.__version__.split(".")[0]) >= 2:
        raise AssertionError(f"need numpy 1.x, got {np.__version__}")
    print("[OK] numpy<2")
    _check("ultralytics", lambda: __import__("ultralytics"))
    _check("mmengine", lambda: __import__("mmengine"))
    _check("mmcv", lambda: __import__("mmcv"))
    _check("mmdet", lambda: __import__("mmdet"))
    _check("mmdet3d", lambda: __import__("mmdet3d"))
    _check("src.fast_pipeline", lambda: __import__("src.fast_pipeline"))
    _check("src.pointpillar_infer", lambda: __import__("src.pointpillar_infer"))

    mmdet3d_root = resolve_mmdet3d_root()
    print(f"[OK] MMDetection3D submodule at {mmdet3d_root}")

    config_path = PAPER_REF_ROOT / "configs" / "paintfusion.py"
    if not config_path.is_file():
        raise SystemExit(f"[FAIL] PaintFusion config: missing {config_path}")
    print(f"[OK] PaintFusion config at {config_path}")

    if args.run_pipeline:
        from src.fast_pipeline import FastPipeline  # noqa: E402

        pipeline = FastPipeline(
            data_root=PAPER_REF_ROOT / "sample_data",
            device=args.device,
        )
        result = pipeline.run_frame("example_scene", 0, score_thr=0.1)
        assert result.detections is not None
        n_det = len(result.detections.scores_3d)
        print(
            f"[OK] pipeline frame: painted={int(result.paint.is_virtual.sum())} "
            f"detections={n_det} total_s={result.timings.total_s:.3f}"
        )

    print("Environment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
