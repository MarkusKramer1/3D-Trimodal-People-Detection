#!/usr/bin/env python3
"""Render the demo visualization to a PNG (no interactive window)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PAPER_REF_ROOT = Path(__file__).resolve().parents[1]
if str(PAPER_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_REF_ROOT))

from src.demo_viz import save_demo_screenshot  # noqa: E402
from src.fast_pipeline import FastPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PAPER_REF_ROOT / "docs" / "demo_screenshot.png")
    parser.add_argument("--data-root", type=Path, default=PAPER_REF_ROOT / "sample_data")
    parser.add_argument("--scene", default="example_scene")
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--score-thr", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-points", type=int, default=120_000)
    args = parser.parse_args()

    pipeline = FastPipeline(data_root=args.data_root, device=args.device, prefer_tensorrt=False)
    result = pipeline.run_frame(args.scene, args.frame_idx, score_thr=args.score_thr)
    if result.detections is None:
        raise SystemExit("PointPillar detector not loaded.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_demo_screenshot(
        args.output,
        images=result.images,
        inline_masks=result.inline_masks,
        xyz_real=result.paint.xyz_real,
        xyz_virtual=result.paint.xyz_virtual,
        detections=result.detections,
        score_thr=args.score_thr,
        max_points=args.max_points,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
