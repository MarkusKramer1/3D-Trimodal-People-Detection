#!/usr/bin/env python3
"""Run the point painting pipeline on sample data and visualize results interactively."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PAPER_REF_ROOT = Path(__file__).resolve().parents[1]
if str(PAPER_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_REF_ROOT))

from src.demo_viz import show_demo_window  # noqa: E402
from src.fast_pipeline import FastPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PAPER_REF_ROOT / "sample_data")
    parser.add_argument("--scene", default="example_scene")
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--score-thr", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-points",
        type=int,
        default=120_000,
        help="Max LiDAR / virtual points drawn in the 3D viewer",
    )
    args = parser.parse_args()

    print(f"Loading models and running pipeline ({args.scene}, frame {args.frame_idx}) …")
    pipeline = FastPipeline(
        data_root=args.data_root,
        device=args.device,
    )
    result = pipeline.run_frame(args.scene, args.frame_idx, score_thr=args.score_thr)

    if result.detections is None:
        raise SystemExit(
            "PointPillar detector not available. Install dependencies:\n"
            "  bash scripts/setup_conda_env.sh"
        )

    n_virt = int(result.paint.is_virtual.sum())
    n_det = len(result.detections.scores_3d)
    print(
        f"Done — real={result.paint.xyz_real.shape[0]} virtual={n_virt} "
        f"detections={n_det} total={result.timings.total_s:.2f}s"
    )

    show_demo_window(
        images=result.images,
        inline_masks=result.inline_masks,
        xyz_real=result.paint.xyz_real,
        xyz_virtual=result.paint.xyz_virtual,
        detections=result.detections,
        score_thr=args.score_thr,
        max_points=args.max_points,
        window_title=f"{args.scene} frame {args.frame_idx} — point painting demo",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
