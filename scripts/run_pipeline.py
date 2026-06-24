#!/usr/bin/env python3
"""Fast in-memory pipeline: YOLO26-seg -> MLP seed painting -> PointPillar (no intermediate files)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PAPER_REF_ROOT = Path(__file__).resolve().parents[1]
if str(PAPER_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_REF_ROOT))

from src.fast_pipeline import FastPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PAPER_REF_ROOT / "sample_data")
    parser.add_argument("--scene", default="example_scene")
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--score-thr", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--save-detections",
        type=Path,
        default=None,
        help="Optional final JSON output path (detections only; no intermediate artifacts)",
    )
    args = parser.parse_args()

    pipeline = FastPipeline(
        data_root=args.data_root,
        device=args.device,
    )
    result = pipeline.run_frame(args.scene, args.frame_idx, score_thr=args.score_thr)

    n_virt = int(result.paint.is_virtual.sum())
    print(f"Scene {args.scene} frame {args.frame_idx}")
    print(f"YOLO masks: {result.inline_mask_counts}")
    print(f"Real points: {result.paint.xyz_real.shape[0]}  painted: {n_virt}")
    print(
        "Timings (s): "
        f"yolo_rgb={result.timings.yolo_rgb_s:.3f} "
        f"yolo_thermal={result.timings.yolo_thermal_s:.3f} "
        f"yolo_wall={result.timings.yolo_wall_s:.3f} "
        f"paint={result.timings.paint_s:.3f} "
        f"detect={result.timings.detect_s:.3f} "
        f"total={result.timings.total_s:.3f}"
    )

    if result.detections is None:
        raise SystemExit("PointPillar returned no detections (detector not loaded).")

    det = result.detections
    n = len(det.scores_3d)
    print(f"Detections (>={args.score_thr:.2f}): {n}")
    for score, box in sorted(zip(det.scores_3d, det.bboxes_3d), reverse=True)[:10]:
        print(
            f"  score={score:.3f} center=({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}) "
            f"size=({box[3]:.2f}, {box[4]:.2f}, {box[5]:.2f})"
        )
    if args.save_detections:
        args.save_detections.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "labels_3d": det.labels_3d,
            "scores_3d": det.scores_3d,
            "bboxes_3d": det.bboxes_3d,
            "box_type_3d": det.box_type_3d,
            "class_names": list(det.class_names),
        }
        args.save_detections.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.save_detections}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
