"""Paper-reference seed painting configuration (YOLO26-seg + fused MLP scorer)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PAPER_REF_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PAPER_REF_ROOT / "configs" / "seed_painting.yaml"


@dataclass
class VirtualPaintConfig:
    """Fixed inference recipe: RGB+thermal YOLO masks → seed painting → 5-D features."""

    mask_erosion_px: int = 3
    seed_outlier_threshold_m: float = 0.35
    min_seeds: int = 3
    max_range_m: float = 40.0
    feature_layout: str = "xyz+intensity+is_virtual"
    seed_cluster_eps_m: float = 0.35
    dedup_voxel_m: float = 0.005
    rng_seed: int = 0
    exclude_ground_points: bool = True
    ground_clearance_m: float = 0.08
    seed_scorer_model: str = ""
    seed_scorer_threshold: float = 0.5
    seed_scorer_topk_fallback: bool = True
    seed_scorer_fast_path: bool = True
    seed_scorer_max_seeds_per_instance: int = 256
    seed_scorer_thermal_from_rgb: bool = True


def load_experiment_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def config_to_virtual_paint(cfg: dict[str, Any]) -> VirtualPaintConfig:
    paint = dict(cfg.get("paint") or {})
    model_path = str(paint.get("seed_scorer_model", ""))
    if model_path and not Path(model_path).is_absolute():
        model_path = str((PAPER_REF_ROOT / model_path).resolve())
    return VirtualPaintConfig(
        mask_erosion_px=int(paint.get("mask_erosion_px", 3)),
        seed_outlier_threshold_m=float(paint.get("seed_outlier_threshold_m", 0.35)),
        min_seeds=int(paint.get("min_seeds", 3)),
        max_range_m=float(paint.get("max_range_m", 40.0)),
        feature_layout=str(paint.get("feature_layout", "xyz+intensity+is_virtual")),
        seed_cluster_eps_m=float(paint.get("seed_cluster_eps_m", 0.35)),
        dedup_voxel_m=float(paint.get("dedup_voxel_m", 0.005)),
        rng_seed=int(paint.get("rng_seed", 0)),
        exclude_ground_points=bool(paint.get("exclude_ground_points", True)),
        ground_clearance_m=float(paint.get("ground_clearance_m", 0.08)),
        seed_scorer_model=model_path,
        seed_scorer_threshold=float(paint.get("seed_scorer_threshold", 0.5)),
        seed_scorer_topk_fallback=bool(paint.get("seed_scorer_topk_fallback", True)),
        seed_scorer_fast_path=bool(paint.get("seed_scorer_fast_path", True)),
        seed_scorer_max_seeds_per_instance=int(
            paint.get("seed_scorer_max_seeds_per_instance", 256)
        ),
        seed_scorer_thermal_from_rgb=bool(paint.get("seed_scorer_thermal_from_rgb", True)),
    )
