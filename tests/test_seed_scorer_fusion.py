"""Tests for the fused (YOLO + point-level RGB<->thermal) seed scorer features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PAPER_REF_ROOT = Path(__file__).resolve().parents[1]
if str(PAPER_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_REF_ROOT))

from src.seed_scorer import SEED_FEATURE_NAMES, compute_seed_features, score_seeds  # noqa: E402
from src.virtual_point_painting import SweepProjection, build_cross_modal_membership  # noqa: E402


def _toy_instance(n: int = 6):
    h, w = 40, 30
    mask = np.zeros((h, w), dtype=bool)
    mask[10:30, 8:22] = True
    rng = np.random.default_rng(0)
    seed_uv = np.column_stack([rng.uniform(9, 21, n), rng.uniform(11, 29, n)])
    seed_z = np.linspace(10.0, 11.0, n)
    seed_xyz = np.column_stack([seed_z, np.zeros(n), np.full(n, 0.5)])
    ground_z = np.full(n, -1.0)
    return seed_uv, seed_z, seed_xyz, mask, ground_z


def test_feature_vector_has_fusion_columns() -> None:
    for name in (
        "yolo_conf",
        "xmod_in_mask",
        "xmod_conf",
        "xmod_boundary_dist_norm",
        "n_xmod_masks",
    ):
        assert name in SEED_FEATURE_NAMES
    seed_uv, seed_z, seed_xyz, mask, ground_z = _toy_instance()
    feats = compute_seed_features(seed_uv, seed_z, seed_xyz, mask, ground_z)
    assert feats.shape == (seed_z.shape[0], len(SEED_FEATURE_NAMES))


def test_score_seeds_selects_columns_by_name() -> None:
    class _PickCol0:
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            p = X[:, 0]
            return np.column_stack([1.0 - p, p])

    names = ["z_cam", "yolo_conf"]
    payload = {"model": _PickCol0(), "feature_names": names}
    full = np.zeros((3, len(SEED_FEATURE_NAMES)))
    zc = SEED_FEATURE_NAMES.index("z_cam")
    full[:, zc] = [0.1, 0.2, 0.3]
    out = score_seeds(payload, full)
    np.testing.assert_allclose(out, [0.1, 0.2, 0.3])
