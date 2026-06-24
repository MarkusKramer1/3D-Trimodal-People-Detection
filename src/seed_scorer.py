"""Learned seed-instance association: score LiDAR seeds as person vs contamination.

Replaces the hand-tuned nearest-depth-cluster heuristic in point painting
with a classifier over the YOLO-predicted instance masks. Masks come from the
detector (no 2D->3D matching); a seed is positive iff its 3D point falls inside
any cohort GT box, so false-positive masks yield all-negative seeds and the model
learns to suppress them. Segmentation evidence from the RGB and thermal masks is
fused at the LiDAR-point level (``xmod_*`` features): both cameras project the
same sweep, so a point is cross-checked against the other modality's masks with no
instance association. Feature extraction is shared between the training dataset
builder and inference inside the painting path.
"""

from __future__ import annotations

import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

SEED_FEATURE_NAMES = [
    "z_cam",
    "z_minus_min",
    "z_minus_median",
    "z_robust_score",
    "cluster_size_frac",
    "cluster_median_offset",
    "in_nearest_cluster",
    "n_depth_clusters",
    "height_above_ground",
    "u_rel",
    "v_rel",
    "boundary_dist_norm",
    "log_mask_area",
    "mask_aspect",
    "support_3d_frac",
    "support_depth_frac",
    "log_n_seeds",
    "depth_rank",
    "uv_knn_depth_diff",
    "uv_knn_depth_spread",
    "mask_fill_ratio",
    "seed_density",
    "lidar_intensity",
    "is_thermal",
    "n_competing_masks",
    "z_minus_competitor",
    "competitor_overlap_frac",
    "claimed_by_competitor",
    # Detector confidence of this instance's own (YOLO) mask. 1.0 for GT masks.
    "yolo_conf",
    # Cross-modal evidence: does the *same* LiDAR point also fall inside a mask
    # of the other camera (rgb<->thermal), and how confident/interior is it?
    # The two cameras project the identical LiDAR sweep, so this association is
    # exact and needs no instance matching. NaN when the point is in no
    # other-modality mask (the model gets a dedicated NaN-indicator column).
    "xmod_in_mask",
    "xmod_conf",
    "xmod_boundary_dist_norm",
    "n_xmod_masks",
]

_DEPTH_CLUSTER_EPS_M = 0.35
_UV_KNN_K = 5
_CLAIM_DEPTH_TOL_M = 0.35
_SUPPORT_3D_R_M = 0.5
_SUPPORT_DEPTH_R_M = 0.3

# Per-frame cache for mask distance transforms (cleared by ``clear_mask_geometry_cache``).
_MASK_GEOM_CACHE: dict[
    int, tuple[float, float, float, float, float, float, float, float, np.ndarray, float]
] = {}


def clear_mask_geometry_cache() -> None:
    """Drop cached ``distanceTransform`` maps before painting a new frame."""
    _MASK_GEOM_CACHE.clear()


def _mask_geometry(
    mask: np.ndarray,
) -> tuple[float, float, float, float, float, float, float, float, np.ndarray, float]:
    """Bounding box, area, distance map, and area scale for one instance mask."""
    key = int(mask.ctypes.data)
    cached = _MASK_GEOM_CACHE.get(key)
    if cached is not None:
        return cached
    m = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(m)
    area = float(ys.size)
    if area > 0:
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        w_box = max(x1 - x0, 1.0)
        h_box = max(y1 - y0, 1.0)
    else:
        x0 = y0 = x1 = y1 = 0.0
        w_box = h_box = 1.0
    dist_map = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    area_scale = max(np.sqrt(max(area, 1.0)), 1.0)
    fill = area / max(w_box * h_box, 1.0)
    out = (area, x0, y0, x1, y1, w_box, h_box, area_scale, dist_map, fill)
    _MASK_GEOM_CACHE[key] = out
    return out


# Depth-stratified cap per instance (0 = unlimited). Keeps scoring fast on huge masks.
_DEFAULT_MAX_SEEDS_PER_INSTANCE = 256


def depth_stratified_subsample_indices(z: np.ndarray, max_n: int) -> np.ndarray:
    """Pick up to ``max_n`` seeds spread across the depth range (stable, deterministic)."""
    n = int(np.asarray(z).size)
    if max_n <= 0 or n <= max_n:
        return np.arange(n, dtype=np.int64)
    order = np.argsort(np.asarray(z, dtype=np.float64), kind="stable")
    picks = np.linspace(0, n - 1, int(max_n))
    return order[np.unique(np.rint(picks).astype(np.int64))]


def _depth_cluster_labels_1d(z: np.ndarray, eps: float) -> np.ndarray:
    """Single-linkage 1D depth clusters (same cut as ``fclusterdata`` on one axis)."""
    n = int(z.size)
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.int64)
    order = np.argsort(z, kind="stable")
    zs = z[order]
    gap_break = np.concatenate([[False], np.diff(zs) > float(eps)])
    lab_sorted = np.cumsum(gap_break, dtype=np.int64) + 1
    labels = np.empty(n, dtype=np.int64)
    labels[order] = lab_sorted
    return labels


def _support_depth_fraction(z: np.ndarray, n: int) -> np.ndarray:
    """Fraction of seeds within ``_SUPPORT_DEPTH_R_M`` in camera depth (O(n log n))."""
    z = np.asarray(z, dtype=np.float64)
    zs = np.sort(z)
    lo = np.searchsorted(zs, z - _SUPPORT_DEPTH_R_M, side="left")
    hi = np.searchsorted(zs, z + _SUPPORT_DEPTH_R_M, side="right")
    return (hi - lo).astype(np.float64) / float(n)


def _support_fractions(
    xyz: np.ndarray,
    z: np.ndarray,
    n: int,
    *,
    fast: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Local 3D / depth neighbour fractions per seed."""
    support_depth = _support_depth_fraction(z, n)
    if fast:
        return support_depth, support_depth
    inv_n = 1.0 / float(n)
    support_3d = np.asarray(
        cKDTree(np.asarray(xyz, dtype=np.float64)).query_ball_point(
            xyz, r=_SUPPORT_3D_R_M, return_length=True
        ),
        dtype=np.float64,
    )
    support_3d *= inv_n
    return support_3d, support_depth


def _uv_knn_depth_stats(seed_uv: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Depth agreement with ``_UV_KNN_K`` nearest seeds in image space."""
    n = int(z.shape[0])
    if n < 2:
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
    k_eff = min(_UV_KNN_K, n - 1)
    _dists, nn_idx = cKDTree(seed_uv).query(seed_uv, k=k_eff + 1)
    z_nn = z[np.asarray(nn_idx[:, 1:], dtype=np.int64)]
    return np.abs(z - np.median(z_nn, axis=1)), z_nn.std(axis=1)


def precompute_mask_geom_for_uv(
    mask: np.ndarray,
    seed_uv: np.ndarray,
) -> dict[str, Any]:
    """Mask scalars + per-seed boundary distance (one ``distanceTransform`` per mask)."""
    m = np.asarray(mask, dtype=bool)
    area, x0, y0, _x1, _y1, w_box, h_box, area_scale, dist_map, mask_fill_ratio = _mask_geometry(m)
    uu = np.clip(np.rint(seed_uv[:, 0]).astype(np.int64), 0, m.shape[1] - 1)
    vv = np.clip(np.rint(seed_uv[:, 1]).astype(np.int64), 0, m.shape[0] - 1)
    boundary_dist = dist_map[vv, uu].astype(np.float64)
    return {
        "area": area,
        "x0": x0,
        "y0": y0,
        "w_box": w_box,
        "h_box": h_box,
        "area_scale": area_scale,
        "mask_fill_ratio": mask_fill_ratio,
        "boundary_dist": boundary_dist,
    }


def compute_seed_features(
    seed_uv: np.ndarray,
    seed_z: np.ndarray,
    seed_xyz: np.ndarray,
    mask: np.ndarray,
    ground_z_at_seed: np.ndarray,
    *,
    intensity: np.ndarray | None = None,
    is_thermal: float = 0.0,
    competitors: list[tuple[np.ndarray, float]] | None = None,
    yolo_conf: float = 1.0,
    xmod_in_mask: np.ndarray | None = None,
    xmod_conf: np.ndarray | None = None,
    xmod_boundary_norm: np.ndarray | None = None,
    xmod_count: np.ndarray | None = None,
    fast: bool = True,
    mask_geom: dict[str, Any] | None = None,
) -> np.ndarray:
    """Per-seed feature matrix ``(N, len(SEED_FEATURE_NAMES))`` for one instance.

    ``seed_uv``: (N, 2) rectified pixel coords; ``seed_z``: (N,) camera depth;
    ``seed_xyz``: (N, 3) lidar-frame points; ``mask``: full instance mask (H, W);
    ``ground_z_at_seed``: (N,) ground plane height at each seed's xy;
    ``intensity``: (N,) LiDAR return intensity or ``None`` (NaN = missing, e.g.
    temporally borrowed seeds); ``is_thermal``: 1.0 when the mask comes from the
    thermal camera; ``competitors``: other instances in the same image as
    ``(mask, median_seed_depth)`` — used for explaining-away when one person
    occludes another (a seed whose depth matches a competitor's consensus depth
    likely belongs to the competitor).

    Fusion inputs (point-level rgb<->thermal): ``yolo_conf`` is the detector
    confidence of this instance's own mask (1.0 for GT). ``xmod_*`` describe,
    per seed, whether the *same* LiDAR point also lands in a mask of the other
    camera — ``xmod_in_mask`` (1.0/0.0), ``xmod_conf`` (best other-modality
    detector confidence, NaN if none), ``xmod_boundary_norm`` (how interior the
    point is in that other mask, NaN if none), ``xmod_count`` (number of
    other-modality masks covering it). Each defaults to "no cross-modal
    evidence" when ``None`` so the function stays usable in single-modality
    contexts.
    """
    n = int(seed_z.shape[0])
    if n == 0:
        return np.zeros((0, len(SEED_FEATURE_NAMES)), dtype=np.float64)

    z = np.asarray(seed_z, dtype=np.float64)
    z_min = float(z.min())
    z_med = float(np.median(z))
    q25, q75 = np.percentile(z, [25.0, 75.0])
    iqr = max(float(q75 - q25), 1e-3)

    # Depth clustering mirrors reject_contaminated_seeds so the model sees the
    # same structure the heuristic acts on (1D single-linkage, eps = 0.35 m).
    labels = _depth_cluster_labels_1d(z, _DEPTH_CLUSTER_EPS_M)
    uniq, inv, cnt = np.unique(labels, return_inverse=True, return_counts=True)
    cluster_medians = np.array([float(np.median(z[labels == lab])) for lab in uniq])
    nearest_i = int(np.argmin(cluster_medians))
    nearest_lab = int(uniq[nearest_i])
    min_cluster_median = float(cluster_medians[nearest_i])
    cluster_size = cnt[inv].astype(np.float64) / float(n)
    cluster_median_per_seed = cluster_medians[inv]

    if mask_geom is not None:
        area = float(mask_geom["area"])
        x0 = float(mask_geom["x0"])
        y0 = float(mask_geom["y0"])
        w_box = float(mask_geom["w_box"])
        h_box = float(mask_geom["h_box"])
        area_scale = float(mask_geom["area_scale"])
        mask_fill_ratio = float(mask_geom["mask_fill_ratio"])
        boundary_dist = np.asarray(mask_geom["boundary_dist"], dtype=np.float64).reshape(-1)
    else:
        m = np.asarray(mask, dtype=bool)
        area, x0, y0, x1, y1, w_box, h_box, area_scale, dist_map, mask_fill_ratio = _mask_geometry(m)
        uu = np.clip(np.rint(seed_uv[:, 0]).astype(np.int64), 0, m.shape[1] - 1)
        vv = np.clip(np.rint(seed_uv[:, 1]).astype(np.int64), 0, m.shape[0] - 1)
        boundary_dist = dist_map[vv, uu].astype(np.float64)

    xyz = np.asarray(seed_xyz, dtype=np.float64).reshape(-1, 3)
    support_3d, support_depth = _support_fractions(xyz, z, n, fast=fast)

    height_above_ground = xyz[:, 2] - np.asarray(ground_z_at_seed, dtype=np.float64)

    # Local pixel-neighborhood depth agreement: contamination shows up as a seed
    # whose depth disagrees with the seeds around it in the image.
    depth_rank = np.argsort(np.argsort(z)).astype(np.float64) / max(float(n - 1), 1.0)
    if fast:
        uv_knn_depth_diff = np.zeros(n, dtype=np.float64)
        uv_knn_depth_spread = np.zeros(n, dtype=np.float64)
    else:
        uv_knn_depth_diff, uv_knn_depth_spread = _uv_knn_depth_stats(seed_uv, z)

    seed_density = float(n) / max(area, 1.0)
    if intensity is None:
        inten = np.full(n, np.nan)
    else:
        inten = np.asarray(intensity, dtype=np.float64).reshape(-1)

    # Cross-instance explaining-away: who else claims these pixels/depths?
    n_competing = np.zeros(n, dtype=np.float64)
    z_minus_comp = np.full(n, np.nan)
    comp_overlap_frac = 0.0
    claimed = np.zeros(n, dtype=np.float64)
    if not fast and competitors and area > 0:
        m = np.asarray(mask, dtype=bool)
        ys, xs = np.nonzero(m)
        if ys.size:
            x1, y1 = float(xs.max()), float(ys.max())
        else:
            x1 = y1 = 0.0
        iy0, iy1 = int(y0), int(y1) + 1
        ix0, ix1 = int(x0), int(x1) + 1
        own_box = m[iy0:iy1, ix0:ix1]
        best_overlap = 0.0
        best_med: float | None = None
        uu = np.clip(np.rint(seed_uv[:, 0]).astype(np.int64), 0, m.shape[1] - 1)
        vv = np.clip(np.rint(seed_uv[:, 1]).astype(np.int64), 0, m.shape[0] - 1)
        for comp_mask, comp_med in competitors:
            comp_box = np.asarray(comp_mask, dtype=bool)[iy0:iy1, ix0:ix1]
            if not comp_box.any():
                continue
            overlap = float(np.logical_and(own_box, comp_box).sum()) / area
            in_comp = np.asarray(comp_mask, dtype=bool)[vv, uu]
            n_competing += in_comp
            if np.isfinite(comp_med):
                claimed = np.maximum(
                    claimed,
                    (np.abs(z - comp_med) <= _CLAIM_DEPTH_TOL_M).astype(np.float64),
                )
            if overlap > best_overlap:
                best_overlap = overlap
                best_med = comp_med if np.isfinite(comp_med) else None
        comp_overlap_frac = best_overlap
        if best_med is not None:
            z_minus_comp = z - best_med

    # Cross-modal evidence columns. Absent inputs collapse to "no other-modality
    # mask": membership 0, count 0, confidence/boundary NaN (NaN-indicator aware).
    def _col(arr: np.ndarray | None, fill: float) -> np.ndarray:
        if arr is None:
            return np.full(n, fill, dtype=np.float64)
        return np.asarray(arr, dtype=np.float64).reshape(-1)[:n]

    xmod_in = _col(xmod_in_mask, 0.0)
    xmod_c = _col(xmod_conf, np.nan)
    xmod_b = _col(xmod_boundary_norm, np.nan)
    xmod_n = _col(xmod_count, 0.0)

    feats = np.column_stack(
        [
            z,
            z - z_min,
            z - z_med,
            (z - z_med) / iqr,
            cluster_size,
            cluster_median_per_seed - min_cluster_median,
            (labels == nearest_lab).astype(np.float64),
            np.full(n, float(uniq.size)),
            height_above_ground,
            (seed_uv[:, 0] - x0) / w_box,
            (seed_uv[:, 1] - y0) / h_box,
            boundary_dist / area_scale,
            np.full(n, np.log(max(area, 1.0))),
            np.full(n, h_box / w_box),
            support_3d,
            support_depth,
            np.full(n, np.log(float(n))),
            depth_rank,
            uv_knn_depth_diff,
            uv_knn_depth_spread,
            np.full(n, mask_fill_ratio),
            np.full(n, seed_density),
            inten,
            np.full(n, float(is_thermal)),
            n_competing,
            z_minus_comp,
            np.full(n, comp_overlap_frac),
            claimed,
            np.full(n, float(yolo_conf)),
            xmod_in,
            xmod_c,
            xmod_b,
            xmod_n,
        ]
    )
    return feats


class NumpyMLPScorer:
    """Pickled MLP with the same ``predict_proba`` contract as the sklearn GBT.

    Owns its preprocessing (median imputation + NaN indicator columns +
    standardization) so callers pass the raw feature matrix, exactly as for the
    tree model. Forward pass is plain numpy (ReLU hidden layers, sigmoid out).
    """

    def __init__(
        self,
        weights: list[tuple[np.ndarray, np.ndarray]],
        *,
        nan_cols: np.ndarray,
        median: np.ndarray,
        mu: np.ndarray,
        sd: np.ndarray,
        quantiles: np.ndarray | None = None,
    ) -> None:
        self.weights = [(np.asarray(w, np.float32), np.asarray(b, np.float32)) for w, b in weights]
        self.nan_cols = np.asarray(nan_cols, np.int64)
        self.median = np.asarray(median, np.float32)
        self.mu = np.asarray(mu, np.float32)
        self.sd = np.asarray(sd, np.float32)
        # (F, Q) per-feature quantile grid for rank normalization; None = plain
        # standardization (kept optional for pickle compatibility with v1).
        self.quantiles = None if quantiles is None else np.asarray(quantiles, np.float64)

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        ind = np.isnan(X[:, self.nan_cols]).astype(np.float32)
        Xi = np.where(np.isnan(X), self.median, X)
        q = self.quantiles
        if q is not None:
            grid = np.linspace(0.0, 1.0, q.shape[1], dtype=np.float32)
            for j in range(Xi.shape[1]):
                xj = Xi[:, j]
                qj = q[j]
                hi = np.searchsorted(qj, xj, side="right")
                hi = np.clip(hi, 1, qj.shape[0] - 1)
                lo = hi - 1
                denom = np.maximum(qj[hi] - qj[lo], np.float64(1e-12))
                t = (xj - qj[lo]) / denom
                Xi[:, j] = grid[lo] + t.astype(np.float32) * (grid[hi] - grid[lo])
        scaled = (Xi - self.mu) / self.sd
        return np.hstack([scaled.astype(np.float32), ind])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        h = self._preprocess(X)
        for i, (w, b) in enumerate(self.weights):
            h = h @ w.T + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0.0)
        p = 1.0 / (1.0 + np.exp(-h[:, 0]))
        return np.column_stack([1.0 - p, p])


_SCORER_CACHE: dict[str, Any] = {}


def load_seed_scorer(model_path: str | Path) -> Any | None:
    """Load (and cache) a pickled scorer payload ``{"model", "feature_names", ...}``."""
    key = str(model_path)
    if key in _SCORER_CACHE:
        return _SCORER_CACHE[key]
    p = Path(model_path)
    if not p.is_file():
        _SCORER_CACHE[key] = None
        return None
    with open(p, "rb") as fp:
        payload = pickle.load(fp)
    _SCORER_CACHE[key] = payload
    return payload


def score_seeds(payload: Any, features: np.ndarray) -> np.ndarray:
    """Probability that each seed belongs to the person instance.

    Models trained on an older (subset) feature list keep working: columns are
    selected by name from the current ``SEED_FEATURE_NAMES`` layout.
    """
    if features.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    names = payload.get("feature_names")
    if names and list(names) != SEED_FEATURE_NAMES:
        idx = [SEED_FEATURE_NAMES.index(n) for n in names]
        features = features[:, idx]
    model = payload["model"]
    return model.predict_proba(features)[:, 1]


def score_seed_batches(payload: Any, batches: list[np.ndarray]) -> list[np.ndarray]:
    """Score multiple per-instance feature matrices in one MLP forward pass."""
    if not batches:
        return []
    sizes = [int(b.shape[0]) for b in batches]
    if sum(sizes) == 0:
        return [np.zeros(0, dtype=np.float64) for _ in batches]
    probs = score_seeds(payload, np.vstack(batches))
    out: list[np.ndarray] = []
    i = 0
    for n in sizes:
        out.append(probs[i : i + n])
        i += n
    return out


def filter_scored_indices(
    probs: np.ndarray,
    *,
    threshold: float,
    min_seeds: int,
    use_topk: bool,
) -> np.ndarray | None:
    """Boolean keep mask for scored seeds; ``None`` when policy rejects the instance."""
    n = int(probs.shape[0])
    if n == 0:
        return None
    keep = probs >= float(threshold)
    if int(np.sum(keep)) < int(min_seeds):
        if not use_topk:
            return None
        k = min(int(min_seeds), n)
        top = np.argsort(-probs)[:k]
        keep = np.zeros(n, dtype=bool)
        keep[top] = True
    return keep


def filter_scored_seeds(
    raw_seeds: list,
    probs: np.ndarray,
    *,
    threshold: float,
    min_seeds: int,
    use_topk: bool,
) -> list | None:
    """Apply threshold / top-k keep policy to scored seeds."""
    if not raw_seeds:
        return None
    keep = filter_scored_indices(
        probs,
        threshold=threshold,
        min_seeds=min_seeds,
        use_topk=use_topk,
    )
    if keep is None:
        return None
    return [s for s, k in zip(raw_seeds, keep) if k]


def featurize_seed_batches_parallel(
    jobs: list[dict[str, Any]],
    *,
    max_workers: int | None = None,
    fast: bool = True,
) -> list[np.ndarray]:
    """Build per-instance feature matrices; parallel when ``len(jobs) > 8``."""

    def _one(job: dict[str, Any]) -> np.ndarray:
        kw = dict(job)
        kw.setdefault("fast", fast)
        return compute_seed_features(**kw)

    if len(jobs) <= 8:
        return [_one(j) for j in jobs]
    workers = max_workers
    if workers is None:
        workers = min(8, max(1, (os.cpu_count() or 4) - 1), len(jobs))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, jobs))


def save_seed_scorer(
    model_path: str | Path,
    model: Any,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload = {
        "model": model,
        "feature_names": list(SEED_FEATURE_NAMES),
        "metadata": metadata or {},
    }
    p = Path(model_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as fp:
        pickle.dump(payload, fp)
