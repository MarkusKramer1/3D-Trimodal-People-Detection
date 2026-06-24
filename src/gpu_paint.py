"""GPU batched seed featurization + MLP scoring for seed painting."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.seed_scorer import (
    NumpyMLPScorer,
    SEED_FEATURE_NAMES,
    _DEPTH_CLUSTER_EPS_M,
    _SUPPORT_DEPTH_R_M,
    compute_seed_features,
    precompute_mask_geom_for_uv,
)

_DEPTH_CLUSTER_EPS = float(_DEPTH_CLUSTER_EPS_M)
_SUPPORT_DEPTH_R = float(_SUPPORT_DEPTH_R_M)


def parse_paint_device(device: str | None) -> torch.device | None:
    if not device or str(device).lower().startswith("cpu"):
        return None
    if not torch.cuda.is_available():
        return None
    return torch.device(device)


def gpu_paint_enabled(device: str | None, cfg: Any) -> bool:
    return parse_paint_device(device) is not None and bool(cfg.seed_scorer_fast_path)


class TorchMLPScorer:
    """Torch inference mirror of ``NumpyMLPScorer``."""

    def __init__(self, scorer: NumpyMLPScorer, device: torch.device) -> None:
        self.device = device
        self.nan_cols = torch.as_tensor(scorer.nan_cols, dtype=torch.long, device=device)
        self.median = torch.as_tensor(scorer.median, dtype=torch.float32, device=device)
        self.mu = torch.as_tensor(scorer.mu, dtype=torch.float32, device=device)
        self.sd = torch.as_tensor(scorer.sd, dtype=torch.float32, device=device)
        self.weights = [
            (
                torch.as_tensor(w, dtype=torch.float32, device=device),
                torch.as_tensor(b, dtype=torch.float32, device=device),
            )
            for w, b in scorer.weights
        ]
        self.quantiles = (
            None
            if scorer.quantiles is None
            else torch.as_tensor(scorer.quantiles, dtype=torch.float64, device=device)
        )
        self.feature_names = list(SEED_FEATURE_NAMES)

    @classmethod
    def from_payload(cls, payload: Any, device: torch.device) -> TorchMLPScorer:
        model = payload["model"]
        if not isinstance(model, NumpyMLPScorer):
            raise TypeError(f"Expected NumpyMLPScorer, got {type(model)}")
        inst = cls(model, device)
        names = payload.get("feature_names")
        if names:
            inst.feature_names = list(names)
        return inst

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        ind = torch.isnan(x[:, self.nan_cols]).to(dtype=torch.float32)
        xi = torch.where(torch.isnan(x), self.median, x)
        q = self.quantiles
        if q is not None:
            grid = torch.linspace(0.0, 1.0, q.shape[1], dtype=torch.float32, device=self.device)
            for j in range(xi.shape[1]):
                xj = xi[:, j]
                qj = q[j].contiguous()
                hi = torch.searchsorted(qj, xj.to(dtype=torch.float64), right=True).clamp(1, qj.shape[0] - 1)
                lo = hi - 1
                denom = torch.clamp(qj[hi] - qj[lo], min=1e-12)
                t = ((xj.to(dtype=torch.float64) - qj[lo]) / denom).to(dtype=torch.float32)
                xi[:, j] = grid[lo] + t * (grid[hi] - grid[lo])
        scaled = (xi - self.mu) / self.sd
        return torch.cat([scaled, ind], dim=1)

    def predict_proba_pos(self, x: torch.Tensor) -> torch.Tensor:
        h = self._preprocess(x)
        for i, (w, b) in enumerate(self.weights):
            h = h @ w.T + b
            if i < len(self.weights) - 1:
                h = torch.relu(h)
        return torch.sigmoid(h[:, 0])


_SCORER_CACHE: dict[str, TorchMLPScorer] = {}


def _torch_scorer(payload: Any, device: torch.device) -> TorchMLPScorer:
    key = f"{id(payload['model'])}:{device}"
    if key not in _SCORER_CACHE:
        _SCORER_CACHE[key] = TorchMLPScorer.from_payload(payload, device)
    return _SCORER_CACHE[key]


def _feature_column_indices(payload: Any) -> list[int] | None:
    names = payload.get("feature_names")
    if names and list(names) != SEED_FEATURE_NAMES:
        return [SEED_FEATURE_NAMES.index(n) for n in names]
    return None


def _depth_cluster_labels_1d_torch(z: torch.Tensor, eps: float) -> torch.Tensor:
    n = int(z.numel())
    if n <= 1:
        return torch.ones(max(n, 1), dtype=torch.long, device=z.device)
    order = torch.argsort(z, stable=True)
    zs = z[order]
    gap_break = torch.cat(
        [torch.zeros(1, dtype=torch.bool, device=z.device), torch.diff(zs) > eps]
    )
    lab_sorted = torch.cumsum(gap_break, dim=0).to(torch.long) + 1
    labels = torch.empty(n, dtype=torch.long, device=z.device)
    labels[order] = lab_sorted
    return labels


def _support_depth_fraction_torch(z: torch.Tensor, n: int) -> torch.Tensor:
    zs, _ = torch.sort(z)
    lo = torch.searchsorted(zs, z - _SUPPORT_DEPTH_R, side="left")
    hi = torch.searchsorted(zs, z + _SUPPORT_DEPTH_R, side="right")
    return (hi - lo).to(dtype=torch.float32) / float(n)


def _col_torch(
    arr: np.ndarray | None,
    n: int,
    fill: float,
    device: torch.device,
) -> torch.Tensor:
    if arr is None:
        return torch.full((n,), fill, dtype=torch.float32, device=device)
    t = torch.as_tensor(np.asarray(arr, dtype=np.float32).reshape(-1)[:n], device=device)
    return t


def compute_seed_features_fast_torch(job: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Fast-path featurization on GPU (``fast=True`` parity with ``compute_seed_features``)."""
    z_np = np.asarray(job["seed_z"], dtype=np.float64).reshape(-1)
    n = int(z_np.size)
    if n == 0:
        return torch.zeros((0, len(SEED_FEATURE_NAMES)), dtype=torch.float32, device=device)

    z = torch.as_tensor(z_np, dtype=torch.float64, device=device)
    seed_uv = torch.as_tensor(np.asarray(job["seed_uv"], dtype=np.float64), device=device)
    seed_xyz = torch.as_tensor(np.asarray(job["seed_xyz"], dtype=np.float64), device=device)
    ground_z = torch.as_tensor(
        np.asarray(job["ground_z_at_seed"], dtype=np.float64).reshape(-1),
        device=device,
    )

    z_min = z.min()
    z_med = z.median()
    q25 = torch.quantile(z, 0.25)
    q75 = torch.quantile(z, 0.75)
    iqr = torch.clamp(q75 - q25, min=1e-3)

    labels = _depth_cluster_labels_1d_torch(z, _DEPTH_CLUSTER_EPS)
    uniq, inv, cnt = torch.unique(labels, return_inverse=True, return_counts=True)
    cluster_medians = torch.stack([z[labels == lab].median() for lab in uniq])
    nearest_i = int(torch.argmin(cluster_medians).item())
    nearest_lab = uniq[nearest_i]
    min_cluster_median = cluster_medians[nearest_i]
    cluster_size = cnt[inv].to(dtype=torch.float64) / float(n)
    cluster_median_per_seed = cluster_medians.to(torch.float64)[inv]

    mask_geom = job.get("mask_geom")
    if mask_geom is None:
        mask_geom = precompute_mask_geom_for_uv(job["mask"], job["seed_uv"])
    area = float(mask_geom["area"])
    x0 = float(mask_geom["x0"])
    y0 = float(mask_geom["y0"])
    w_box = float(mask_geom["w_box"])
    h_box = float(mask_geom["h_box"])
    area_scale = float(mask_geom["area_scale"])
    mask_fill_ratio = float(mask_geom["mask_fill_ratio"])
    boundary_dist = torch.as_tensor(
        np.asarray(mask_geom["boundary_dist"], dtype=np.float64).reshape(-1)[:n],
        device=device,
    )

    support = _support_depth_fraction_torch(z.to(torch.float32), n).to(torch.float64)
    height_above_ground = seed_xyz[:, 2] - ground_z
    depth_rank = torch.argsort(torch.argsort(z)).to(dtype=torch.float64) / max(float(n - 1), 1.0)
    seed_density = float(n) / max(area, 1.0)

    inten = job.get("intensity")
    if inten is None:
        inten_t = torch.full((n,), float("nan"), dtype=torch.float64, device=device)
    else:
        inten_t = torch.as_tensor(
            np.asarray(inten, dtype=np.float64).reshape(-1)[:n], device=device
        )

    is_thermal = float(job.get("is_thermal", 0.0))
    yolo_conf = float(job.get("yolo_conf", 1.0))
    zeros = torch.zeros(n, dtype=torch.float64, device=device)
    nan_col = torch.full((n,), float("nan"), dtype=torch.float64, device=device)

    xmod_in = _col_torch(job.get("xmod_in_mask"), n, 0.0, device).to(torch.float64)
    xmod_c = _col_torch(job.get("xmod_conf"), n, float("nan"), device).to(torch.float64)
    xmod_b = _col_torch(job.get("xmod_boundary_norm"), n, float("nan"), device).to(torch.float64)
    xmod_n = _col_torch(job.get("xmod_count"), n, 0.0, device).to(torch.float64)

    feats = torch.stack(
        [
            z,
            z - z_min,
            z - z_med,
            (z - z_med) / iqr,
            cluster_size.to(torch.float64),
            cluster_median_per_seed.to(torch.float64) - min_cluster_median.to(torch.float64),
            (labels == nearest_lab).to(dtype=torch.float64),
            torch.full((n,), float(uniq.numel()), dtype=torch.float64, device=device),
            height_above_ground,
            (seed_uv[:, 0] - x0) / w_box,
            (seed_uv[:, 1] - y0) / h_box,
            boundary_dist / area_scale,
            torch.full((n,), np.log(max(area, 1.0)), dtype=torch.float64, device=device),
            torch.full((n,), h_box / w_box, dtype=torch.float64, device=device),
            support,
            support,
            torch.full((n,), np.log(float(n)), dtype=torch.float64, device=device),
            depth_rank,
            zeros,
            zeros,
            torch.full((n,), mask_fill_ratio, dtype=torch.float64, device=device),
            torch.full((n,), seed_density, dtype=torch.float64, device=device),
            inten_t,
            torch.full((n,), is_thermal, dtype=torch.float64, device=device),
            zeros,
            nan_col,
            zeros,
            zeros,
            torch.full((n,), yolo_conf, dtype=torch.float64, device=device),
            xmod_in,
            xmod_c,
            xmod_b,
            xmod_n,
        ],
        dim=1,
    )
    return feats.to(torch.float32)


def featurize_and_score_seed_batches_gpu(
    jobs: list[dict[str, Any]],
    payload: Any,
    device: torch.device,
) -> list[np.ndarray]:
    """Featurize (GPU, fast path) + MLP forward in one device-resident pass."""
    if not jobs:
        return []
    sizes = [int(np.asarray(j["seed_z"]).size) for j in jobs]
    if sum(sizes) == 0:
        return [np.zeros(0, dtype=np.float64) for _ in jobs]

    col_idx = _feature_column_indices(payload)
    scorer = _torch_scorer(payload, device)
    feat_parts: list[torch.Tensor] = []
    for job in jobs:
        n = int(np.asarray(job["seed_z"]).size)
        if n == 0:
            continue
        if not job.get("mask_geom"):
            job = dict(job)
            job["mask_geom"] = precompute_mask_geom_for_uv(job["mask"], job["seed_uv"])
        feat_parts.append(compute_seed_features_fast_torch(job, device))

    stacked = torch.cat(feat_parts, dim=0)
    if col_idx is not None:
        stacked = stacked[:, col_idx]
    probs = scorer.predict_proba_pos(stacked).detach().cpu().numpy()

    out: list[np.ndarray] = []
    i = 0
    for n in sizes:
        out.append(probs[i : i + n].astype(np.float64, copy=False))
        i += n
    return out


def score_seed_batches_gpu(
    payload: Any,
    batches: list[np.ndarray],
    device: torch.device,
) -> list[np.ndarray]:
    """Batched MLP forward on GPU from pre-built CPU feature matrices."""
    if not batches:
        return []
    sizes = [int(b.shape[0]) for b in batches]
    if sum(sizes) == 0:
        return [np.zeros(0, dtype=np.float64) for _ in batches]

    col_idx = _feature_column_indices(payload)
    feats = batches
    if col_idx is not None:
        feats = [b[:, col_idx] for b in batches]

    stacked = np.vstack(feats).astype(np.float32, copy=False)
    scorer = _torch_scorer(payload, device)
    probs = scorer.predict_proba_pos(torch.as_tensor(stacked, device=device)).detach().cpu().numpy()

    out: list[np.ndarray] = []
    i = 0
    for n in sizes:
        out.append(probs[i : i + n].astype(np.float64, copy=False))
        i += n
    return out
