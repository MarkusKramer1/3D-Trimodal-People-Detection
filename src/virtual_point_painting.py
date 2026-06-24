"""Point painting: synthesize LiDAR points from 2D instance masks and depth seeds."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.cluster.hierarchy import fclusterdata

from src.paint_config import VirtualPaintConfig
from src.ground_plane import GlobalGroundPlane, fit_global_ground_plane
from src.dataset_io import (
    build_token_lookup,
    link_token_field,
    load_links,
    load_samples,
    read_pcd_xyz_intensity,
    sample_full_path,
)
from src.seed_scorer import (
    _DEFAULT_MAX_SEEDS_PER_INSTANCE,
    _depth_cluster_labels_1d,
    _mask_geometry,
    clear_mask_geometry_cache,
    compute_seed_features,
    depth_stratified_subsample_indices,
    featurize_seed_batches_parallel,
    filter_scored_indices,
    filter_scored_seeds,
    load_seed_scorer,
    precompute_mask_geom_for_uv,
    score_seed_batches,
    score_seeds,
)
from src.geometry import (
    camera_intrinsic_matrix,
    distortion_coeffs_from_calibration,
    intrinsic_for_actual_image,
    lidar_to_camera_from_calibrations,
    se3_matrix_from_calibration,
    load_calibrations,
    point_to_pixel_z_cam_like_projection,
    rectify_pil_rgb_for_calibration,
)


def _ensure_night_placeholder(img_path: Path, width: int, height: int) -> None:
    """Night RGB samples reference a shared black ``night.jpg`` placeholder."""
    if img_path.name != "night.jpg" or img_path.is_file():
        return
    img_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (max(1, int(width)), max(1, int(height))), (0, 0, 0)).save(
        img_path, format="JPEG", quality=85
    )


@dataclass
class InstancePaintStats:
    ann_id: int
    modality: str
    n_seeds_raw: int
    n_seeds_kept: int
    n_virtual: int
    skipped_reason: str | None
    median_seed_depth_m: float | None
    n_candidates: int = 0


@dataclass
class VirtualPaintResult:
    xyz_real: np.ndarray
    xyz_virtual: np.ndarray
    features: np.ndarray
    is_virtual: np.ndarray
    per_instance: list[InstancePaintStats] = field(default_factory=list)
    modality_runs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SeedPoint:
    u: float
    v: float
    z_cam: float
    orig_idx: int


SeedArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
InstanceSeedPreScore = tuple[SeedArrays, np.ndarray]


def _seed_lists_equal(a: list[SeedPoint], b: list[SeedPoint]) -> bool:
    if len(a) != len(b):
        return False
    return all(x.orig_idx == y.orig_idx for x, y in zip(a, b))


def _seed_arrays_equal(a: SeedArrays, b: SeedArrays) -> bool:
    return int(a[3].shape[0]) == int(b[3].shape[0]) and np.array_equal(a[3], b[3])


def _capped_seed_arrays_from_raw(
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    oi: np.ndarray,
    cfg: VirtualPaintConfig,
) -> SeedArrays:
    cap = int(cfg.seed_scorer_max_seeds_per_instance)
    if not cfg.seed_scorer_fast_path or cap <= 0 or z.size <= cap:
        return u, v, z, oi
    pick = depth_stratified_subsample_indices(z, cap)
    return u[pick], v[pick], z[pick], oi[pick]


@dataclass(frozen=True)
class SweepProjection:
    uv: np.ndarray
    z_cam: np.ndarray
    valid_idx: np.ndarray
    orig_idx: np.ndarray


@dataclass
class RectifiedView:
    """Rectified camera intrinsics and image size for one sample token."""

    K: np.ndarray
    K_raw: np.ndarray
    dist: np.ndarray | None
    h_rect: int
    w_rect: int


@dataclass
class SequencePaintCache:
    """Preloaded sequence metadata for fast per-frame painting and export."""

    subset: str
    project_root: Path
    links: list[dict[str, Any]]
    by_token: dict[str, dict[str, Any]]
    calibs: dict[str, Any]
    rectified_views: dict[str, RectifiedView] = field(default_factory=dict)
    rectified_images: dict[str, np.ndarray] = field(default_factory=dict)
    lidar_by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        project_root: Path,
        subset: str,
    ) -> SequencePaintCache:
        samples = load_samples(project_root, subset)
        return cls(
            subset=subset,
            project_root=project_root,
            links=load_links(project_root, subset),
            by_token=build_token_lookup(samples),
            calibs=load_calibrations(project_root, subset),
        )


def get_rectified_view(
    cache: SequencePaintCache,
    sample: dict[str, Any],
    cam_cal: dict[str, Any],
) -> tuple[RectifiedView | None, str | None]:
    """Open image once per sample token and cache rectified intrinsics + size."""
    tok = str(sample["token"])
    cached = cache.rectified_views.get(tok)
    if cached is not None:
        return cached, None

    img_path = sample_full_path(cache.project_root, cache.subset, sample["filename"])
    _ensure_night_placeholder(
        img_path, int(sample["width"]), int(sample["height"])
    )
    if not img_path.is_file():
        return None, f"Image not found: {img_path}"

    pil = ImageOps.exif_transpose(Image.open(img_path).convert("RGB"))
    pil_rect, k_rect = rectify_pil_rgb_for_calibration(
        pil,
        cam_cal,
        (int(sample["width"]), int(sample["height"])),
    )
    h_rect, w_rect = pil_rect.size[1], pil_rect.size[0]
    w_raw, h_raw = pil.size
    try:
        k_nom = camera_intrinsic_matrix(cam_cal)
    except KeyError:
        k_nom = np.eye(3, dtype=np.float64)
    k_raw = intrinsic_for_actual_image(
        k_nom,
        (int(sample["width"]), int(sample["height"])),
        (w_raw, h_raw),
    )
    dist = distortion_coeffs_from_calibration(cam_cal)
    view = RectifiedView(K=k_rect, K_raw=k_raw, dist=dist, h_rect=h_rect, w_rect=w_rect)
    cache.rectified_views[tok] = view
    if tok not in cache.rectified_images:
        cache.rectified_images[tok] = np.asarray(pil_rect.convert("RGB"), dtype=np.uint8)
    return view, None


def load_rectified_image_cached(
    cache: SequencePaintCache,
    frame_idx: int,
    modality: Literal["rgb", "thermal"],
) -> tuple[np.ndarray, int, int] | None:
    """Load one rectified camera frame using ``SequencePaintCache`` (no JSON re-read)."""
    if frame_idx < 0 or frame_idx >= len(cache.links):
        return None
    link = cache.links[frame_idx]
    tok_key = link_token_field(modality)
    sample = cache.by_token.get(str(link[tok_key]))
    if sample is None:
        return None
    tok = str(sample["token"])
    cached = cache.rectified_images.get(tok)
    if cached is not None:
        h, w = cached.shape[:2]
        view = cache.rectified_views.get(tok)
        if view is not None:
            return cached, int(view.h_rect), int(view.w_rect)
        return cached, h, w

    cam_cal = cache.calibs.get(str(sample.get("calibrated_sensor_token") or ""))
    if not cam_cal:
        return None

    img_path = sample_full_path(cache.project_root, cache.subset, str(sample["filename"]))
    _ensure_night_placeholder(
        img_path, int(sample["width"]), int(sample["height"])
    )
    if not img_path.is_file():
        return None
    try:
        pil = ImageOps.exif_transpose(Image.open(img_path).convert("RGB"))
    except OSError:
        return None
    pil_rect, k_rect = rectify_pil_rgb_for_calibration(
        pil,
        cam_cal,
        (int(sample.get("width") or pil.size[0]), int(sample.get("height") or pil.size[1])),
    )
    h_rect, w_rect = pil_rect.size[1], pil_rect.size[0]
    w_raw, h_raw = pil.size
    if tok not in cache.rectified_views:
        try:
            k_nom = camera_intrinsic_matrix(cam_cal)
        except KeyError:
            k_nom = np.eye(3, dtype=np.float64)
        k_raw = intrinsic_for_actual_image(
            k_nom,
            (int(sample["width"]), int(sample["height"])),
            (w_raw, h_raw),
        )
        dist = distortion_coeffs_from_calibration(cam_cal)
        cache.rectified_views[tok] = RectifiedView(
            K=k_rect, K_raw=k_raw, dist=dist, h_rect=h_rect, w_rect=w_rect
        )

    image_rgb = np.asarray(pil_rect.convert("RGB"), dtype=np.uint8)
    cache.rectified_images[tok] = image_rgb
    return image_rgb, int(h_rect), int(w_rect)


def load_frame_lidar_cached(
    cache: SequencePaintCache,
    frame_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load synchronized LiDAR xyz + intensity once per frame index."""
    cached = cache.lidar_by_frame.get(frame_idx)
    if cached is not None:
        return cached
    link = cache.links[frame_idx]
    lid_s = cache.by_token[link["lidar_token"]]
    pcd_path = sample_full_path(cache.project_root, cache.subset, lid_s["filename"])
    xyz_real, intensity_real = read_pcd_xyz_intensity(pcd_path)
    cache.lidar_by_frame[frame_idx] = (xyz_real, intensity_real)
    return xyz_real, intensity_real


def project_sweep_to_image(
    xyz: np.ndarray,
    T_lc: np.ndarray,
    K: np.ndarray,
    h: int,
    w: int,
    *,
    max_range_m: float = 0.0,
) -> SweepProjection:
    """Project LiDAR sweep to rectified image; keep in-bounds front-facing points."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if max_range_m > 0.0:
        in_range = np.linalg.norm(pts, axis=1) <= float(max_range_m)
        range_idx = np.flatnonzero(in_range)
        pts = pts[in_range]
    else:
        range_idx = None

    uv_all, valid_z, z_cam_all = point_to_pixel_z_cam_like_projection(pts, T_lc, K)
    u = uv_all[:, 0]
    v = uv_all[:, 1]
    ok = (
        valid_z
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0.0)
        & (u < float(w))
        & (v >= 0.0)
        & (v < float(h))
    )
    valid_idx = np.flatnonzero(ok)
    if range_idx is not None:
        orig_idx = range_idx[valid_idx]
    else:
        orig_idx = valid_idx.copy()
    return SweepProjection(
        uv=uv_all[ok],
        z_cam=z_cam_all[ok],
        valid_idx=valid_idx,
        orig_idx=orig_idx,
    )


def _projection_uv_int(
    projection: SweepProjection,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rounded, clipped pixel coords for every projected point (cached per sweep)."""
    n = int(projection.uv.shape[0])
    key = (int(projection.uv.ctypes.data), h, w, n)
    cached = getattr(_projection_uv_int, "_cache", {}).get(key)  # type: ignore[attr-defined]
    if cached is not None:
        return cached
    uu = np.clip(np.rint(projection.uv[:, 0]).astype(np.int64), 0, w - 1)
    vv = np.clip(np.rint(projection.uv[:, 1]).astype(np.int64), 0, h - 1)
    store = getattr(_projection_uv_int, "_cache", {})
    store[key] = (uu, vv)
    _projection_uv_int._cache = store  # type: ignore[attr-defined]
    return uu, vv


def clear_projection_uv_cache() -> None:
    """Drop cached pixel coords before painting a new frame."""
    _projection_uv_int._cache = {}  # type: ignore[attr-defined]


def _seed_arrays_from_projection(
    projection: SweepProjection,
    mask: np.ndarray,
    *,
    uu: np.ndarray | None = None,
    vv: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(u, v, z_cam, orig_idx)`` for LiDAR points inside ``mask``."""
    h, w = mask.shape
    if uu is None or vv is None:
        uu, vv = _projection_uv_int(projection, h, w)
    hit_idx = np.flatnonzero(np.asarray(mask, dtype=bool)[vv, uu])
    if hit_idx.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, np.empty(0, dtype=np.int64)
    return (
        projection.uv[hit_idx, 0].astype(np.float64),
        projection.uv[hit_idx, 1].astype(np.float64),
        projection.z_cam[hit_idx].astype(np.float64),
        projection.orig_idx[hit_idx].astype(np.int64),
    )


def _seed_points_from_arrays(
    u: np.ndarray,
    v: np.ndarray,
    z_cam: np.ndarray,
    orig_idx: np.ndarray,
) -> list[SeedPoint]:
    return [
        SeedPoint(u=float(uu), v=float(vv), z_cam=float(zz), orig_idx=int(oi))
        for uu, vv, zz, oi in zip(u, v, z_cam, orig_idx)
    ]


def build_cross_modal_membership(
    projection: SweepProjection,
    masks_with_scores: list,
    n_lidar: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-LiDAR-point summary of the *other* modality's masks for fusion.

    Both cameras project the same sweep, so a LiDAR point indexed by ``orig_idx``
    can be looked up against this (other-modality) projection + masks with no
    instance association. Returns three arrays of length ``n_lidar`` indexed by
    ``orig_idx``: best detector confidence among covering masks (NaN if none),
    most-interior normalized boundary distance among covering masks (NaN if
    none), and the count of covering masks (0 if none).
    """
    conf = np.full(n_lidar, np.nan, dtype=np.float64)
    boundary = np.full(n_lidar, np.nan, dtype=np.float64)
    count = np.zeros(n_lidar, dtype=np.float64)
    if not masks_with_scores:
        return conf, boundary, count
    h, w = np.asarray(masks_with_scores[0][1], dtype=bool).shape
    uu, vv = _projection_uv_int(projection, h, w)
    oi_all = projection.orig_idx
    for entry in masks_with_scores:
        ann_id, mask = entry[0], entry[1]
        score = float(entry[2]) if len(entry) > 2 else 1.0
        m = np.asarray(mask, dtype=bool)
        _area, _x0, _y0, _x1, _y1, _w_box, _h_box, area_scale, dist_map, _fill = _mask_geometry(m)
        hit_idx = np.flatnonzero(m[vv, uu])
        if hit_idx.size == 0:
            continue
        orig = oi_all[hit_idx]
        valid = (orig >= 0) & (orig < n_lidar)
        if not np.any(valid):
            continue
        orig = orig[valid]
        b = dist_map[vv[hit_idx][valid], uu[hit_idx][valid]].astype(np.float64) / area_scale
        sc = np.full(orig.shape[0], score, dtype=np.float64)
        np.add.at(count, orig, 1.0)
        cur_conf = conf[orig]
        better = np.isnan(cur_conf) | (sc > cur_conf)
        if np.any(better):
            idx = orig[better]
            conf[idx] = sc[better]
        cur_b = boundary[orig]
        better_b = np.isnan(cur_b) | (b > cur_b)
        if np.any(better_b):
            idx = orig[better_b]
            boundary[idx] = b[better_b]
    return conf, boundary, count


def collect_seeds(projection: SweepProjection, mask: np.ndarray) -> list[SeedPoint]:
    """Projected LiDAR points whose pixels fall inside ``mask``."""
    u, v, z, oi = _seed_arrays_from_projection(projection, mask)
    return _seed_points_from_arrays(u, v, z, oi)


def fit_frame_ground_plane(xyz: np.ndarray) -> GlobalGroundPlane | None:
    """Fit ground plane ``z = a·x + b·y + c`` from one sweep's low returns."""
    return fit_global_ground_plane(
        np.asarray(xyz, dtype=np.float64),
        min_points=100,
        max_fit_points=12_000,
    )


def fallback_ground_z(xyz: np.ndarray, *, z_percentile: float = 5.0) -> float:
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return 0.0
    return float(np.percentile(pts[:, 2], float(z_percentile)))


def ground_height_at_xy(
    x: np.ndarray,
    y: np.ndarray,
    plane: GlobalGroundPlane | None,
    *,
    fallback_z: float,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if plane is not None:
        return plane.a * x_arr + plane.b * y_arr + plane.c
    return np.full(x_arr.shape[0], float(fallback_z), dtype=np.float64)


def above_ground_mask(
    xyz: np.ndarray,
    plane: GlobalGroundPlane | None,
    *,
    ground_clearance_m: float,
    fallback_ground_z: float,
) -> np.ndarray:
    """True where lidar-frame ``z`` is above the ground plane plus clearance."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    z_g = ground_height_at_xy(pts[:, 0], pts[:, 1], plane, fallback_z=fallback_ground_z)
    return pts[:, 2] > z_g + float(ground_clearance_m)


def filter_ground_seeds(
    seeds: list[SeedPoint],
    xyz: np.ndarray,
    plane: GlobalGroundPlane | None,
    *,
    ground_clearance_m: float,
    fallback_ground_z: float,
) -> list[SeedPoint]:
    if not seeds:
        return seeds
    u, v, z, oi = _seed_arrays_from_seed_points(seeds)
    fu, fv, fz, foi = _filter_ground_seed_arrays(
        u, v, z, oi, xyz, plane,
        ground_clearance_m=ground_clearance_m,
        fallback_ground_z=fallback_ground_z,
    )
    return _seed_points_from_arrays(fu, fv, fz, foi)


def _seed_arrays_from_seed_points(seeds: list[SeedPoint]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not seeds:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, np.empty(0, dtype=np.int64)
    return (
        np.array([s.u for s in seeds], dtype=np.float64),
        np.array([s.v for s in seeds], dtype=np.float64),
        np.array([s.z_cam for s in seeds], dtype=np.float64),
        np.array([s.orig_idx for s in seeds], dtype=np.int64),
    )


def _filter_ground_seed_arrays(
    u: np.ndarray,
    v: np.ndarray,
    z_cam: np.ndarray,
    orig_idx: np.ndarray,
    xyz: np.ndarray,
    plane: GlobalGroundPlane | None,
    *,
    ground_clearance_m: float,
    fallback_ground_z: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if orig_idx.size == 0:
        return u, v, z_cam, orig_idx
    pts = np.asarray(xyz, dtype=np.float64)
    p = pts[orig_idx]
    z_g = ground_height_at_xy(
        p[:, 0], p[:, 1], plane, fallback_z=fallback_ground_z
    )
    keep = p[:, 2] > z_g + float(ground_clearance_m)
    return u[keep], v[keep], z_cam[keep], orig_idx[keep]


def apply_seed_scorer(
    raw_seeds: list[SeedPoint],
    mask: np.ndarray,
    xyz: np.ndarray,
    *,
    borrowed: bool,
    T_lc: np.ndarray,
    K: np.ndarray,
    ground_plane: GlobalGroundPlane | None,
    ground_fallback_z: float,
    cfg: VirtualPaintConfig,
    intensity: np.ndarray | None = None,
    modality: str = "rgb",
    competitors: list[tuple[np.ndarray, float]] | None = None,
    use_topk: bool = False,
    yolo_conf: float = 1.0,
    xmod: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> list[SeedPoint] | None:
    """Learned seed-instance association: keep seeds the model assigns to the person.

    Returns ``None`` (caller falls back) when the model is unavailable or keeps
    fewer than ``min_seeds`` seeds — unless ``use_topk``, which then keeps the
    ``min_seeds`` highest-probability seeds instead.

    ``yolo_conf`` is this mask's detector confidence. ``xmod`` is the per-LiDAR
    other-modality membership ``(conf, boundary, count)`` arrays (indexed by
    ``orig_idx``) from ``build_cross_modal_membership``; it is skipped for
    borrowed seeds whose ``orig_idx`` refer to neighbor frames.
    """
    payload = load_seed_scorer(cfg.seed_scorer_model)
    if payload is None or not raw_seeds:
        return None
    if borrowed:
        seed_xyz = lidar_xyz_from_seed_points(raw_seeds, T_lc=T_lc, K=K)
        seed_inten: np.ndarray | None = None
    else:
        seed_xyz = lidar_xyz_from_seeds(raw_seeds, xyz)
        if intensity is not None:
            idx = np.array([s.orig_idx for s in raw_seeds], dtype=np.int64)
            seed_inten = np.asarray(intensity, dtype=np.float64).reshape(-1)[idx]
        else:
            seed_inten = None
    seed_uv = np.array([[s.u, s.v] for s in raw_seeds], dtype=np.float64)
    seed_z = np.array([s.z_cam for s in raw_seeds], dtype=np.float64)
    ground_z = ground_height_at_xy(
        seed_xyz[:, 0], seed_xyz[:, 1], ground_plane, fallback_z=ground_fallback_z
    )
    xmod_in = xmod_conf = xmod_boundary = xmod_count = None
    if xmod is not None and not borrowed:
        conf_by, boundary_by, count_by = xmod
        oi = np.array([s.orig_idx for s in raw_seeds], dtype=np.int64)
        valid = (oi >= 0) & (oi < conf_by.shape[0])
        idx = np.where(valid, oi, 0)
        xmod_conf = np.where(valid, conf_by[idx], np.nan)
        xmod_boundary = np.where(valid, boundary_by[idx], np.nan)
        xmod_count = np.where(valid, count_by[idx], 0.0)
        xmod_in = (xmod_count > 0).astype(np.float64)
    fast = bool(cfg.seed_scorer_fast_path)
    cap = int(cfg.seed_scorer_max_seeds_per_instance) if fast and not borrowed else 0
    if cap > 0:
        (
            raw_seeds,
            seed_uv,
            seed_z,
            seed_xyz,
            ground_z,
            seed_inten,
            xmod_in,
            xmod_conf,
            xmod_boundary,
            xmod_count,
        ) = _cap_scorer_seed_arrays(
            raw_seeds,
            seed_uv,
            seed_z,
            seed_xyz,
            ground_z,
            seed_inten,
            xmod_in,
            xmod_conf,
            xmod_boundary,
            xmod_count,
            max_seeds=cap,
        )
    feats = compute_seed_features(
        seed_uv,
        seed_z,
        seed_xyz,
        mask,
        ground_z,
        intensity=seed_inten,
        is_thermal=1.0 if modality == "thermal" else 0.0,
        competitors=competitors,
        yolo_conf=yolo_conf,
        xmod_in_mask=xmod_in,
        xmod_conf=xmod_conf,
        xmod_boundary_norm=xmod_boundary,
        xmod_count=xmod_count,
        fast=fast,
    )
    probs = score_seeds(payload, feats)
    return filter_scored_seeds(
        raw_seeds,
        probs,
        threshold=float(cfg.seed_scorer_threshold),
        min_seeds=int(cfg.min_seeds),
        use_topk=use_topk,
    )


def _cap_scorer_seed_arrays(
    raw_seeds: list[SeedPoint],
    seed_uv: np.ndarray,
    seed_z: np.ndarray,
    seed_xyz: np.ndarray,
    ground_z: np.ndarray,
    seed_inten: np.ndarray | None,
    xmod_in: np.ndarray | None,
    xmod_conf: np.ndarray | None,
    xmod_boundary: np.ndarray | None,
    xmod_count: np.ndarray | None,
    *,
    max_seeds: int,
) -> tuple[
    list[SeedPoint],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Depth-stratified cap so huge masks stay within the scorer latency budget."""
    pick = depth_stratified_subsample_indices(seed_z, max_seeds)
    if pick.shape[0] == seed_z.shape[0]:
        return (
            raw_seeds,
            seed_uv,
            seed_z,
            seed_xyz,
            ground_z,
            seed_inten,
            xmod_in,
            xmod_conf,
            xmod_boundary,
            xmod_count,
        )

    def _take(arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        return np.asarray(arr, dtype=np.float64).reshape(-1)[pick]

    return (
        [raw_seeds[int(i)] for i in pick],
        seed_uv[pick],
        seed_z[pick],
        seed_xyz[pick],
        ground_z[pick],
        _take(seed_inten),
        _take(xmod_in),
        _take(xmod_conf),
        _take(xmod_boundary),
        _take(xmod_count),
    )


def _precompute_seed_scorer_frame(
    *,
    cfg: VirtualPaintConfig,
    xyz: np.ndarray,
    intensity: np.ndarray | None,
    modality_plans: list[dict[str, Any]],
    xmod_by_mod: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    score_device: Any | None = None,
    ground_plane: GlobalGroundPlane | None = None,
    ground_fallback_z: float | None = None,
    paint_plans_for_xmod: list[dict[str, Any]] | None = None,
) -> dict[tuple[str, int], InstanceSeedPreScore]:
    """One parallel featurize + batched MLP for every instance on this frame."""
    payload = load_seed_scorer(cfg.seed_scorer_model)
    if payload is None or not modality_plans:
        return {}

    if ground_plane is None and cfg.exclude_ground_points:
        ground_plane = fit_frame_ground_plane(xyz)
    if ground_fallback_z is None:
        ground_fallback_z = fallback_ground_z(xyz)
    fast = bool(cfg.seed_scorer_fast_path)
    cap = int(cfg.seed_scorer_max_seeds_per_instance) if fast else 0
    mirror_thermal = bool(cfg.seed_scorer_thermal_from_rgb)
    has_rgb = any(str(p["modality"]) == "rgb" for p in modality_plans)
    has_thermal = any(str(p["modality"]) == "thermal" for p in modality_plans)
    mirror_thermal = mirror_thermal and has_rgb and has_thermal

    def _plan_context(
        plan: dict[str, Any],
    ) -> tuple[
        SweepProjection,
        dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        list[tuple[int, np.ndarray, float]],
        np.ndarray,
        np.ndarray,
        tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    ]:
        projection: SweepProjection = plan["projection"]
        context_masks: list = plan["context_masks"]
        instance_masks: list = plan.get("instance_masks") or context_masks
        xmod = xmod_by_mod.get(str(plan["modality"]))
        if context_masks:
            h, w = np.asarray(context_masks[0][1], dtype=bool).shape
        else:
            h, w = 1, 1
        uu, vv = _projection_uv_int(projection, h, w)
        seeds_by_ann: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        instance_context: list[tuple[int, np.ndarray, float]] = []
        seed_targets = instance_masks if fast else context_masks
        for ctx_entry in seed_targets:
            ctx_ann_id, ctx_mask = int(ctx_entry[0]), ctx_entry[1]
            arrays = _seed_arrays_from_projection(projection, ctx_mask, uu=uu, vv=vv)
            seeds_by_ann[ctx_ann_id] = arrays
            if not fast:
                med = float(np.median(arrays[2])) if arrays[2].size else float("nan")
                instance_context.append((ctx_ann_id, ctx_mask, med))
        return projection, seeds_by_ann, instance_context, uu, vv, xmod

    def _instance_bundle(
        mod: str,
        plan_ctx: tuple[
            SweepProjection,
            dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
            list[tuple[int, np.ndarray, float]],
            np.ndarray,
            np.ndarray,
            tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        ],
        entry: tuple,
    ) -> tuple[tuple[str, int], SeedArrays, dict[str, Any]] | None:
        projection, seeds_by_ann, instance_context, uu, vv, xmod = plan_ctx
        ann_id = int(entry[0])
        mask = entry[1]
        yolo_conf = float(entry[2]) if len(entry) > 2 else 1.0

        capped = _capped_seed_arrays(plan_ctx, entry)
        if capped is None:
            return None
        u, v, z, oi = capped
        seed_uv = np.column_stack([u, v]).astype(np.float64)
        seed_z = np.asarray(z, dtype=np.float64)
        seed_xyz = np.asarray(xyz, dtype=np.float64)[oi]
        ground_z = ground_height_at_xy(
            seed_xyz[:, 0], seed_xyz[:, 1], ground_plane, fallback_z=ground_fallback_z
        )
        seed_inten: np.ndarray | None
        if intensity is not None:
            seed_inten = np.asarray(intensity, dtype=np.float64).reshape(-1)[oi]
        else:
            seed_inten = None

        xmod_in = xmod_conf = xmod_boundary = xmod_count = None
        if xmod is not None:
            conf_by, boundary_by, count_by = xmod
            valid = (oi >= 0) & (oi < conf_by.shape[0])
            sidx = np.where(valid, oi, 0)
            xmod_conf = np.where(valid, conf_by[sidx], np.nan)
            xmod_boundary = np.where(valid, boundary_by[sidx], np.nan)
            xmod_count = np.where(valid, count_by[sidx], 0.0)
            xmod_in = (xmod_count > 0).astype(np.float64)

        seed_arrays: SeedArrays = (seed_uv[:, 0], seed_uv[:, 1], seed_z, oi)

        competitors = None
        if not fast:
            competitors = [
                (c_mask, c_med)
                for c_aid, c_mask, c_med in instance_context
                if c_aid != ann_id and np.isfinite(c_med)
            ] or None

        job = dict(
            seed_uv=seed_uv,
            seed_z=seed_z,
            seed_xyz=seed_xyz,
            mask=mask,
            ground_z_at_seed=ground_z,
            intensity=seed_inten,
            is_thermal=1.0 if mod == "thermal" else 0.0,
            competitors=competitors,
            yolo_conf=yolo_conf,
            xmod_in_mask=xmod_in,
            xmod_conf=xmod_conf,
            xmod_boundary_norm=xmod_boundary,
            xmod_count=xmod_count,
            fast=fast,
            mask_geom=precompute_mask_geom_for_uv(mask, seed_uv),
        )
        return (mod, ann_id), seed_arrays, job

    def _capped_seed_arrays(
        plan_ctx: tuple[
            SweepProjection,
            dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
            list[tuple[int, np.ndarray, float]],
            np.ndarray,
            np.ndarray,
            tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        ],
        entry: tuple,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        projection, seeds_by_ann, _instance_context, uu, vv, _xmod = plan_ctx
        ann_id = int(entry[0])
        mask = entry[1]
        if ann_id in seeds_by_ann:
            u, v, z, oi = seeds_by_ann[ann_id]
        else:
            u, v, z, oi = _seed_arrays_from_projection(projection, mask, uu=uu, vv=vv)
        if cfg.exclude_ground_points:
            u, v, z, oi = _filter_ground_seed_arrays(
                u, v, z, oi, xyz, ground_plane,
                ground_clearance_m=cfg.ground_clearance_m,
                fallback_ground_z=ground_fallback_z,
            )
        if oi.size == 0:
            return None
        if cap > 0 and z.size > cap:
            pick = depth_stratified_subsample_indices(z, cap)
            u, v, z, oi = u[pick], v[pick], z[pick], oi[pick]
        return u, v, z, oi

    def _score_jobs(
        bundles: list[tuple[tuple[str, int], SeedArrays, dict[str, Any]]],
    ) -> dict[tuple[str, int], InstanceSeedPreScore]:
        if not bundles:
            return {}
        jobs = [b[2] for b in bundles]
        if fast:
            feats = [
                compute_seed_features(
                    j["seed_uv"],
                    j["seed_z"],
                    j["seed_xyz"],
                    j["mask"],
                    j["ground_z_at_seed"],
                    intensity=j.get("intensity"),
                    is_thermal=j.get("is_thermal", 0.0),
                    competitors=j.get("competitors"),
                    yolo_conf=j.get("yolo_conf", 1.0),
                    xmod_in_mask=j.get("xmod_in_mask"),
                    xmod_conf=j.get("xmod_conf"),
                    xmod_boundary_norm=j.get("xmod_boundary_norm"),
                    xmod_count=j.get("xmod_count"),
                    fast=True,
                    mask_geom=j.get("mask_geom"),
                )
                for j in jobs
            ]
        else:
            feats = featurize_seed_batches_parallel(jobs, fast=fast)
        if score_device is not None:
            from src.gpu_paint import score_seed_batches_gpu

            probs_flat = score_seed_batches_gpu(payload, feats, score_device)
        else:
            probs_flat = score_seed_batches(payload, feats)
        return {
            b[0]: (b[1], p)
            for b, p in zip(bundles, probs_flat)
        }

    out: dict[tuple[str, int], InstanceSeedPreScore] = {}
    prob_by_orig: dict[int, float] = {}

    plan_ctx_by_mod: dict[str, tuple] = {
        str(plan["modality"]): _plan_context(plan) for plan in modality_plans
    }

    primary_plans = (
        [p for p in modality_plans if str(p["modality"]) == "rgb"]
        if mirror_thermal
        else modality_plans
    )
    primary_bundles: list[tuple[tuple[str, int], SeedArrays, dict[str, Any]]] = []
    for plan in primary_plans:
        mod = str(plan["modality"])
        pctx = plan_ctx_by_mod[mod]
        for entry in plan["instance_masks"]:
            bundle = _instance_bundle(mod, pctx, entry)
            if bundle is not None:
                primary_bundles.append(bundle)

    out.update(_score_jobs(primary_bundles))
    if mirror_thermal:
        for b in primary_bundles:
            if b[0][0] != "rgb":
                continue
            _mod, _aid = b[0]
            seed_arrays = out[b[0]][0]
            _u, _v, z, oi = seed_arrays
            probs = out[("rgb", _aid)][1]
            for o, p in zip(oi, probs):
                oi_i = int(o)
                if float(p) > prob_by_orig.get(oi_i, 0.0):
                    prob_by_orig[oi_i] = float(p)

        thermal_bundles: list[tuple[tuple[str, int], SeedArrays, dict[str, Any]]] = []
        thermal_fallback_entries: list[tuple] = []
        th_ctx = plan_ctx_by_mod.get("thermal")
        if th_ctx is not None:
            for plan in modality_plans:
                if str(plan["modality"]) != "thermal":
                    continue
                for entry in plan["instance_masks"]:
                    ann_id = int(entry[0])
                    key = ("thermal", ann_id)
                    arrays = _capped_seed_arrays(th_ctx, entry)
                    if arrays is None:
                        continue
                    u, v, z, oi = arrays
                    if prob_by_orig:
                        hit = int(np.isin(oi, np.fromiter(prob_by_orig.keys(), dtype=np.int64)).sum())
                    else:
                        hit = 0
                    if hit >= max(1, int(0.25 * oi.size)):
                        probs = np.array(
                            [prob_by_orig.get(int(o), 0.0) for o in oi],
                            dtype=np.float64,
                        )
                        out[key] = ((u, v, z, oi), probs)
                    else:
                        thermal_fallback_entries.append(entry)
        if thermal_fallback_entries:
            if "thermal" not in xmod_by_mod and paint_plans_for_xmod:
                xmod_by_mod.update(
                    _build_xmod_from_paint_plans(
                        paint_plans_for_xmod,
                        int(xyz.shape[0]),
                        only_targets=["thermal"],
                    )
                )
                th_ctx = _plan_context(
                    next(p for p in modality_plans if str(p["modality"]) == "thermal")
                )
            if th_ctx is not None:
                for entry in thermal_fallback_entries:
                    bundle = _instance_bundle("thermal", th_ctx, entry)
                    if bundle is not None:
                        thermal_bundles.append(bundle)
        out.update(_score_jobs(thermal_bundles))

    return out


def _scorer_pool_seeds(raw_seeds: list[SeedPoint], cfg: VirtualPaintConfig) -> list[SeedPoint]:
    """Seeds actually sent to the scorer (depth-stratified cap in fast mode)."""
    cap = int(cfg.seed_scorer_max_seeds_per_instance)
    if not cfg.seed_scorer_fast_path or cap <= 0 or len(raw_seeds) <= cap:
        return raw_seeds
    z = np.array([s.z_cam for s in raw_seeds], dtype=np.float64)
    pick = depth_stratified_subsample_indices(z, cap)
    return [raw_seeds[int(i)] for i in pick]


def reject_contaminated_seed_arrays(
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    orig_idx: np.ndarray,
    *,
    eps_m: float,
    outlier_threshold_m: float,
    min_seeds: int,
) -> tuple[SeedArrays | None, str | None]:
    """Keep dominant near depth cluster; drop outliers beyond threshold from median."""
    if z.size < min_seeds:
        return None, f"fewer than {min_seeds} raw seeds ({z.size})"

    labels = _depth_cluster_labels_1d(np.asarray(z, dtype=np.float64), float(eps_m))
    best_label: int | None = None
    best_median = float("inf")
    for lab in np.unique(labels):
        cluster_z = z[labels == lab]
        med = float(np.median(cluster_z))
        if med < best_median:
            best_median = med
            best_label = int(lab)

    if best_label is None:
        return None, "depth clustering failed"

    cluster_mask = labels == best_label
    median_z = float(np.median(z[cluster_mask]))
    inlier = cluster_mask & (np.abs(z - median_z) <= float(outlier_threshold_m))
    if int(np.sum(inlier)) < min_seeds:
        return None, (
            f"fewer than {min_seeds} seeds after outlier rejection "
            f"({int(np.sum(inlier))} kept, median z={median_z:.2f} m)"
        )
    return (u[inlier], v[inlier], z[inlier], orig_idx[inlier]), None


def reject_contaminated_seeds(
    seeds: list[SeedPoint],
    *,
    eps_m: float,
    outlier_threshold_m: float,
    min_seeds: int,
) -> tuple[list[SeedPoint], str | None]:
    """Keep dominant near depth cluster; drop outliers beyond threshold from median."""
    if len(seeds) < min_seeds:
        return [], f"fewer than {min_seeds} raw seeds ({len(seeds)})"

    u, v, z, oi = _seed_arrays_from_seed_points(seeds)
    kept, reason = reject_contaminated_seed_arrays(
        u, v, z, oi,
        eps_m=eps_m,
        outlier_threshold_m=outlier_threshold_m,
        min_seeds=min_seeds,
    )
    if kept is None:
        return [], reason
    return _seed_points_from_arrays(*kept), None


def filter_virtual_candidates(
    candidates: np.ndarray,
    *,
    max_range_m: float,
    exclude_ground: bool,
    ground_plane: GlobalGroundPlane | None,
    ground_clearance_m: float,
    ground_fallback_z: float,
) -> np.ndarray:
    pts = np.asarray(candidates, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return pts
    dist = np.linalg.norm(pts, axis=1)
    keep = dist <= float(max_range_m)
    if exclude_ground:
        keep = keep & above_ground_mask(
            pts,
            ground_plane,
            ground_clearance_m=ground_clearance_m,
            fallback_ground_z=ground_fallback_z,
        )
    return pts[keep]


def cluster_select_medoid(
    candidates: np.ndarray,
    *,
    eps_m: float,
    is_seed: np.ndarray | None = None,
    prefer_seed_cluster: bool = False,
    prefer_seed_medoid: bool = False,
) -> tuple[np.ndarray, int]:
    """Cluster 3D candidates; return medoid of best cluster and cluster size."""
    pts = np.asarray(candidates, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64), 0
    if n == 1:
        return pts, 1
    try:
        labels = fclusterdata(pts, t=float(eps_m), criterion="distance", method="single")
    except Exception:
        labels = np.ones(n, dtype=np.int64)
    seed_mask = (
        np.asarray(is_seed, dtype=bool).reshape(-1) if is_seed is not None else None
    )

    def _cluster_score(lab: int) -> tuple[int, int]:
        m = labels == lab
        seed_n = (
            int(np.sum(seed_mask[m]))
            if seed_mask is not None and prefer_seed_cluster
            else 0
        )
        return (seed_n, int(np.sum(m)))

    best_label = int(max(np.unique(labels), key=_cluster_score))
    cluster_mask = labels == best_label
    cluster_pts = pts[cluster_mask]
    cluster_n = int(cluster_pts.shape[0])
    if cluster_n == 1:
        return cluster_pts, 1
    medoid_pts = cluster_pts
    if (
        prefer_seed_medoid
        and seed_mask is not None
        and np.any(seed_mask[cluster_mask])
    ):
        medoid_pts = cluster_pts[seed_mask[cluster_mask]]
        if medoid_pts.shape[0] == 1:
            return medoid_pts, cluster_n
    d2 = np.sum((medoid_pts[:, None, :] - medoid_pts[None, :, :]) ** 2, axis=2)
    medoid_idx = int(np.argmin(np.sum(d2, axis=1)))
    return medoid_pts[medoid_idx : medoid_idx + 1], cluster_n


def assign_depths(
    query_uv: np.ndarray,
    seeds: list[SeedPoint],
    estimator: DepthEstimator,
    *,
    K: np.ndarray,
    depth_k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not seeds:
        return np.zeros(query_uv.shape[0], dtype=np.float64)
    if estimator == "nearest":
        return assign_depth_nearest(query_uv, seeds)
    if estimator == "knn_median":
        return assign_depth_knn_median(query_uv, seeds, depth_k)
    if estimator == "idw":
        return assign_depth_idw(query_uv, seeds, depth_k)
    return assign_depth_plane_ransac(query_uv, seeds, K, rng)


def lidar_xyz_from_orig_idx(orig_idx: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    idx = np.asarray(orig_idx, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(xyz, dtype=np.float64)[idx]


def _paint_prescore_seeds_instance(
    *,
    ann_id: int,
    modality: str,
    pre_val: InstanceSeedPreScore,
    xyz: np.ndarray,
    cfg: VirtualPaintConfig,
    ground_plane: GlobalGroundPlane | None,
    ground_fallback_z: float,
    gt_annotations: list[dict[str, Any]] | None,
) -> tuple[np.ndarray, InstancePaintStats] | None:
    """Fast path: reuse batched MLP scores and emit painted LiDAR points directly."""
    arrays, probs = pre_val
    u, v, z, oi = arrays
    keep = filter_scored_indices(
        probs,
        threshold=float(cfg.seed_scorer_threshold),
        min_seeds=int(cfg.min_seeds),
        use_topk=False,
    )
    if keep is None or not np.any(keep):
        if not cfg.seed_scorer_topk_fallback:
            return None
        keep = filter_scored_indices(
            probs,
            threshold=float(cfg.seed_scorer_threshold),
            min_seeds=int(cfg.min_seeds),
            use_topk=True,
        )
        if keep is None or not np.any(keep):
            return None

    z_kept = z[keep]
    oi_kept = oi[keep]
    xyz_virt = lidar_xyz_from_orig_idx(oi_kept, xyz)
    pre_oracle_n = int(xyz_virt.shape[0])
    xyz_virt = filter_virtual_candidates(
        xyz_virt,
        max_range_m=cfg.max_range_m,
        exclude_ground=cfg.exclude_ground_points,
        ground_plane=ground_plane,
        ground_clearance_m=cfg.ground_clearance_m,
        ground_fallback_z=ground_fallback_z,
    )
    median_depth = float(np.median(z_kept)) if z_kept.size else None
    skip_final: str | None = None
    if xyz_virt.shape[0] == 0:
        if pre_oracle_n == 0:
            skip_final = (
                "all virtual points on/near ground plane"
                if cfg.exclude_ground_points
                else "no virtual points kept"
            )
        else:
            skip_final = "no virtual points kept"

    stats = InstancePaintStats(
        ann_id=ann_id,
        modality=modality,
        n_seeds_raw=int(z.size),
        n_seeds_kept=int(z_kept.size),
        n_virtual=int(xyz_virt.shape[0]),
        skipped_reason=skip_final,
        median_seed_depth_m=median_depth,
        n_candidates=0,
    )
    return xyz_virt, stats


def paint_one_modality(
    xyz: np.ndarray,
    projection: SweepProjection,
    instance_masks: list,
    *,
    T_lc: np.ndarray,
    K: np.ndarray,
    cfg: VirtualPaintConfig,
    modality: str,
    rng: np.random.Generator,
    gt_annotations: list[dict[str, Any]] | None = None,
    ann_points_per_instance: dict[int, int] | None = None,
    ann_dense_sample_cap: dict[int, int] | None = None,
    neighbor_projections: list[SweepProjection] | None = None,
    intensity: np.ndarray | None = None,
    context_masks: list | None = None,
    xmod: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    frame_pre_scored: dict[tuple[str, int], InstanceSeedPreScore] | None = None,
    ground_plane: GlobalGroundPlane | None = None,
    ground_fallback_z: float | None = None,
) -> tuple[np.ndarray, list[InstancePaintStats]]:
    """Paint scorer-kept LiDAR seeds for each YOLO instance mask (no synthetic points)."""
    del projection, T_lc, K, rng, gt_annotations, ann_points_per_instance
    del ann_dense_sample_cap, neighbor_projections, intensity, context_masks, xmod

    if ground_plane is None and cfg.exclude_ground_points:
        ground_plane = fit_frame_ground_plane(xyz)
    if ground_fallback_z is None:
        ground_fallback_z = fallback_ground_z(xyz)

    pre_scored: dict[int, InstanceSeedPreScore] = {}
    if frame_pre_scored:
        for (mod_k, ann_k), val in frame_pre_scored.items():
            if mod_k == modality:
                pre_scored[int(ann_k)] = val

    virtual_pts: list[np.ndarray] = []
    stats: list[InstancePaintStats] = []
    for entry in instance_masks:
        ann_id = int(entry[0])
        pre_val = pre_scored.get(ann_id)
        if pre_val is None:
            stats.append(
                InstancePaintStats(
                    ann_id=ann_id,
                    modality=modality,
                    n_seeds_raw=0,
                    n_seeds_kept=0,
                    n_virtual=0,
                    skipped_reason="missing precomputed scorer result",
                    median_seed_depth_m=None,
                )
            )
            continue
        painted = _paint_prescore_seeds_instance(
            ann_id=ann_id,
            modality=modality,
            pre_val=pre_val,
            xyz=xyz,
            cfg=cfg,
            ground_plane=ground_plane,
            ground_fallback_z=float(ground_fallback_z),
            gt_annotations=None,
        )
        if painted is None:
            stats.append(
                InstancePaintStats(
                    ann_id=ann_id,
                    modality=modality,
                    n_seeds_raw=0,
                    n_seeds_kept=0,
                    n_virtual=0,
                    skipped_reason="scorer rejected all seeds",
                    median_seed_depth_m=None,
                )
            )
            continue
        xyz_virt, st = painted
        stats.append(st)
        if xyz_virt.shape[0] > 0:
            virtual_pts.append(xyz_virt)

    if virtual_pts:
        return np.vstack(virtual_pts), stats
    return np.zeros((0, 3), dtype=np.float64), stats

def dedup_virtual_points(xyz: np.ndarray, voxel_m: float) -> np.ndarray:
    """Voxel-grid deduplication keeping one point per cell."""
    if xyz.shape[0] == 0 or voxel_m <= 0:
        return xyz
    keys = np.floor(xyz / float(voxel_m)).astype(np.int64)
    _, uniq_idx = np.unique(keys, axis=0, return_index=True)
    return xyz[np.sort(uniq_idx)]


def build_feature_matrix(
    xyz: np.ndarray,
    intensity: np.ndarray,
    is_virtual: np.ndarray,
    layout: FeatureLayout,
) -> np.ndarray:
    is_v = np.asarray(is_virtual, dtype=np.float32).reshape(-1, 1)
    if layout == "xyz+is_virtual":
        return np.concatenate([xyz.astype(np.float32), is_v], axis=1)
    inten = np.asarray(intensity, dtype=np.float32).reshape(-1, 1)
    if layout in ("xyz+intensity+is_virtual", "xyz+intensity+person_score"):
        return np.concatenate([xyz.astype(np.float32), inten, is_v], axis=1)
    raise ValueError(f"Unsupported feature layout: {layout}")


def _build_xmod_from_paint_plans(
    paint_plans: list[dict[str, Any]],
    n_lidar: int,
    *,
    only_targets: list[str] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Cross-modal membership using already-built sweep projections."""
    proj_masks: dict[str, tuple[SweepProjection, list]] = {
        str(p["modality"]): (p["projection"], p["masks"]) for p in paint_plans
    }
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    target_set = {str(m) for m in only_targets} if only_targets else None
    for mod in proj_masks:
        if target_set is not None and mod not in target_set:
            continue
        others = [(o, pm) for o, pm in proj_masks.items() if o != mod]
        if not others:
            continue
        conf = np.full(n_lidar, np.nan, dtype=np.float64)
        boundary = np.full(n_lidar, np.nan, dtype=np.float64)
        count = np.zeros(n_lidar, dtype=np.float64)
        for _o, (proj, masks) in others:
            c2, b2, n2 = build_cross_modal_membership(proj, masks, n_lidar)
            count += n2
            take = ~np.isnan(c2) & (np.isnan(conf) | (c2 > conf))
            conf[take] = c2[take]
            takeb = ~np.isnan(b2) & (np.isnan(boundary) | (b2 > boundary))
            boundary[takeb] = b2[takeb]
        out[mod] = (conf, boundary, count)
    return out


def augment_point_cloud(
    project_root: Path,
    subset: str,
    frame_idx: int,
    cfg: VirtualPaintConfig,
    *,
    cache: SequencePaintCache | None = None,
    matched_ann_ids: dict[str, set[int]] | None = None,
    ann_points_per_instance: dict[int, int] | None = None,
    ann_dense_sample_cap: dict[int, int] | None = None,
    inline_masks_by_modality: dict[str, list[tuple[int, np.ndarray, float]]] | None = None,
    device: str | None = None,
) -> tuple[VirtualPaintResult | None, str | None]:
    """YOLO26-seg masks (inline) → fused MLP seed painting → 5-D point features."""
    del matched_ann_ids, ann_points_per_instance, ann_dense_sample_cap
    from src.gpu_paint import gpu_paint_enabled, parse_paint_device

    if inline_masks_by_modality is None:
        return None, "inline YOLO masks are required"

    clear_mask_geometry_cache()
    clear_projection_uv_cache()
    use_gpu = gpu_paint_enabled(device, cfg)
    paint_device = parse_paint_device(device)
    cache = cache or SequencePaintCache.load(project_root, subset)

    links = cache.links
    if frame_idx < 0 or frame_idx >= len(links):
        return None, "Invalid frame index."
    link = links[frame_idx]
    by_token = cache.by_token
    calibs = cache.calibs

    try:
        lid_s = by_token[link["lidar_token"]]
        rgb_s = by_token[link["rgb_token"]]
        th_s = by_token[link["thermal_token"]]
    except KeyError as exc:
        return None, f"Missing sample token: {exc}"

    lid_cal = calibs.get(str(lid_s.get("calibrated_sensor_token") or ""))
    rgb_cal = calibs.get(str(rgb_s.get("calibrated_sensor_token") or ""))
    th_cal = calibs.get(str(th_s.get("calibrated_sensor_token") or ""))
    if not lid_cal or not rgb_cal or not th_cal:
        return None, "Missing sensor calibration."

    try:
        xyz_real, intensity_real = load_frame_lidar_cached(cache, frame_idx)
    except (OSError, ValueError) as exc:
        return None, str(exc)

    rng = np.random.default_rng(cfg.rng_seed)
    modalities = ("rgb", "thermal")
    warnings: list[str] = []
    paint_plans: list[dict[str, Any]] = []

    for mod in modalities:
        masks = list(inline_masks_by_modality.get(mod) or [])
        if not masks:
            warnings.append(f"Skipping {mod}: no YOLO masks on this frame.")
            continue
        cam_cal = rgb_cal if mod == "rgb" else th_cal
        sample = rgb_s if mod == "rgb" else th_s
        view, verr = get_rectified_view(cache, sample, cam_cal)
        if verr or view is None:
            warnings.append(f"Skipping {mod}: {verr or 'rectified view failed'}.")
            continue
        T_lc = lidar_to_camera_from_calibrations(lid_cal, cam_cal)
        projection = project_sweep_to_image(
            xyz_real,
            T_lc,
            view.K,
            view.h_rect,
            view.w_rect,
            max_range_m=float(cfg.max_range_m),
        )
        paint_plans.append(
            dict(
                modality=mod,
                masks=masks,
                masks_all=masks,
                projection=projection,
                T_lc=T_lc,
                K=view.K,
            )
        )

    if not paint_plans:
        detail = " ".join(warnings) if warnings else "No camera modality could be used."
        return None, detail

    xmod_by_mod: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if cfg.seed_scorer_model:
        for plan in paint_plans:
            for entry in plan["masks"]:
                _mask_geometry(np.asarray(entry[1], dtype=bool))
        if len(paint_plans) > 1 and cfg.seed_scorer_thermal_from_rgb:
            xmod_by_mod = _build_xmod_from_paint_plans(
                paint_plans,
                int(xyz_real.shape[0]),
                only_targets=["rgb"],
            )

    frame_ground_plane = fit_frame_ground_plane(xyz_real) if cfg.exclude_ground_points else None
    frame_ground_fallback_z = fallback_ground_z(xyz_real)
    scorer_plans = [
        dict(
            modality=p["modality"],
            projection=p["projection"],
            instance_masks=p["masks"],
            context_masks=p["masks_all"],
        )
        for p in paint_plans
    ]
    frame_pre_scored = (
        _precompute_seed_scorer_frame(
            cfg=cfg,
            xyz=xyz_real,
            intensity=intensity_real,
            modality_plans=scorer_plans,
            xmod_by_mod=xmod_by_mod,
            score_device=paint_device if use_gpu else None,
            ground_plane=frame_ground_plane,
            ground_fallback_z=frame_ground_fallback_z,
            paint_plans_for_xmod=paint_plans if len(paint_plans) > 1 else None,
        )
        if cfg.seed_scorer_model
        else {}
    )

    all_virtual: list[np.ndarray] = []
    all_stats: list[InstancePaintStats] = []
    modality_runs: list[str] = []
    for plan in paint_plans:
        xyz_virt, inst_stats = paint_one_modality(
            xyz_real,
            plan["projection"],
            plan["masks"],
            T_lc=plan["T_lc"],
            K=plan["K"],
            cfg=cfg,
            modality=plan["modality"],
            rng=rng,
            frame_pre_scored=frame_pre_scored or None,
            ground_plane=frame_ground_plane,
            ground_fallback_z=frame_ground_fallback_z,
        )
        modality_runs.append(str(plan["modality"]))
        all_stats.extend(inst_stats)
        if xyz_virt.shape[0] > 0:
            all_virtual.append(xyz_virt)

    if all_virtual:
        xyz_virtual = np.vstack(all_virtual)
        if len(modality_runs) > 1:
            xyz_virtual = dedup_virtual_points(xyz_virtual, cfg.dedup_voxel_m)
    else:
        xyz_virtual = np.zeros((0, 3), dtype=np.float64)

    n_real = xyz_real.shape[0]
    n_virt = xyz_virtual.shape[0]
    xyz_all = np.vstack([xyz_real, xyz_virtual]) if n_virt > 0 else xyz_real.copy()
    is_virtual = np.zeros(n_real + n_virt, dtype=np.uint8)
    if n_virt > 0:
        is_virtual[n_real:] = 1
    intensity_all = np.concatenate([intensity_real, np.zeros(n_virt, dtype=np.float64)])
    features = build_feature_matrix(xyz_all, intensity_all, is_virtual, cfg.feature_layout)

    return (
        VirtualPaintResult(
            xyz_real=xyz_real,
            xyz_virtual=xyz_virtual,
            features=features,
            is_virtual=is_virtual,
            per_instance=all_stats,
            modality_runs=modality_runs,
            warnings=warnings,
        ),
        None,
    )

