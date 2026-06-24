"""Ground plane fitting for seed height features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GlobalGroundPlane:
    """Ground as z = a·x + b·y + c in the lidar frame."""

    a: float
    b: float
    c: float

    def z_at(self, x: float, y: float) -> float:
        return float(self.a * x + self.b * y + self.c)


def fit_global_ground_plane(
    points: np.ndarray,
    *,
    z_percentile: float = 5.0,
    min_points: int = 200,
    max_fit_points: int = 0,
) -> GlobalGroundPlane | None:
    """Fit one ground plane from pooled low-Z LiDAR (robust percentile + plane on lowest returns)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < int(min_points):
        return None
    fit_pts = pts
    if max_fit_points > 0 and pts.shape[0] > int(max_fit_points):
        step = max(1, pts.shape[0] // int(max_fit_points))
        fit_pts = pts[::step, :][: int(max_fit_points)]
    z_pct = float(np.percentile(fit_pts[:, 2], float(z_percentile)))
    z_cut = float(np.percentile(fit_pts[:, 2], 25.0))
    low = fit_pts[fit_pts[:, 2] <= z_cut + 0.08]
    if low.shape[0] < max(50, int(min_points) // 4):
        low = fit_pts[fit_pts[:, 2] <= z_pct + 0.15]
    if low.shape[0] < 15:
        return None
    coef, _, _, _ = np.linalg.lstsq(
        np.c_[low[:, 0], low[:, 1], np.ones(low.shape[0], dtype=np.float64)],
        low[:, 2],
        rcond=None,
    )
    if coef.size < 3 or not np.all(np.isfinite(coef)):
        return None
    return GlobalGroundPlane(a=float(coef[0]), b=float(coef[1]), c=float(coef[2]))
