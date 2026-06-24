# Copyright (c) OpenMMLab. All rights reserved.
"""Intensity normalization used during PointPillar training."""

from __future__ import annotations

import numpy as np

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import BasePoints

INTENSITY_NORM_MAX = 255.0


def normalize_intensity_log1p(values: np.ndarray, *, norm_max: float = INTENSITY_NORM_MAX) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    denom = float(np.log1p(norm_max))
    return (np.log1p(np.maximum(v, 0.0)) / denom).astype(np.float32)


@TRANSFORMS.register_module()
class NormalizeIntensity:
    """Apply ``log1p(x) / log1p(norm_max)`` to the intensity channel."""

    def __init__(self, norm_max: float = INTENSITY_NORM_MAX) -> None:
        self.norm_max = float(norm_max)

    def __call__(self, results: dict) -> dict:
        points = results['points']
        if isinstance(points, BasePoints):
            tensor = points.tensor.numpy()
            tensor[:, 3] = normalize_intensity_log1p(tensor[:, 3], norm_max=self.norm_max)
            points.tensor = points.tensor.new_tensor(tensor)
            results['points'] = points
        else:
            arr = np.asarray(points, dtype=np.float32)
            arr[:, 3] = normalize_intensity_log1p(arr[:, 3], norm_max=self.norm_max)
            results['points'] = arr
        return results
