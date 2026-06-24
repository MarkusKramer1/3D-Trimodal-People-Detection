# Copyright (c) OpenMMLab. All rights reserved.
"""PointPillar detector used for the pedestrian paper experiments."""

from mmdet3d.models.detectors import VoxelNet
from mmdet3d.registry import MODELS


@MODELS.register_module()
class VoxelNetWithValLoss(VoxelNet):
    """PointPillars / VoxelNet wrapper (same architecture as ``VoxelNet``)."""
