# Copyright (c) OpenMMLab. All rights reserved.
"""TriSAR-3D dataset registration for the MMDetection3D configs in this repo."""

from mmdet3d.datasets import Det3DDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class TriSAR3DDataset(Det3DDataset):
    """Pedestrian infos in the MMDetection3D v2 format.

    LiDAR-only bins have 5 float32 columns (x, y, z, intensity, unused).
    PointPainting bins append one segmentation score, or two when both
    cameras are painted. Occlusion is a filter on which boxes enter the
    info files, not a second category.
    """

    METAINFO = {
        'classes': ('Pedestrian',),
        'palette': [(0, 255, 0)],
    }
