# Copyright (c) OpenMMLab. All rights reserved.
"""Custom pedestrian LiDAR dataset (OpenPCDet-style bin layout)."""

from mmdet3d.datasets import Det3DDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class MyCustomDataset(Det3DDataset):
    """Pedestrian detection dataset with 5-dim bins (xyz + intensity + extra)."""

    METAINFO = {
        'classes': ('Pedestrian', 'Pedestrian.occluded'),
        'palette': [(0, 255, 0), (255, 0, 0)],
    }
