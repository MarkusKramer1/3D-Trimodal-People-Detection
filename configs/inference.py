"""Inference config for paper PointPillar pedestrian model (epoch 200)."""

custom_imports = dict(imports=[
    'projects.PointPillar.models.VoxelNetWithValLoss',
    'projects.PointPillar.transforms.normalize_intensity',
    'projects.PointPillar.datasets.MyCustomDataset',
])

default_scope = 'mmdet3d'

class_names = ['Pedestrian']
metainfo = dict(classes=['Pedestrian'])

point_cloud_range = [-30.0, -39.68, -6, -0.72, 39.68, 6]

model = dict(
    type='VoxelNetWithValLoss',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            max_voxels=(16000, 40000),
            point_cloud_range=point_cloud_range,
            voxel_size=[0.16, 0.16, 12.0],
        ),
    ),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[16],
        with_distance=False,
        voxel_size=[0.16, 0.16, 12.0],
        point_cloud_range=point_cloud_range,
    ),
    middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=16,
        output_shape=[496, 192],
    ),
    backbone=dict(
        type='SECOND',
        in_channels=16,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[16, 32, 64],
    ),
    neck=dict(
        type='SECONDFPN',
        in_channels=[16, 32, 64],
        out_channels=[16, 16, 16],
        upsample_strides=[1, 2, 4],
    ),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        in_channels=48,
        feat_channels=48,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[point_cloud_range],
            sizes=[[0.55, 0.65, 1.75], [0.65, 0.9, 1.1], [0.7, 1.7, 0.5]],
            rotations=[0.0, 1.57],
            reshape_out=False,
        ),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss',
            beta=0.1111111111111111,
            loss_weight=2.0,
        ),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.2,
        ),
    ),
    train_cfg=dict(
        assigner=dict(
            type='Max3DIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.35,
            neg_iou_thr=0.2,
            min_pos_iou=0.2,
            ignore_iof_thr=-1,
        ),
        allowed_border=0,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=100,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        max_num=50,
    ),
)

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3],
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points']),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=dict(
        type='MyCustomDataset',
        data_root='.',
        ann_file='',
        box_type_3d='LiDAR',
        metainfo=metainfo,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)
