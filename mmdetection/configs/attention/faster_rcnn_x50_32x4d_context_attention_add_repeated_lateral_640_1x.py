_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    './_base_/dataset_640.py',
    './_base_/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    pretrained='open-mmlab://resnext50_32x4d',
    backbone=dict(
        type='ResNeXt',
        depth=50,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
    neck=dict(
        type='Attention',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        attention_type='context',
        fusion_types=['channel_add'],
        add_extra_convs='on_lateral',
        repeated_layer=2
    )
)
