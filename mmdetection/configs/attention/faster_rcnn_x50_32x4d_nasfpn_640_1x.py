_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    './_base_/dataset_640.py',
    './_base_/schedule_1x.py', '../_base_/default_runtime.py'
]
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://resnext50_32x4d',
    neck=dict(
        type='NASFPN',
        # in_channels=[256, 512, 1024, 2048],
        # out_channels=256,
        # start_level=1,s
        # num_outs=5,
        # add_extra_convs='on_input',
        stack_times=7,
        norm_cfg=norm_cfg
    ),
    backbone=dict(
        type='ResNeXt',
        depth=50,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch')
)
