_base_ = [
    '../_base_/models/faster_rcnn_mobilenet_v2_fpn.py',
    './_base_/dataset_640.py',
    './_base_/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        type='MobileNetV2',
        out_indices=(2, 4, 6),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2')),
    neck=dict(
        type='Attention',
        in_channels=[32, 96, 320],
        out_channels=256,
        num_outs=4,
        attention_type='context',
        fusion_types=['channel_add'],
        add_extra_convs='on_lateral',
    )
)
