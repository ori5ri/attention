_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    './_base_/dataset.py',
    './_base_/schedule_2x.py', '../_base_/default_runtime.py'
]

model = dict(
    neck=dict(
        type='Attention',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        stacking=2,
        num_outs=5,
        residual_map = True
        )
)